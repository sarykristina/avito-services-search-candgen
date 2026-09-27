"""
Offline validation of the candidate-generation pipeline using ONLY
train.parquet (benchmark labels are never touched).

Methodology
-----------
train.parquet is a log of (query, chosen item) pairs. We reconstruct
distinct *query instances* by grouping rows on the full set of query
features (search_query, search_location_id, search_is_delivery_search,
search_infm_params_text, search_category); all rows in a group share
those features and the group's item_ids are the "relevant set" for that
query instance (a query can have >1 relevant item, matching the task
description).

We split query instances 90/10 into FIT / EVAL. FIT plays the role of
"historical query log" (used to fit the BM25 vocabulary/weights and to
build the query-text -> item / query-text -> microcategory priors).
EVAL plays the role of the held-out benchmark queries. This mirrors the
real benchmark setup, where ~37% of benchmark query texts also appear
verbatim in train.parquet (checked directly on benchmark_queries.parquet)
-- i.e. it is realistic, not an artificially easier setting.

The item corpus used at scoring time is every unique item_id in the
*whole* train.parquet (FIT+EVAL): this plays the role of
benchmark_items.parquet (a fixed, fully-known corpus you search against;
knowing the corpus is not a label leak, only knowing which EVAL query
maps to which item would be).

The expensive one-time setup (building item texts + fitting BM25) is
cached to disk on first run so later parameter sweeps are fast.
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import build_item_corpus_text, build_query_text, normalize_query_text
from src.bm25 import BM25Index
from src.eval_utils import recall_at_k
from src.ranking import rank_all, build_memo_prior

RNG_SEED = 42
K = 50
CHUNK = 200
CACHE_PATH = Path("data/cache/validation_setup.pkl")

GROUP_KEYS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]


def log(t0, msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def build_setup(t0):
    log(t0, "Loading train.parquet ...")
    train = pd.read_parquet("data/train.parquet")
    log(t0, f"rows={len(train)}  unique items={train['item_id'].nunique()}")

    train["_qtext_norm"] = normalize_query_text(train["search_query"])
    group_id = train.groupby(GROUP_KEYS, sort=False).ngroup()
    train["_group_id"] = group_id
    n_groups = group_id.nunique()
    log(t0, f"distinct query instances={n_groups}")

    rng = np.random.default_rng(RNG_SEED)
    unique_groups = train["_group_id"].unique()
    rng.shuffle(unique_groups)
    n_eval = min(2000, int(0.1 * len(unique_groups)))
    eval_groups = set(unique_groups[:n_eval])
    fit_mask = ~train["_group_id"].isin(eval_groups)

    fit_rows = train[fit_mask]
    eval_rows = train[~fit_mask]
    log(t0, f"fit rows={len(fit_rows)}  eval rows={len(eval_rows)}  eval groups={n_eval}")

    eval_query_df = eval_rows.drop_duplicates("_group_id").set_index("_group_id")
    relevant_sets = eval_rows.groupby("_group_id")["item_id"].apply(set)
    eval_query_df = eval_query_df.loc[relevant_sets.index]

    items = train.drop_duplicates("item_id").set_index("item_id").sort_index()
    item_ids = items.index.to_numpy()
    item_microcat = items["item_microcat_id"].to_numpy()

    log(t0, "Building item corpus text ...")
    item_texts = build_item_corpus_text(items)
    log(t0, "done")

    log(t0, "Fitting BM25 index (max_df=0.4 to drop template/label boilerplate) ...")
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2, max_df=0.4).fit(item_texts)
    log(t0, f"vocab size={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Building eval query texts ...")
    eval_query_texts = build_query_text(eval_query_df).tolist()
    eval_qtext_list = eval_query_df["_qtext_norm"].tolist()
    true_relevant = list(relevant_sets.values)

    fit_by_qtext = fit_rows.groupby("_qtext_norm")
    qtext_to_microcat = fit_by_qtext["item_microcat_id"].agg(
        lambda s: s.value_counts(normalize=True).to_dict()
    )

    setup = dict(
        bm25=bm25, item_ids=item_ids, item_microcat=item_microcat,
        eval_query_texts=eval_query_texts, eval_qtext_list=eval_qtext_list,
        true_relevant=true_relevant, qtext_to_microcat=qtext_to_microcat,
        fit_qtext_series=fit_rows["_qtext_norm"], fit_item_series=fit_rows["item_id"],
    )
    return setup


def get_setup(t0):
    if CACHE_PATH.exists():
        log(t0, f"Loading cached setup from {CACHE_PATH} ...")
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    setup = build_setup(t0)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(setup, f)
    log(t0, f"Cached setup to {CACHE_PATH}")
    return setup


def main():
    t0 = time.time()
    s = get_setup(t0)
    log(t0, "Setup ready.")

    assert (np.sort(s["item_ids"]) == s["item_ids"]).all()

    def evaluate(alpha_microcat, memo_series, label):
        ranked = rank_all(
            s["eval_query_texts"], s["eval_qtext_list"], s["bm25"],
            s["item_ids"], s["item_microcat"], memo_series, s["qtext_to_microcat"],
            k=K, alpha_microcat=alpha_microcat, chunk_size=CHUNK,
        )
        r, _ = recall_at_k(s["true_relevant"], ranked, k=K)
        log(t0, f"[{label}]  Recall@{K} = {r:.4f}")
        return r

    empty_series = pd.Series(dtype=object)

    evaluate(0.0, empty_series, "BASELINE text-only BM25")

    for max_distinct, top_n in [(1, 1), (3, 3), (5, 3), (10, 3)]:
        memo = build_memo_prior(
            s["fit_qtext_series"], s["fit_item_series"],
            max_distinct=max_distinct, top_n=top_n,
        )
        evaluate(0.0, memo,
                 f"BM25 + memo(max_distinct={max_distinct}, top_n={top_n}), no microcat")

    # pick a good memo config, then sweep microcat alpha on top of it
    best_memo = build_memo_prior(s["fit_qtext_series"], s["fit_item_series"],
                                  max_distinct=5, top_n=3)
    for alpha in [0.0, 0.3, 0.5, 1.0, 2.0]:
        evaluate(alpha, best_memo, f"BM25 + memo(5,3) + microcat(alpha={alpha})")

    log(t0, "done")


if __name__ == "__main__":
    main()
