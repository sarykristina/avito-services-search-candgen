"""
Final candidate-generation run: builds the submission `answer.csv` for
benchmark_queries.parquet against benchmark_items.parquet.

Pipeline (identical to the one validated in scripts/run_validation.py,
see also README.md for the full write-up):

  1. BM25 over a field-weighted bag-of-words text (title x3 + structured
     params x2 + description x1) built for every item in the corpus.
     Terms appearing in >40% of items (max_df=0.4) are dropped from the
     vocabulary: these turned out to be template/label boilerplate from
     item_infm_params_text ("вид услуги", "место оказания услуг", weekday
     names, etc.) that add no ranking value but make the query x item
     score matrix far too dense to keep in memory.
  2. Query text is built the same way (search_query x3 + filters x2) and
     scored against the BM25 index, in memory-bounded chunks.
  3. Two priors learned from ALL of train.parquet (never from benchmark
     labels, which do not exist) are layered on top:
       - a soft multiplicative boost for items whose microcategory matches
         the microcategory historically chosen for the same query text;
       - a hard "memorization" bonus that forces in any item_id that was
         historically chosen for the exact same query text and that still
         exists in the benchmark corpus.
  4. Top 50 items per query (by boosted score) are written to answer.csv.

Run:
    python3 scripts/generate_answer.py
"""

import sys
import time
import pandas as pd

sys.path.insert(0, ".")
from src.data_prep import build_item_corpus_text, build_query_text, normalize_query_text
from src.bm25 import BM25Index
from src.ranking import rank_all, build_memo_prior

K = 50
# Chosen from the offline validation sweep in scripts/run_validation.py
# (see README.md for the full numbers):
#   - the microcategory-prior boost consistently *hurt* Recall@50
#     (0.182 -> 0.176 as alpha increased), so it is disabled here.
#   - the historical-item memorization prior is a (near) free, slightly
#     positive addition once capped to query texts with a small
#     (<=MEMO_MAX_DISTINCT) set of historically-chosen items -- without
#     that cap it collapses recall (0.182 -> 0.150) because generic
#     one-word queries like "маникюр" have thousands of historical items
#     across the whole country.
ALPHA_MICROCAT = 0.0
MEMO_MAX_DISTINCT = 5
MEMO_TOP_N = 3
MAX_DF = 0.4
CHUNK = 200


def log(t0, msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    log(t0, "Loading data ...")
    train = pd.read_parquet("data/train.parquet",
                             columns=["search_query", "item_id", "item_microcat_id"])
    items = pd.read_parquet("data/benchmark_items.parquet").set_index("item_id").sort_index()
    queries = pd.read_parquet("data/benchmark_queries.parquet")
    log(t0, f"train={len(train)}  items={len(items)}  queries={len(queries)}")

    item_ids = items.index.to_numpy()
    item_microcat = items["item_microcat_id"].to_numpy()

    log(t0, "Building item corpus text + BM25 index ...")
    item_texts = build_item_corpus_text(items)
    bm25 = BM25Index(k1=1.5, b=0.75, min_df=2, max_df=MAX_DF).fit(item_texts)
    log(t0, f"vocab size={len(bm25.vectorizer.vocabulary_)}")

    log(t0, "Building query text ...")
    query_texts = build_query_text(queries).tolist()
    qtext_norm_list = normalize_query_text(queries["search_query"]).tolist()

    log(t0, "Building historical priors from train.parquet ...")
    train["_qtext_norm"] = normalize_query_text(train["search_query"])
    qtext_to_items = build_memo_prior(
        train["_qtext_norm"], train["item_id"],
        max_distinct=MEMO_MAX_DISTINCT, top_n=MEMO_TOP_N,
    )
    qtext_to_microcat = pd.Series(dtype=object)  # microcat boost disabled, see ALPHA_MICROCAT above

    log(t0, "Scoring + ranking (BM25 + priors, chunked) ...")
    ranked = rank_all(
        query_texts, qtext_norm_list, bm25, item_ids, item_microcat,
        qtext_to_items, qtext_to_microcat, k=K, alpha_microcat=ALPHA_MICROCAT,
        chunk_size=CHUNK,
    )
    log(t0, "done ranking")

    # Fallback for the rare query with zero lexical overlap with the whole
    # corpus (e.g. an out-of-vocabulary slang term matching no item at
    # all): rather than submit an empty row, fall back to the most
    # established (most-reviewed) items in the query's own category. This
    # has near-zero expected recall benefit but is a strictly better
    # default than an empty candidate list, and keeps every row non-empty.
    popularity_fallback = (
        items.sort_values("item_rating_reviews_count", ascending=False).index.to_numpy()
    )
    n_empty = sum(1 for r in ranked if len(r) == 0)
    if n_empty:
        log(t0, f"{n_empty} quer(y/ies) had zero candidates -- applying popularity fallback")
        cat_to_fallback = {}
        for cat, grp in items.groupby("item_category_id"):
            cat_to_fallback[cat] = grp.sort_values(
                "item_rating_reviews_count", ascending=False
            ).index.to_numpy()[:K]
        for i, r in enumerate(ranked):
            if len(r) == 0:
                cat = queries["search_category"].iloc[i]
                ranked[i] = cat_to_fallback.get(cat, popularity_fallback[:K])

    answer = pd.DataFrame({
        "query_id": queries["query_id"].tolist(),
        "answer": [" ".join(map(str, r[:K])) for r in ranked],
    })

    # sanity checks required by the task spec
    assert answer["query_id"].is_unique
    assert set(answer["query_id"]) == set(queries["query_id"])
    valid_items = set(items.index)
    for row in answer["answer"]:
        ids = row.split()
        assert len(ids) <= K
        assert len(ids) == len(set(ids))
        assert all(i in valid_items for i in ids)

    answer.to_csv("answer.csv", index=False)
    log(t0, f"Wrote answer.csv with {len(answer)} rows.")

    empty = (answer["answer"] == "").sum()
    log(t0, f"queries with 0 candidates: {empty}")


if __name__ == "__main__":
    main()
