"""
Turns raw BM25 scores into the final top-K candidate list per query, by
layering two extra signals learned from historical (query -> chosen item)
pairs in train.parquet:

1. Microcategory prior: if the (normalized) query text was seen before,
   look at which item_microcat_id(s) users ended up choosing for it, and
   give a multiplicative boost to same-microcategory candidates. This is
   a soft boost (never a hard filter), so it cannot hurt recall when the
   category prior is wrong or the query text is unseen -- it can only
   re-rank within the BM25 shortlist and, when combined with (2), pull in
   items that pure text overlap ranked outside the top 50.

2. Historical exact-item memorization: if the same query text previously
   led to a specific item_id that still exists in the current corpus,
   that item is forced into the candidate list (huge additive bonus, or
   appended if the BM25 pass didn't retrieve it at all). This directly
   captures "this exact query has a known good answer in the corpus"
   without ever touching benchmark labels -- the prior comes only from
   train.parquet's OTHER (fit-side) rows.

   IMPORTANT: this is only safe for query texts that map to a SMALL,
   concentrated set of historical items. A generic one-word query like
   "маникюр" has >5000 distinct historically-chosen items in train.parquet
   (one per provider per city) -- forcing all of them into the top 50
   would drown out the location/text-specific BM25 signal entirely and
   *reduce* recall (confirmed empirically: offline Recall@50 dropped from
   0.182 to 0.151 before this cap was added). We only apply the
   memorization boost when a query's historical item set has at most
   `memo_max_distinct` distinct items (80% of train query texts have <=3,
   so this still covers the large majority of "specific" queries where
   memorization is trustworthy), and additionally cap how many items we
   inject per query at `memo_top_n`.

Both boosts sit on top of the BM25 shortlist; pure text relevance is
always the fallback for queries with no (trustworthy) historical match.
"""

import numpy as np
import pandas as pd


def build_memo_prior(qtext_series, item_id_series, max_distinct=5, top_n=3):
    """Build the query-text -> historical-item memorization prior used by
    `boosted_top_k`, from (qtext_series, item_id_series) rows of a
    query-log dataframe (e.g. train.parquet).

    Only keeps query texts whose historical item set has at most
    `max_distinct` distinct items (see module docstring for why this
    matters), and within those, only the `top_n` most frequently chosen
    items. Returns a pd.Series: qtext -> list[item_id].
    """
    grouped = item_id_series.groupby(qtext_series)
    result = {}
    for qtext, items in grouped:
        counts = items.value_counts()
        if len(counts) <= max_distinct:
            result[qtext] = counts.index[:top_n].tolist()
    return pd.Series(result, dtype=object)


def rank_all(
    query_texts,
    qtext_norm_list,
    bm25_index,
    item_ids_sorted,
    item_microcat,
    qtext_to_items,
    qtext_to_microcat,
    k=50,
    alpha_microcat=0.5,
    memo_bonus=1e6,
    chunk_size=200,
):
    """Convenience wrapper: scores `query_texts` against `bm25_index` in
    memory-safe chunks (see BM25Index.score_chunked) and applies
    `boosted_top_k` to each chunk, returning the concatenated per-query
    ranked item_id lists in original order."""
    results = []
    for start in range(0, len(query_texts), chunk_size):
        chunk_texts = query_texts[start:start + chunk_size]
        chunk_qtext = qtext_norm_list[start:start + chunk_size]
        scores_chunk = next(bm25_index.score_chunked(chunk_texts, chunk_size=len(chunk_texts)))
        results.extend(boosted_top_k(
            chunk_qtext, scores_chunk, item_ids_sorted, item_microcat,
            qtext_to_items, qtext_to_microcat, k=k,
            alpha_microcat=alpha_microcat, memo_bonus=memo_bonus,
        ))
    return results


def boosted_top_k(
    qtext_norm_list,
    bm25_scores_csr,
    item_ids_sorted,
    item_microcat,
    qtext_to_items,
    qtext_to_microcat,
    k=50,
    alpha_microcat=0.5,
    memo_bonus=1e6,
):
    """
    qtext_norm_list      -- list[str], normalized query text per query row
    bm25_scores_csr       -- scipy CSR (n_queries, n_items) BM25 scores
    item_ids_sorted       -- np.array of item_id, SORTED ascending, aligned
                             with bm25_scores_csr columns
    item_microcat         -- np.array, item_microcat_id aligned with item_ids_sorted
    qtext_to_items        -- pd.Series: normalized query text -> set(item_id)
    qtext_to_microcat     -- pd.Series: normalized query text -> {microcat_id: prob}
    """
    results = []
    n = bm25_scores_csr.shape[0]
    for i in range(n):
        start, end = bm25_scores_csr.indptr[i], bm25_scores_csr.indptr[i + 1]
        cols = bm25_scores_csr.indices[start:end].copy()
        vals = bm25_scores_csr.data[start:end].astype(np.float64).copy()

        qtext = qtext_norm_list[i]

        microcat_prior = (
            qtext_to_microcat.get(qtext) if qtext in qtext_to_microcat.index else None
        )
        if microcat_prior:
            max_v = vals.max() if len(vals) else 1.0
            cand_microcats = item_microcat[cols] if len(cols) else np.array([])
            for mc, p in microcat_prior.items():
                boost_mask = cand_microcats == mc
                if boost_mask.any():
                    vals[boost_mask] += alpha_microcat * p * max_v

        memo_items = qtext_to_items.get(qtext) if qtext in qtext_to_items.index else None
        if memo_items:
            col_pos = {c: j for j, c in enumerate(cols)}
            extra_cols, extra_vals = [], []
            for it in memo_items:
                pos = np.searchsorted(item_ids_sorted, it)
                if pos < len(item_ids_sorted) and item_ids_sorted[pos] == it:
                    if pos in col_pos:
                        vals[col_pos[pos]] += memo_bonus
                    else:
                        extra_cols.append(pos)
                        extra_vals.append(memo_bonus)
            if extra_cols:
                cols = np.concatenate([cols, np.array(extra_cols)])
                vals = np.concatenate([vals, np.array(extra_vals)])

        if len(vals) > k:
            part = np.argpartition(vals, -k)[-k:]
            cols, vals = cols[part], vals[part]
        order = np.argsort(-vals)
        results.append(item_ids_sorted[cols[order]])
    return results
