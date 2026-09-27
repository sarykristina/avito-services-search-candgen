"""
A minimal, vectorized BM25 implementation on top of scipy sparse matrices.

We don't use the `rank_bm25` PyPI package: it scores query-vs-corpus with a
pure-Python loop over every document, which is far too slow for a corpus of
~190k items x ~2.5k queries. Everything below is standard BM25 (Robertson &
Sparck Jones), just expressed as sparse-matrix algebra so it runs in
seconds via scipy/numpy instead of minutes/hours in pure Python.

Reference formula (Okapi BM25):
    score(q, d) = sum_{t in q} IDF(t) * f(t,d)*(k1+1) /
                  (f(t,d) + k1*(1 - b + b*|d|/avgdl))

    IDF(t) = log(1 + (N - n_t + 0.5) / (n_t + 0.5))

where f(t,d) is the raw term count of t in document d, |d| is document
length, avgdl the average document length, N the number of documents and
n_t the number of documents containing t.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, min_df: int = 2, max_df: float = 1.0):
        self.k1 = k1
        self.b = b
        # max_df prunes very common tokens (e.g. domain-wide words like
        # "услуга"/"работа") from the vocabulary entirely. This matters a
        # lot more here than in typical TF-IDF use: without it, the
        # query x item score matrix (query_terms @ item_terms.T) stops
        # being meaningfully sparse -- almost every item shares at least
        # one common word with almost every query, so the sparse dot
        # product materializes a near-fully-dense result and blows up
        # memory (observed >10GB RSS on the full item corpus). BM25's own
        # IDF term would eventually down-weight these tokens too, but
        # that only fixes the *score*, not the *memory blow-up*, since
        # the matrix stays structurally dense either way.
        self.vectorizer = CountVectorizer(min_df=min_df, max_df=max_df, dtype=np.float32)
        self.bm25_matrix = None   # (n_docs, n_terms), BM25 document weights
        self.idf_ = None          # (n_terms,)

    def fit(self, doc_texts):
        """Fit vocabulary + BM25 weights on the item corpus."""
        tf = self.vectorizer.fit_transform(doc_texts).tocsr()
        n_docs, n_terms = tf.shape

        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean()

        # document frequency per term = number of docs with a nonzero entry
        df = np.diff(tf.tocsc().indptr)
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        self.idf_ = idf.astype(np.float32)

        # length-normalization factor per document, broadcast to every
        # nonzero of that row via CSR row pointers
        len_norm = (1.0 - self.b + self.b * doc_len / avgdl).astype(np.float32)
        row_of_nnz = np.repeat(np.arange(n_docs), np.diff(tf.indptr))

        f = tf.data
        denom = f + self.k1 * len_norm[row_of_nnz]
        tf_weight = f * (self.k1 + 1.0) / denom
        bm25_data = tf_weight * self.idf_[tf.indices]

        self.bm25_matrix = sparse.csr_matrix(
            (bm25_data, tf.indices, tf.indptr), shape=tf.shape
        )
        return self

    def transform_query(self, query_texts):
        """Raw term-count vectors for queries, in the corpus vocabulary."""
        return self.vectorizer.transform(query_texts).tocsr()

    def score(self, query_texts):
        """Return sparse (n_queries, n_docs) BM25 score matrix.

        Only safe to call for a small number of queries / a pruned
        vocabulary -- see `score_chunked` for the memory-safe version used
        on the real (~190k-item) corpus."""
        q = self.transform_query(query_texts)
        return q @ self.bm25_matrix.T

    def score_chunked(self, query_texts, chunk_size: int = 200):
        """Score queries against the corpus in chunks, yielding one CSR
        block at a time instead of materializing the full (n_queries x
        n_docs) product at once.

        Even after max_df pruning, a handful of moderately common terms
        shared between a chunk of queries and the ~190k-item corpus can
        still produce a fairly dense intermediate matrix. Chunking bounds
        peak memory to O(chunk_size * n_docs) instead of
        O(n_queries * n_docs), which is what let this run comfortably
        without exhausting RAM (we saw >10GB RSS before adding this).
        """
        q_all = self.transform_query(query_texts)
        n = q_all.shape[0]
        for start in range(0, n, chunk_size):
            q_chunk = q_all[start:start + chunk_size]
            yield q_chunk @ self.bm25_matrix.T


def top_k_per_row(score_matrix: sparse.csr_matrix, k: int):
    """For a sparse (n_rows, n_cols) score matrix, return, for every row,
    the column indices of its k highest values (descending), without ever
    densifying the whole matrix."""
    score_matrix = score_matrix.tocsr()
    n_rows = score_matrix.shape[0]
    results = []
    for i in range(n_rows):
        start, end = score_matrix.indptr[i], score_matrix.indptr[i + 1]
        cols = score_matrix.indices[start:end]
        vals = score_matrix.data[start:end]
        if len(vals) > k:
            part = np.argpartition(vals, -k)[-k:]
            cols, vals = cols[part], vals[part]
        order = np.argsort(-vals)
        results.append(cols[order])
    return results
