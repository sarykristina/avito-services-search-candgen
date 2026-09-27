"""
Shared data-loading / feature-building helpers used by both the offline
validation notebook and the final benchmark run, so the two paths can not
silently drift apart.
"""

import pandas as pd
from src.text_utils import build_weighted_text

# How much extra weight (as a token-repeat count) each field gets when we
# fold title / description / structured params into one bag-of-words string.
# Title is the single strongest signal for a short query ("баня на дровах"
# is almost always echoed near-verbatim in the title). Structured filter
# text (item_infm_params_text) is short but highly discriminative
# (service type / subtype), so it also gets a boost. Free-text description
# still contributes real signal -- dropping it entirely cost 0.03 Recall@50
# in offline validation -- but is the noisiest field (long, includes
# phone-call style pitches), so it is kept at the lowest weight.
#
# These specific numbers (5 / 3 / 1) were chosen by an offline grid search
# against train.parquet (see scripts/run_validation.py / README.md):
# Recall@50 rose from 0.182 (unweighted split 3/2/1) to ~0.185 at 5/3/1
# and then plateaued (7/4/1 and 10/5/1 gave no further gain), so 5/3/1 is
# the smallest weighting that reaches the plateau.
ITEM_FIELD_WEIGHTS = {
    "item_title_raw": 5,
    "item_infm_params_text": 3,
    "item_description_raw": 1,
}
QUERY_FIELD_WEIGHTS = {
    "search_query": 5,
    "search_infm_params_text": 3,
}


def normalize_query_text(s: pd.Series) -> pd.Series:
    return s.fillna("").str.lower().str.strip()


def build_item_corpus_text(items: pd.DataFrame) -> pd.Series:
    def _row_text(row):
        parts = [(row[f], w) for f, w in ITEM_FIELD_WEIGHTS.items()]
        return build_weighted_text(parts)
    return items.apply(_row_text, axis=1)


def build_query_text(queries: pd.DataFrame) -> pd.Series:
    def _row_text(row):
        parts = [(row[f], w) for f, w in QUERY_FIELD_WEIGHTS.items()]
        return build_weighted_text(parts)
    return queries.apply(_row_text, axis=1)
