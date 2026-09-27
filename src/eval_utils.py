"""Recall@K helper for offline validation against train.parquet."""

import numpy as np


def recall_at_k(list_of_relevant_sets, list_of_ranked_item_ids, k=50):
    """
    list_of_relevant_sets[i]   -- set of item_ids that are truly relevant
                                   for query i
    list_of_ranked_item_ids[i] -- our ranked candidate item_ids for query i
                                   (already truncated/ordered, length <= k)
    Returns the mean per-query recall, matching the competition metric:
        mean_i( |topK_i ∩ relevant_i| / |relevant_i| )
    """
    scores = []
    for relevant, ranked in zip(list_of_relevant_sets, list_of_ranked_item_ids):
        if not relevant:
            continue
        topk = set(ranked[:k])
        scores.append(len(topk & relevant) / len(relevant))
    return float(np.mean(scores)), scores
