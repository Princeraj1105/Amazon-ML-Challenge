import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Set
from collections import defaultdict


def generate_blocking_candidates(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    top_k: int = 25
) -> Dict[str, List[str]]:
    candidates: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_df["entity_id"]}

    countries = s1_df["country_clean"].unique()

    for country in countries:
        s1_sub = s1_df[s1_df["country_clean"] == country]
        target_sub = target_df[target_df["country_clean"] == country]

        if s1_sub.empty or target_sub.empty:
            continue

        print(f"Blocking [{country}]: S1={len(s1_sub)}, Target={len(target_sub)}")

        # Token inverted index on first meaningful word (catches variations instantly)
        token_index = defaultdict(list)
        target_records = target_sub[["entity_id", "clean_name"]].to_dict("records")
        for rec in target_records:
            words = [w for w in rec["clean_name"].split() if len(w) > 2]
            if words:
                token_index[words[0]].append(rec["entity_id"])

        s1_records = s1_sub[["entity_id", "clean_name"]].to_dict("records")
        for rec in s1_records:
            words = [w for w in rec["clean_name"].split() if len(w) > 2]
            if words and words[0] in token_index:
                candidates[rec["entity_id"]].extend(token_index[words[0]][:10])

        # Sub-word char n-grams over names (3-4 grams)
        s1_names = s1_sub["clean_name"].tolist()
        target_names = target_sub["clean_name"].tolist()
        s1_ids = s1_sub["entity_id"].tolist()
        target_ids = target_sub["entity_id"].tolist()

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            min_df=1,
            max_features=50000,
            dtype=np.float32
        )

        vectorizer.fit(target_names)
        target_mat = vectorizer.transform(target_names)
        s1_mat = vectorizer.transform(s1_names)

        k = min(top_k, target_mat.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(target_mat)

        batch_size = 20000
        for start_idx in range(0, len(s1_ids), batch_size):
            end_idx = min(start_idx + batch_size, len(s1_ids))
            batch_s1_mat = s1_mat[start_idx:end_idx]
            batch_s1_ids = s1_ids[start_idx:end_idx]

            distances, indices = nn.kneighbors(batch_s1_mat)

            for i, s1_id in enumerate(batch_s1_ids):
                valid_matches = [
                    target_ids[idx]
                    for d, idx in zip(distances[i], indices[i])
                    if d < 0.75  # Cosine distance cutoff
                ]
                candidates[s1_id].extend(valid_matches)

        # Deduplicate while preserving rank order
        for s1_id in s1_ids:
            candidates[s1_id] = list(dict.fromkeys(candidates[s1_id]))[:top_k]

    return candidates