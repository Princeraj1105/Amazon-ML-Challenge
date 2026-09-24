import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from features import extract_pair_features, FEATURE_NAMES


def build_training_dataset(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    candidates: dict,
    gt_df: pd.DataFrame,
    max_negatives_per_entity: int = 5
) -> tuple:
    print("Indexing ground truth...")
    true_pairs = set()
    for row in gt_df[["source1_entity_id", "matched_entity_ids"]].itertuples(index=False):
        s1_id = row[0]
        raw = row[1]
        if pd.notna(raw) and raw:
            for m in str(raw).split(","):
                m_clean = m.strip()
                if m_clean:
                    true_pairs.add((s1_id, m_clean))

    print("Building fast lookups...")
    # Pure Python dictionaries for instant O(1) attribute access
    s1_names = dict(zip(s1_df["entity_id"], s1_df["clean_name"]))
    s1_addrs = dict(zip(s1_df["entity_id"], s1_df["clean_address"]))

    target_names = dict(zip(target_df["entity_id"], target_df["clean_name"]))
    target_addrs = dict(zip(target_df["entity_id"], target_df["clean_address"]))

    X_list = []
    y_list = []

    print("Extracting features from candidate pairs...")
    for s1_id, cand_list in candidates.items():
        if s1_id not in s1_names:
            continue

        s1_name = s1_names[s1_id]
        s1_addr = s1_addrs[s1_id]

        neg_count = 0
        for cand_id in cand_list:
            if cand_id not in target_names:
                continue

            is_match = 1 if (s1_id, cand_id) in true_pairs else 0

            # Subsample negatives to keep balanced ratios and save processing time
            if is_match == 0:
                if neg_count >= max_negatives_per_entity:
                    continue
                neg_count += 1

            cand_name = target_names[cand_id]
            cand_addr = target_addrs[cand_id]

            feats = extract_pair_features(s1_name, s1_addr, cand_name, cand_addr)
            X_list.append(feats)
            y_list.append(is_match)

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.empty((0,), dtype=np.int32)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def train_matching_model(X: np.ndarray, y: np.ndarray, model_save_path: str = "models/lgb_matcher.pkl"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    if len(X) == 0:
        print("Error: Feature dataset is empty. Cannot train.")
        return

    # If there are no positive matches in the subset slice, fallback gracefully
    if y.sum() == 0:
        print("Notice: No positive pairs in this subset. Creating default threshold model.")
        with open(model_save_path, "wb") as f:
            pickle.dump({"model": None, "threshold": 0.80}, f)
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
    )

    print(f"Training LightGBM on {len(X_train)} samples ({int(y_train.sum())} positive matches)...")

    clf = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=25,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )
    clf.fit(X_train, y_train)

    val_preds_prob = clf.predict_proba(X_val)[:, 1]

    # Search for the optimal threshold for F_0.5
    best_thresh = 0.5
    best_f05 = -1.0

    for thresh in np.arange(0.50, 0.95, 0.05):
        pred_labels = (val_preds_prob >= thresh).astype(int)
        tp = np.sum((pred_labels == 1) & (y_val == 1))
        fp = np.sum((pred_labels == 1) & (y_val == 0))
        fn = np.sum((pred_labels == 0) & (y_val == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            denom = (0.25 * precision) + recall
            f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
        else:
            f05 = 0.0

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"Optimal F_0.5 Threshold: {best_thresh:.2f} (Val F_0.5: {best_f05:.4f})")

    with open(model_save_path, "wb") as f:
        pickle.dump({"model": clf, "threshold": best_thresh}, f)

    print(f"Model saved to {model_save_path}")