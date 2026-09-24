import pandas as pd
import numpy as np

def compute_entity_f05(pred_set: set, true_set: set) -> float:
    # Singletons: entity has no true matches
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision == 0.0 and recall == 0.0:
        return 0.0

    # Macro F_0.5 weighting: precision weighted 2x over recall
    beta_sq = 0.5 ** 2  # 0.25
    denom = beta_sq * precision + recall
    if denom == 0.0:
        return 0.0
    return ((1.0 + beta_sq) * precision * recall) / denom

def score_predictions(ground_truth_df: pd.DataFrame, preds_df: pd.DataFrame) -> float:
    merged = pd.merge(
        ground_truth_df, preds_df, on="source1_entity_id", how="left"
    )
    scores = []
    for _, row in merged.iterrows():
        gt_raw = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        true_matches = set(x.strip() for x in gt_raw.split(",") if x.strip())

        pred_raw = str(row.get("matched_entity_ids_pred", "")) if pd.notna(row.get("matched_entity_ids_pred", "")) else ""
        pred_matches = set(x.strip() for x in pred_raw.split(",") if x.strip())

        scores.append(compute_entity_f05(pred_matches, true_matches))
    return float(np.mean(scores))