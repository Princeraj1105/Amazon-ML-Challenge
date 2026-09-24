import os
import pandas as pd
from typing import Dict, List

def export_submission_files(
    s1_all_ids: List[str],
    candidate_dict: Dict[str, List[str]],
    matched_dict: Dict[str, List[str]],
    output_dir: str = "output"
):
    os.makedirs(output_dir, exist_ok=True)
    
    cand_records = []
    match_records = []

    for s1_id in s1_all_ids:
        # Guarantee uniqueness while preserving order
        cands = list(dict.fromkeys(candidate_dict.get(s1_id, [])))
        matches = list(dict.fromkeys(matched_dict.get(s1_id, [])))

        # Hackathon rule: matches must be a strict subset of candidates
        matches = [m for m in matches if m in cands]

        cand_records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(cands)
        })
        match_records.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matches)
        })

    cand_df = pd.DataFrame(cand_records)
    match_df = pd.DataFrame(match_records)

    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    # Strictly tab-separated
    cand_df.to_csv(cand_path, sep="\t", index=False)
    match_df.to_csv(match_path, sep="\t", index=False)
    print(f"Exported:\n -> {match_path}\n -> {cand_path}")