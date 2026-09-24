import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from typing import List, Dict, Any


def extract_pair_features(
    s1_name: str,
    s1_addr: str,
    target_name: str,
    target_addr: str
) -> List[float]:
    """
    Computes a vector of fine-grained string similarity features between two records.
    """
    # Name similarities
    name_ratio = fuzz.ratio(s1_name, target_name) / 100.0
    name_partial = fuzz.partial_ratio(s1_name, target_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, target_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, target_name) / 100.0

    # Address similarities
    addr_ratio = fuzz.ratio(s1_addr, target_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, target_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, target_addr) / 100.0

    # Length and character count features
    len_diff_name = abs(len(s1_name) - len(target_name))
    len_diff_addr = abs(len(s1_addr) - len(target_addr))
    exact_name_match = 1.0 if s1_name == target_name and len(s1_name) > 0 else 0.0

    # Token overlap count
    s1_tokens = set(s1_name.split())
    target_tokens = set(target_name.split())
    token_overlap = len(s1_tokens & target_tokens) / max(1, len(s1_tokens | target_tokens))

    return [
        name_ratio,
        name_partial,
        name_token_sort,
        name_token_set,
        addr_ratio,
        addr_token_sort,
        addr_token_set,
        float(len_diff_name),
        float(len_diff_addr),
        exact_name_match,
        token_overlap,
    ]


FEATURE_NAMES = [
    "name_ratio",
    "name_partial",
    "name_token_sort",
    "name_token_set",
    "addr_ratio",
    "addr_token_sort",
    "addr_token_set",
    "len_diff_name",
    "len_diff_addr",
    "exact_name_match",
    "token_overlap",
]