import re
import pandas as pd

# Standard replacements
NAME_REPLACEMENTS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
    "&": "and",
}

ADDR_REPLACEMENTS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "flr": "floor",
    "opp": "opposite",
    "nr": "near",
}

# Compile unified word-boundary regex patterns for fast one-pass substitution
def _build_regex(repl_dict: dict):
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in repl_dict.keys()) + r")\b")
    return pattern, {k: v for k, v in repl_dict.items()}

_NAME_REGEX, _NAME_MAP = _build_regex(NAME_REPLACEMENTS)
_ADDR_REGEX, _ADDR_MAP = _build_regex(ADDR_REPLACEMENTS)
_PUNCT_REGEX = re.compile(r"[^\w\s]")
_SPACE_REGEX = re.compile(r"\s+")

def clean_series(series: pd.Series, regex_pat, mapping: dict) -> pd.Series:
    """Fast vectorized cleaning on a Pandas Series."""
    s = series.fillna("").astype(str).str.lower()
    # Replace matched whole words in a single regex sweep
    s = s.apply(lambda text: regex_pat.sub(lambda m: mapping[m.group(0)], text) if text else "")
    # Remove punctuation
    s = s.apply(lambda text: _PUNCT_REGEX.sub(" ", text) if text else "")
    # Normalize whitespaces
    s = s.apply(lambda text: _SPACE_REGEX.sub(" ", text).strip() if text else "")
    return s

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_name"] = clean_series(df["business_name"], _NAME_REGEX, _NAME_MAP)
    df["clean_address"] = clean_series(df["business_address"], _ADDR_REGEX, _ADDR_MAP)
    # Open-set country string handling
    df["country_clean"] = df["country"].fillna("unknown").astype(str).str.strip().str.upper()
    return df