import os
import re
import logging
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RAW_PATH = os.path.join("data", "raw_full", "full_emails.csv")
OUT_DIR = os.path.join("data", "processed_full")
RANDOM_STATE = 42
MAX_VALID_CHARS = 50_000
PREFIX_LEAKAGE_LEN = 1500


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving URLs and punctuation."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def prepare_full_dataset():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    initial_len = len(df)

    # 1. Validate labels strictly
    df = df.dropna(subset=["body", "label"]).copy()
    valid_labels = {0, 1}
    actual_labels = set(df["label"].unique())
    if not actual_labels.issubset(valid_labels):
        raise ValueError(f"Unexpected label values found: {actual_labels - valid_labels}")
    df["label"] = df["label"].astype(int)

    # 2. Clean whitespace and drop empty texts
    df["body_clean"] = df["body"].apply(clean_text)
    df = df[df["body_clean"].str.len() > 0].copy()

    # 3. Remove corrupted concatenated dumps (abnormally long records)
    before_len_filter = len(df)
    df = df[df["body_clean"].str.len() <= MAX_VALID_CHARS].copy()
    removed_corrupted = before_len_filter - len(df)
    if removed_corrupted > 0:
        logging.info(f"Removed {removed_corrupted} corrupted/oversized records (> {MAX_VALID_CHARS} chars).")

    # 4. Remove conflicting duplicates (same text with different labels)
    label_counts = df.groupby("body_clean")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    if conflicting_texts:
        before_conflict = len(df)
        df = df[~df["body_clean"].isin(conflicting_texts)].copy()
        logging.info(
            f"Removed {before_conflict - len(df)} rows across {len(conflicting_texts)} conflicting duplicate groups."
        )

    # 5. Deduplicate exact texts before splitting
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["body_clean"], keep="first").reset_index(drop=True)
    logging.info(f"Removed {before_dedup - len(df)} exact duplicates. Remaining: {len(df)} / {initial_len}")

    df["full_text"] = df["body_clean"]
    # Group by 1500-char prefix so truncated inputs (DistilBERT 256 tokens / Ollama 1500 chars) never leak across splits
    df["_prefix_group"] = df["body_clean"].str[:PREFIX_LEAKAGE_LEN].str.lower()

    # 6. Stratified Group 7-fold split (~71.4% train / ~14.3% val / ~14.3% test)
    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
    folds = list(sgkf.split(df, df["label"], groups=df["_prefix_group"]))

    val_idx = folds[0][1]
    test_idx = folds[1][1]
    train_idx = [i for f in range(2, 7) for i in folds[f][1]]

    cols = ["full_text", "body_clean", "label"]
    train_df = df.iloc[train_idx][cols].reset_index(drop=True)
    val_df = df.iloc[val_idx][cols].reset_index(drop=True)
    test_df = df.iloc[test_idx][cols].reset_index(drop=True)

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    logging.info(
        f"Saved leak-free splits to {OUT_DIR}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )


if __name__ == "__main__":
    prepare_full_dataset()