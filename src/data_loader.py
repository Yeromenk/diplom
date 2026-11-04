import os
import re
import yaml
import logging
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_raw_dataset(dataset_name: str, raw_dir: str) -> pd.DataFrame:
    """Download a Hugging Face dataset and save a local raw copy."""
    os.makedirs(raw_dir, exist_ok=True)
    logging.info(f"Downloading dataset {dataset_name} from Hugging Face...")

    # This dataset stores all records in its train split.
    hf_ds = load_dataset(dataset_name, split="train")
    df = hf_ds.to_pandas()

    raw_path = os.path.join(raw_dir, "raw_emails.csv")
    df.to_csv(raw_path, index=False)
    logging.info(f"Raw dataset saved to {raw_path} (rows: {len(df)})")
    return df


def clean_text(text: str) -> str:
    """Normalize whitespace in email text."""
    if not isinstance(text, str):
        return ""
    # Collapse repeated whitespace and line breaks.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels and email text for PhishingEmailCuratedDatasets_Cleaned."""
    df = df.copy()

    # 1. Read binary labels without silently treating unknown values as legitimate.
    if "label_int" in df.columns:
        df["label"] = pd.to_numeric(df["label_int"], errors="coerce")
    elif "label_str" in df.columns:
        df["label"] = df["label_str"].astype("string").str.strip().str.lower().map(
            {"phishing": 1, "legitimate": 0}
        )
    else:
        raise KeyError("Missing class label column: expected label_int or label_str.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Class labels must be 0 (legitimate) or 1 (phishing), with no missing values.")
    df["label"] = df["label"].astype(int)

    # 2. Extract the email subject.
    subject_col = "email_subject" if "email_subject" in df.columns else "subject"
    df["subject_clean"] = df[subject_col].fillna("").apply(clean_text) if subject_col in df.columns else ""

    # 3. Extract actual email content; source filenames are metadata, not message text.
    body_col = None
    for col in ["body", "text", "email_body"]:
        if col in df.columns:
            body_col = col
            break

    df["body_clean"] = df[body_col].fillna("").apply(clean_text) if body_col else ""
    if not df["body_clean"].str.len().gt(0).any():
        logging.warning(
            "No email body text found. Only subjects will be used; "
            "URL models will only see URLs present in subjects."
        )

    # 4. Combine subject and body for NLP models.
    df["full_text"] = (df["subject_clean"] + " " + df["body_clean"]).str.strip()

    # 5. Remove empty messages, conflicting labels, and exact text duplicates.
    initial_len = len(df)
    df = df[df["full_text"].str.len() > 3].reset_index(drop=True)
    conflicting = df.groupby("full_text")["label"].transform("nunique") > 1
    if conflicting.any():
        logging.warning("Removing %d rows with conflicting labels for identical text.", conflicting.sum())
        df = df.loc[~conflicting]
    df = df.drop_duplicates(subset=["full_text"]).reset_index(drop=True)

    logging.info(f"Removed empty, duplicate, or conflicting messages: {initial_len - len(df)} (remaining: {len(df)})")

    return df


def make_leakage_aware_split(
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split preprocessed, deduplicated text with class stratification.
    Return (train_df, val_df, test_df). Similar messages may still span splits.
    """
    if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
        raise ValueError("test_size and val_size must be positive and sum to less than 1.")
    if df["full_text"].duplicated().any():
        raise ValueError("Remove duplicate full_text values before splitting to prevent text leakage.")

    # Reserve the test set first.
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed
    )

    # Express validation size as a fraction of the remaining data.
    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=seed
    )

    logging.info(f"Split sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def prepare_dataset_pipeline(config_path: str = "config/config.yaml"):
    """Run the dataset download and preparation pipeline."""
    cfg = load_config(config_path)

    # 1. Download raw data.
    df_raw = download_raw_dataset(cfg["data"]["hf_dataset_name"], cfg["data"]["raw_dir"])

    # 2. Clean and deduplicate messages.
    df_clean = preprocess_data(df_raw)

    # 3. Split into training, validation, and test sets.
    train_df, val_df, test_df = make_leakage_aware_split(
        df_clean,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        seed=cfg["data"]["random_seed"]
    )

    # 4. Save processed splits.
    processed_dir = cfg["data"]["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

    logging.info(f"Processed data saved to directory: {processed_dir}")


if __name__ == "__main__":
    prepare_dataset_pipeline()
