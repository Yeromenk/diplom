import os
import re
import time
import joblib
import logging
import ipaddress
import numpy as np
import pandas as pd
from urllib.parse import urlparse

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = os.path.join("data", "processed_full")
RESULTS_DIR = os.path.join("results", "full_dataset")
ARTIFACTS_DIR = os.path.join(RESULTS_DIR, "artifacts")
SAMPLE_SIZE = 300
RANDOM_STATE = 42
BERT_MODEL_NAME = "distilbert-base-uncased"
# Use 128 tokens on CPU for practical training time (~30 min); 256 when CUDA GPU is available
MAX_LENGTH = 256 if torch.cuda.is_available() else 128

URL_REGEX = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.IGNORECASE)
SUSPICIOUS_KEYWORDS = ("login", "verify", "secure", "account", "update", "bank", "signin", "confirm")


def is_valid_ip(hostname: str) -> bool:
    """Validate IPv4 or IPv6 address strictly using the standard ipaddress module."""
    if not hostname:
        return False
    clean_host = hostname.strip("[]")
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def extract_url_features(texts: pd.Series) -> np.ndarray:
    """Extract lexical and structural URL features from raw email texts."""
    features = []
    for text in texts.astype(str):
        urls = URL_REGEX.findall(text)
        num_urls = len(urls)
        if num_urls == 0:
            features.append([0, 0, 0, 0, 0, 0, 0, 0])
            continue

        lengths = [len(u) for u in urls]
        max_len = max(lengths)
        avg_len = sum(lengths) / num_urls
        num_dots = sum(u.count(".") for u in urls)
        num_hyphens = sum(u.count("-") for u in urls)
        has_at = int(any("@" in u for u in urls))
        has_ip = 0
        has_kw = 0

        for u in urls:
            lower_u = u.lower()
            normalized = u if lower_u.startswith(("http://", "https://")) else f"http://{u}"
            try:
                host = urlparse(normalized).hostname or ""
                if is_valid_ip(host):
                    has_ip = 1
            except Exception:
                pass
            if any(kw in lower_u for kw in SUSPICIOUS_KEYWORDS):
                has_kw = 1

        features.append([num_urls, max_len, avg_len, num_dots, num_hyphens, has_at, has_ip, has_kw])

    return np.array(features, dtype=np.float32)


def get_dir_or_file_size_mb(path: str) -> float:
    """Calculate total size in megabytes for a file or directory."""
    if os.path.isfile(path):
        return round(os.path.getsize(path) / (1024 * 1024), 2)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / (1024 * 1024), 2)


def compute_metrics_row(model_name: str, split_name: str, y_true, y_pred, avg_ms: float, size_mb: float) -> dict:
    """Compute classification metrics and confusion matrix components."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "eval_set": split_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "avg_e2e_inference_ms": round(avg_ms, 4),
        "model_size_mb": size_mb,
    }


def save_progress(records: list[dict], test_preds_df: pd.DataFrame, sample_preds_df: pd.DataFrame):
    """Save metrics and row-level predictions incrementally after each model."""
    pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, "models_benchmark.csv"), index=False)
    test_preds_df.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
    sample_preds_df.to_csv(os.path.join(RESULTS_DIR, "sample_300_predictions.csv"), index=False)


def run_benchmark():
    set_seed(RANDOM_STATE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    # Fixed 300-row stratified test subset for fair comparison with Ollama
    sample_300_path = os.path.join(RESULTS_DIR, "test_sample_300.csv")
    sample_df, _ = train_test_split(
        test_df, train_size=SAMPLE_SIZE, stratify=test_df["label"], random_state=RANDOM_STATE
    )
    sample_df = sample_df.reset_index(drop=True)
    sample_df.to_csv(sample_300_path, index=False)
    logging.info(f"Saved shared 300-email test sample to {sample_300_path}")

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    y_sample = sample_df["label"].values

    test_preds_df = test_df[["full_text", "label"]].copy()
    sample_preds_df = sample_df[["full_text", "label"]].copy()
    records = []

    """1. TF-IDF + Logistic Regression (End-to-End timing)"""
    logging.info("Training TF-IDF + Logistic Regression...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(train_df["full_text"])
    clf_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_lr.fit(X_train_tfidf, y_train)

    lr_artifact = os.path.join(ARTIFACTS_DIR, "tfidf_logreg.joblib")
    joblib.dump({"vectorizer": vectorizer, "model": clf_lr}, lr_artifact)
    lr_size = get_dir_or_file_size_mb(lr_artifact)

    # Warm-up
    _ = clf_lr.predict(vectorizer.transform(test_df["full_text"].iloc[:10]))

    t0 = time.perf_counter()
    preds_lr_test = clf_lr.predict(vectorizer.transform(test_df["full_text"]))
    lr_ms_test = ((time.perf_counter() - t0) / len(test_df)) * 1000

    t0 = time.perf_counter()
    preds_lr_sample = clf_lr.predict(vectorizer.transform(sample_df["full_text"]))
    lr_ms_sample = ((time.perf_counter() - t0) / len(sample_df)) * 1000

    test_preds_df["pred_tfidf_logreg"] = preds_lr_test
    sample_preds_df["pred_tfidf_logreg"] = preds_lr_sample
    records.append(compute_metrics_row("TF-IDF + Logistic Regression", "full_test", y_test, preds_lr_test, lr_ms_test, lr_size))
    records.append(compute_metrics_row("TF-IDF + Logistic Regression", "sample_300", y_sample, preds_lr_sample, lr_ms_sample, lr_size))
    save_progress(records, test_preds_df, sample_preds_df)

    """2. URL Features + Random Forest / XGBoost (End-to-End timing)"""
    logging.info("Extracting URL features for training...")
    X_train_url = extract_url_features(train_df["full_text"])

    for name, col_key, model, fname in [
        ("URL Features + Random Forest", "pred_url_rf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), "url_rf.joblib"),
        ("URL Features + XGBoost", "pred_url_xgb", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="logloss"), "url_xgb.joblib"),
    ]:
        logging.info(f"Training {name}...")
        model.fit(X_train_url, y_train)
        art_path = os.path.join(ARTIFACTS_DIR, fname)
        joblib.dump(model, art_path)
        m_size = get_dir_or_file_size_mb(art_path)

        # Warm-up
        _ = model.predict(extract_url_features(test_df["full_text"].iloc[:10]))

        # End-to-end: raw text -> URL feature extraction -> prediction
        t0 = time.perf_counter()
        preds_test = model.predict(extract_url_features(test_df["full_text"]))
        m_ms_test = ((time.perf_counter() - t0) / len(test_df)) * 1000

        t0 = time.perf_counter()
        preds_sample = model.predict(extract_url_features(sample_df["full_text"]))
        m_ms_sample = ((time.perf_counter() - t0) / len(sample_df)) * 1000

        test_preds_df[col_key] = preds_test
        sample_preds_df[col_key] = preds_sample
        records.append(compute_metrics_row(name, "full_test", y_test, preds_test, m_ms_test, m_size))
        records.append(compute_metrics_row(name, "sample_300", y_sample, preds_sample, m_ms_sample, m_size))
        save_progress(records, test_preds_df, sample_preds_df)

    """3. Fine-tuned DistilBERT (End-to-End timing)"""
    logging.info(f"Fine-tuning DistilBERT (max_length={MAX_LENGTH}, cuda={torch.cuda.is_available()})...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["full_text"], truncation=True, max_length=MAX_LENGTH)

    ds_train = Dataset.from_pandas(train_df[["full_text", "label"]]).map(tokenize_fn, batched=True)
    ds_val = Dataset.from_pandas(val_df[["full_text", "label"]]).map(tokenize_fn, batched=True)

    bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    bert_out_dir = os.path.join(ARTIFACTS_DIR, "distilbert_full")

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1": f1_score(labels, preds, zero_division=0)}

    training_args = TrainingArguments(
        output_dir=os.path.join(RESULTS_DIR, "bert_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_STATE,
        data_seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=bert_model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=hf_metrics,
    )

    trainer.train()
    trainer.save_model(bert_out_dir)
    tokenizer.save_pretrained(bert_out_dir)
    bert_size = get_dir_or_file_size_mb(bert_out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model.to(device)
    bert_model.eval()

    def predict_bert_e2e(texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Measure pure end-to-end tokenization + forward pass without Trainer metric overhead."""
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                enc = tokenizer(
                    batch_texts,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                logits = bert_model(**enc).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.append(preds)
        return np.concatenate(all_preds, axis=0)

    # Warm-up
    _ = predict_bert_e2e(test_df["full_text"].iloc[:16].tolist())

    t0 = time.perf_counter()
    preds_bert_test = predict_bert_e2e(test_df["full_text"].tolist())
    bert_ms_test = ((time.perf_counter() - t0) / len(test_df)) * 1000

    t0 = time.perf_counter()
    preds_bert_sample = predict_bert_e2e(sample_df["full_text"].tolist())
    bert_ms_sample = ((time.perf_counter() - t0) / len(sample_df)) * 1000

    test_preds_df["pred_distilbert"] = preds_bert_test
    sample_preds_df["pred_distilbert"] = preds_bert_sample
    records.append(compute_metrics_row("DistilBERT (Fine-Tuned)", "full_test", y_test, preds_bert_test, bert_ms_test, bert_size))
    records.append(compute_metrics_row("DistilBERT (Fine-Tuned)", "sample_300", y_sample, preds_bert_sample, bert_ms_sample, bert_size))
    save_progress(records, test_preds_df, sample_preds_df)

    df_res = pd.DataFrame(records)
    logging.info(f"Benchmark completed! Final summary:\n{df_res.to_string(index=False)}")


if __name__ == "__main__":
    run_benchmark()