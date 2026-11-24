import os
import time
import joblib
import logging
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from model.full_dataset_benchmark import extract_url_features, MAX_LENGTH

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DS1_DIR = os.path.join("data", "processed")
DS2_DIR = os.path.join("data", "processed_full")
ARTIFACTS_DIR = os.path.join("results", "full_dataset", "artifacts")
OUT_PATH = os.path.join("results", "full_dataset", "cross_dataset_metrics.csv")
RANDOM_STATE = 42


def get_text_series(df: pd.DataFrame) -> pd.Series:
    """Safely extract the primary non-empty text column across both dataset schemas."""
    for col in ["full_text", "subject_clean", "subject", "text", "body_clean"]:
        if col in df.columns:
            series = df[col].fillna("").astype(str)
            if series.str.strip().str.len().sum() > 0:
                return series
    raise ValueError(f"Could not find a non-empty text column in columns: {list(df.columns)}")


def evaluate_predictions(direction: str, model_name: str, y_true, y_pred, avg_ms: float) -> dict:
    """Compute standardized classification and confusion matrix metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "direction": direction,
        "model": model_name,
        "n_samples": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "avg_e2e_inference_ms": round(avg_ms, 4),
    }


def predict_bert_batched(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Run batched inference for DistilBERT."""
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            all_preds.append(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def run_cross_dataset_eval():
    ds1_train = pd.read_csv(os.path.join(DS1_DIR, "train.csv"))
    ds1_test = pd.read_csv(os.path.join(DS1_DIR, "test.csv"))
    ds2_test = pd.read_csv(os.path.join(DS2_DIR, "test.csv"))

    X_ds1_train = get_text_series(ds1_train)
    y_ds1_train = ds1_train["label"].astype(int).values

    X_ds1_test = get_text_series(ds1_test)
    y_ds1_test = ds1_test["label"].astype(int).values

    X_ds2_test = get_text_series(ds2_test)
    y_ds2_test = ds2_test["label"].astype(int).values

    records = []

    # =========================================================================
    # Direction A: Trained on Full-Text (DS2) -> Evaluated on Subject-Only (DS1)
    # =========================================================================
    dir_a = "Train: Full-Text (DS2) -> Test: Subject-Only (DS1)"
    logging.info(f"Running {dir_a} (n={len(ds1_test)})...")

    # 1. TF-IDF + Logistic Regression
    lr_bundle = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_logreg.joblib"))
    vec_ds2, clf_lr_ds2 = lr_bundle["vectorizer"], lr_bundle["model"]
    _ = clf_lr_ds2.predict(vec_ds2.transform(X_ds1_test.iloc[:10]))

    t0 = time.perf_counter()
    preds_lr_a = clf_lr_ds2.predict(vec_ds2.transform(X_ds1_test))
    ms_lr_a = ((time.perf_counter() - t0) / len(X_ds1_test)) * 1000
    records.append(evaluate_predictions(dir_a, "TF-IDF + Logistic Regression", y_ds1_test, preds_lr_a, ms_lr_a))

    # 2. URL Features + RF / XGBoost
    for m_name, fname in [
        ("URL Features + Random Forest", "url_rf.joblib"),
        ("URL Features + XGBoost", "url_xgb.joblib"),
    ]:
        clf = joblib.load(os.path.join(ARTIFACTS_DIR, fname))
        _ = clf.predict(extract_url_features(X_ds1_test.iloc[:10]))

        t0 = time.perf_counter()
        preds = clf.predict(extract_url_features(X_ds1_test))
        ms = ((time.perf_counter() - t0) / len(X_ds1_test)) * 1000
        records.append(evaluate_predictions(dir_a, m_name, y_ds1_test, preds, ms))

    # 3. DistilBERT (Fine-tuned on DS2)
    bert_ds2_dir = os.path.join(ARTIFACTS_DIR, "distilbert_full")
    if os.path.exists(bert_ds2_dir):
        logging.info("Evaluating DS2 DistilBERT on DS1 test set...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok_ds2 = AutoTokenizer.from_pretrained(bert_ds2_dir)
        bert_ds2 = AutoModelForSequenceClassification.from_pretrained(bert_ds2_dir).to(device)
        bert_ds2.eval()

        _ = predict_bert_batched(bert_ds2, tok_ds2, X_ds1_test.iloc[:16].tolist(), device)
        t0 = time.perf_counter()
        preds_bert_a = predict_bert_batched(bert_ds2, tok_ds2, X_ds1_test.tolist(), device)
        ms_bert_a = ((time.perf_counter() - t0) / len(X_ds1_test)) * 1000
        records.append(evaluate_predictions(dir_a, "DistilBERT (Fine-Tuned)", y_ds1_test, preds_bert_a, ms_bert_a))

    # =========================================================================
    # Direction B: Trained on Subject-Only (DS1) -> Evaluated on Full-Text (DS2)
    # =========================================================================
    dir_b = "Train: Subject-Only (DS1) -> Test: Full-Text (DS2)"
    logging.info(f"Running {dir_b} (n={len(ds2_test)})...")

    # 1. TF-IDF + Logistic Regression trained on DS1
    vec_ds1 = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tfidf_ds1 = vec_ds1.fit_transform(X_ds1_train)
    clf_lr_ds1 = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf_lr_ds1.fit(X_tr_tfidf_ds1, y_ds1_train)

    _ = clf_lr_ds1.predict(vec_ds1.transform(X_ds2_test.iloc[:10]))
    t0 = time.perf_counter()
    preds_lr_b = clf_lr_ds1.predict(vec_ds1.transform(X_ds2_test))
    ms_lr_b = ((time.perf_counter() - t0) / len(X_ds2_test)) * 1000
    records.append(evaluate_predictions(dir_b, "TF-IDF + Logistic Regression", y_ds2_test, preds_lr_b, ms_lr_b))

    # 2. URL Features + RF / XGBoost trained on DS1
    X_tr_url_ds1 = extract_url_features(X_ds1_train)
    for m_name, clf in [
        ("URL Features + Random Forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ("URL Features + XGBoost", XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, eval_metric="logloss")),
    ]:
        clf.fit(X_tr_url_ds1, y_ds1_train)
        _ = clf.predict(extract_url_features(X_ds2_test.iloc[:10]))

        t0 = time.perf_counter()
        preds = clf.predict(extract_url_features(X_ds2_test))
        ms = ((time.perf_counter() - t0) / len(X_ds2_test)) * 1000
        records.append(evaluate_predictions(dir_b, m_name, y_ds2_test, preds, ms))

    # 3. Check if a DS1 DistilBERT checkpoint exists on disk
    for candidate_dir in [
        os.path.join("results", "distilbert_subject"),
        os.path.join("results", "bert_model"),
        os.path.join("models", "distilbert"),
    ]:
        if os.path.exists(os.path.join(candidate_dir, "config.json")):
            logging.info(f"Found DS1 DistilBERT checkpoint at {candidate_dir}, evaluating on DS2...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tok_ds1 = AutoTokenizer.from_pretrained(candidate_dir)
            bert_ds1 = AutoModelForSequenceClassification.from_pretrained(candidate_dir).to(device)
            bert_ds1.eval()

            _ = predict_bert_batched(bert_ds1, tok_ds1, X_ds2_test.iloc[:16].tolist(), device)
            t0 = time.perf_counter()
            preds_bert_b = predict_bert_batched(bert_ds1, tok_ds1, X_ds2_test.tolist(), device)
            ms_bert_b = ((time.perf_counter() - t0) / len(X_ds2_test)) * 1000
            records.append(evaluate_predictions(dir_b, "DistilBERT (Fine-Tuned)", y_ds2_test, preds_bert_b, ms_bert_b))
            break

    df_cross = pd.DataFrame(records)
    df_cross.to_csv(OUT_PATH, index=False)
    logging.info(f"Cross-dataset evaluation saved to {OUT_PATH}:\n{df_cross.to_string(index=False)}")


if __name__ == "__main__":
    run_cross_dataset_eval()