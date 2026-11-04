import logging
import os

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from src.features import TextFeatureExtractor, extract_url_features_for_df

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def evaluate_model(name: str, y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    logging.info(f"--- {name} ---")
    logging.info(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

def run_baselines(config_path: str = "config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    processed_dir = cfg["data"]["processed_dir"]
    seed = cfg["data"].get("random_seed", 42)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    all_metrics = []

    # 1. TF-IDF + Logistic Regression
    logging.info("Training TF-IDF + Logistic Regression...")
    tfidf = TextFeatureExtractor(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(train_df["full_text"])
    X_val_tfidf = tfidf.transform(val_df["full_text"])

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_val_tfidf)
    all_metrics.append(evaluate_model("TF-IDF + LogisticRegression", y_val, y_pred_lr))

    # 2. URL features + tree models.
    logging.info("Extracting URL features...")
    X_train_url = extract_url_features_for_df(train_df)
    X_val_url = extract_url_features_for_df(val_df)
    logging.info(
        "Emails containing URLs: train=%d/%d, validation=%d/%d",
        X_train_url["num_urls"].gt(0).sum(), len(X_train_url),
        X_val_url["num_urls"].gt(0).sum(), len(X_val_url),
    )
    if not X_train_url["num_urls"].gt(0).any():
        logging.warning("No training URLs found; URL models can only learn a constant prediction.")

    logging.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train_url, y_train)
    all_metrics.append(evaluate_model("URL + Random Forest", y_val, rf.predict(X_val_url)))

    logging.info("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=seed, n_jobs=-1)
    xgb.fit(X_train_url, y_train)
    all_metrics.append(evaluate_model("URL + XGBoost", y_val, xgb.predict(X_val_url)))

    # Save validation metrics. Keep the test split for final model comparison.
    df_res = pd.DataFrame(all_metrics)
    df_res.to_csv(os.path.join(results_dir, "baseline_metrics.csv"), index=False)
    logging.info(f"Validation metrics saved to {results_dir}/baseline_metrics.csv")

if __name__ == "__main__":
    run_baselines()
