import os
import json
import time
import logging
import pandas as pd
from tqdm import tqdm
import ollama
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Experiment configuration
MODEL_NAME = "qwen2.5:7b"
SAMPLE_PATH = os.path.join("results", "full_dataset", "test_sample_300.csv")
RESULTS_DIR = os.path.join("results", "full_dataset")
MAX_CHARS = 1500  # Truncate long bodies to keep inference latency bounded

ZERO_SHOT_SYSTEM_PROMPT = """You are a cybersecurity expert specializing in email phishing detection.
Analyze the provided email body and classify it as either phishing/malicious (1) or legitimate (0).
Respond strictly in JSON format with a single key "label":
- Use 1 if the email is phishing, credential harvesting, financial scam, or malicious spam.
- Use 0 if the email is legitimate business, personal, or safe automated communication.
Example output: {"label": 1}"""

FEW_SHOT_SYSTEM_PROMPT = """You are a cybersecurity expert specializing in email phishing detection.
Analyze the provided email body and classify it as either phishing/malicious (1) or legitimate (0).
Respond strictly in JSON format with a single key "label" (1 for phishing, 0 for legitimate).

Here are examples:

Example 1:
Email: "URGENT: Your account has been suspended due to unusual login activity. Click here http://192.168.1.45/verify-login immediately to restore access or your data will be deleted within 24 hours."
Output: {"label": 1}

Example 2:
Email: "Hi team, please find attached the agenda for tomorrow's quarterly review meeting at 10:00 AM in Conference Room B. Let me know if you have any questions. Best regards, Sarah."
Output: {"label": 0}

Example 3:
Email: "Congratulations! You have been selected to receive a $500 reward. Confirm your banking details and password at http://bit.ly/claim-reward-now to claim your prize."
Output: {"label": 1}

Example 4:
Email: "Your GitHub pull request #42 has been merged into main. You can review the commit diff directly in the repository."
Output: {"label": 0}

Now classify the following email."""


def get_ollama_model_size_mb(model_name: str) -> float:
    """Query local Ollama registry to retrieve the model size in megabytes."""
    try:
        models_info = ollama.list()
        models_list = getattr(models_info, "models", None) or models_info.get("models", [])
        for m in models_list:
            name = getattr(m, "model", None) or m.get("model") or m.get("name", "")
            if name == model_name or name.startswith(model_name.split(":")[0]):
                size_bytes = getattr(m, "size", None) or m.get("size", 0)
                if size_bytes:
                    return round(size_bytes / (1024 * 1024), 2)
    except Exception as e:
        logging.warning(f"Could not retrieve Ollama model size: {e}")
    return 0.0


def warmup_model(model: str):
    """Run a single dummy inference so model loading into VRAM/RAM is excluded from timing."""
    logging.info(f"Warming up Ollama model '{model}'...")
    try:
        ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
                {"role": "user", "content": "Email:\nHello, meeting is at 2 PM."},
            ],
            format="json",
            options={"temperature": 0.0},
        )
    except Exception as e:
        raise RuntimeError(f"Ollama warm-up failed. Ensure Ollama service is running and '{model}' is pulled: {e}")


def classify_email(text: str, system_prompt: str, model: str) -> tuple[int | None, float]:
    """Send a single email to Ollama and return (predicted_label_or_None, elapsed_seconds)."""
    truncated_text = str(text)[:MAX_CHARS]
    start_time = time.perf_counter()

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Email:\n{truncated_text}"},
            ],
            format="json",
            options={"temperature": 0.0},
        )
        elapsed = time.perf_counter() - start_time
        content = response["message"]["content"]
        parsed = json.loads(content)
        raw_label = parsed.get("label", None)

        if raw_label in (0, 1, "0", "1"):
            return int(raw_label), elapsed
        else:
            logging.warning(f"Invalid label value returned by LLM: {raw_label}")
            return None, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logging.warning(f"Inference/parsing error: {e}")
        return None, elapsed


def evaluate_mode(df: pd.DataFrame, mode_name: str, system_prompt: str, model: str, size_mb: float) -> tuple[dict, list]:
    """Evaluate the shared 300-email sample in Zero-Shot or Few-Shot mode."""
    logging.info(f"Starting evaluation in {mode_name} mode (Model: {model})...")
    preds = []
    times = []

    for text in tqdm(df["full_text"], desc=mode_name):
        pred, elapsed = classify_email(text, system_prompt, model)
        preds.append(pred)
        times.append(elapsed)

    # Separate valid predictions from failed/unparseable responses
    valid_pairs = [(yt, yp) for yt, yp in zip(df["label"].values, preds) if yp is not None]
    parse_errors = len(preds) - len(valid_pairs)

    if valid_pairs:
        y_true_valid, y_pred_valid = zip(*valid_pairs)
        acc = accuracy_score(y_true_valid, y_pred_valid)
        prec = precision_score(y_true_valid, y_pred_valid, zero_division=0)
        rec = recall_score(y_true_valid, y_pred_valid, zero_division=0)
        f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1]).ravel()
    else:
        acc = prec = rec = f1 = 0.0
        tn = fp = fn = tp = 0

    avg_time_ms = (sum(times) / len(times)) * 1000

    logging.info(
        f"[{mode_name}] Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | "
        f"F1: {f1:.4f} | Errors: {parse_errors} | Avg Latency: {avg_time_ms:.1f} ms/email"
    )

    metrics = {
        "model": f"Ollama ({model}) - {mode_name}",
        "eval_set": "sample_300",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "parse_errors": int(parse_errors),
        "avg_e2e_inference_ms": round(avg_time_ms, 2),
        "model_size_mb": size_mb,
    }
    return metrics, preds


def run_ollama_evaluation():
    if not os.path.exists(SAMPLE_PATH):
        raise FileNotFoundError(f"Shared sample not found at {SAMPLE_PATH}. Run full_dataset_benchmark.py first.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    sample_df = pd.read_csv(SAMPLE_PATH)
    size_mb = get_ollama_model_size_mb(MODEL_NAME)
    logging.info(f"Loaded shared sample ({len(sample_df)} emails). Ollama model size: {size_mb} MB.")

    warmup_model(MODEL_NAME)

    all_metrics = []

    # 1. Zero-shot evaluation
    zs_metrics, zs_preds = evaluate_mode(sample_df, "Zero-Shot", ZERO_SHOT_SYSTEM_PROMPT, MODEL_NAME, size_mb)
    all_metrics.append(zs_metrics)
    sample_df["pred_ollama_zero_shot"] = zs_preds

    # 2. Few-shot evaluation
    fs_metrics, fs_preds = evaluate_mode(sample_df, "Few-Shot", FEW_SHOT_SYSTEM_PROMPT, MODEL_NAME, size_mb)
    all_metrics.append(fs_metrics)
    sample_df["pred_ollama_few_shot"] = fs_preds

    # Save standalone Ollama metrics and predictions
    metrics_path = os.path.join(RESULTS_DIR, "ollama_metrics.csv")
    preds_path = os.path.join(RESULTS_DIR, "ollama_predictions.csv")
    pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
    sample_df.to_csv(preds_path, index=False)

    # Also append Ollama predictions to shared sample_300_predictions.csv if present
    shared_preds_path = os.path.join(RESULTS_DIR, "sample_300_predictions.csv")
    if os.path.exists(shared_preds_path):
        shared_df = pd.read_csv(shared_preds_path)
        shared_df["pred_ollama_zero_shot"] = zs_preds
        shared_df["pred_ollama_few_shot"] = fs_preds
        shared_df.to_csv(shared_preds_path, index=False)

    logging.info(f"Ollama metrics saved to {metrics_path}")
    logging.info(f"Ollama predictions saved to {preds_path}")


if __name__ == "__main__":
    run_ollama_evaluation()