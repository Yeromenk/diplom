import os
import yaml
import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class EmailDataset(Dataset):
    """Expose tokenized emails and labels as a PyTorch dataset."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(eval_pred):
    """Compute binary classification metrics for Hugging Face Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def train_distilbert(config_path: str = "config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seed = cfg["data"].get("random_seed", 42)
    # Seed before initializing the randomly initialized classification head.
    set_seed(seed)

    processed_dir = cfg["data"]["processed_dir"]
    model_name = cfg["models"].get("distilbert_name", "distilbert-base-uncased")
    results_dir = "results/distilbert"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load training and validation data.
    logging.info("Loading data for DistilBERT...")
    train_df = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_dir, "val.csv"))

    # Replace missing email text with empty strings.
    train_texts = train_df["full_text"].fillna("").tolist()
    train_labels = train_df["label"].values.tolist()
    val_texts = val_df["full_text"].fillna("").tolist()
    val_labels = val_df["label"].values.tolist()

    # 2. Tokenize emails; pad each batch to its longest sequence during training.
    logging.info(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info("Tokenizing email text...")
    train_encodings = tokenizer(train_texts, truncation=True, max_length=256)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=256)

    train_dataset = EmailDataset(train_encodings, train_labels)
    val_dataset = EmailDataset(val_encodings, val_labels)

    # 3. Initialize the model with a binary classification head.
    logging.info(f"Loading pretrained model {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4. Configure training.
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=2,
        per_device_train_batch_size=16,  # Reduce to 8 if GPU or system memory is insufficient.
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),  # Enable mixed precision on CUDA devices.
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=seed,
        data_seed=seed,
        report_to="none"  # Disable external experiment tracking.
    )

    # 5. Train and restore the checkpoint with the best validation F1.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logging.info("Training device: %s", training_args.device)
    logging.info("Starting fine-tuning...")
    trainer.train()

    # 6. Evaluate on validation data and save the best model.
    logging.info("Evaluating the best model on the validation split...")
    eval_results = trainer.evaluate()
    logging.info(f"DistilBERT validation results: {eval_results}")

    # Save the model and tokenizer together for inference.
    trainer.save_model(os.path.join(results_dir, "best_model"))
    tokenizer.save_pretrained(os.path.join(results_dir, "best_model"))

    # Save validation metrics. Reserve the test split for final model comparison.
    metrics_data = [{
        "model": "DistilBERT Fine-Tuned",
        "accuracy": eval_results.get("eval_accuracy"),
        "precision": eval_results.get("eval_precision"),
        "recall": eval_results.get("eval_recall"),
        "f1": eval_results.get("eval_f1")
    }]
    pd.DataFrame(metrics_data).to_csv("results/distilbert_metrics.csv", index=False)
    logging.info("Training complete. Validation metrics saved to results/distilbert_metrics.csv")


if __name__ == "__main__":
    train_distilbert()
