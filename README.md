# Phishing URL Detection Using Language Models

This thesis project benchmarks machine learning methods for detecting phishing in email. It compares text classifiers, models based on URL features, and a local language model in terms of classification quality, inference time, and stored model size. The current experiments assign a phishing or legitimate label to each email. URL features describe the links found in a message, and individual links have no ground-truth labels in these experiments.

## Thesis assignment

The aim of the thesis is to design and implement a system for detecting phishing URLs in emails using machine learning methods, particularly language models. The student will build a dataset from publicly available data, train several models on it, and compare them, focusing on the relationship between model size, inference speed, and prediction accuracy. The resulting code should be properly documented, and a version control system such as Git should be used during development.

1. Study methods for analyzing spam and fraudulent emails.
2. Build a training dataset of emails containing phishing URLs.
3. Train several different types of models on the dataset and compare them.
4. Propose how these analysis methods could be integrated into existing email clients.

## Data and models

| Dataset | Input used in this project | Role |
| --- | --- | --- |
| `it4lia/PhishingEmailCuratedDatasets_Cleaned` | Email subjects | Subject-only baseline and cross-dataset evaluation |
| `zefang-liu/phishing-email-dataset` | Full email text | Main benchmark and cross-dataset evaluation |

The benchmark trains TF-IDF with Logistic Regression, Random Forest and XGBoost on aggregated URL features, and a fine-tuned DistilBERT classifier. Qwen2.5 7B is evaluated through Ollama with zero-shot and few-shot prompts. The full-text benchmark uses a held-out test set and a stratified 300-email subset shared by all model families.

## Run the experiments

Run the commands from the repository root in a Python environment, such as a virtual environment or Conda environment. Downloading the datasets and pretrained weights requires internet access. DistilBERT can train on CPU or CUDA, and CPU training can take substantially longer.

```shell
python -m pip install -r requirements.txt
```

### Full-text benchmark

```shell
python -m src.download_full_dataset
python -m src.prepare_full_dataset
python -m model.full_dataset_benchmark
```

The first two commands regenerate the full-text CSV files under `data/raw_full/` and `data/processed_full/`, including the copies tracked in Git. If you want to evaluate the checked-in splits, start with the benchmark command.

The benchmark writes metrics, predictions, the shared test subset, and model artifacts to `results/full_dataset/`.

### Ollama evaluation

Install and start Ollama locally. Run the following commands after generating the shared test subset with the full-text benchmark.

```shell
ollama pull qwen2.5:7b
python -m model.ollama_eval
```

### Subject-only experiments and cross-dataset evaluation

```shell
python -m src.data_loader
python -m model.baseline
python -m model.bert_finetune
python -m model.cross_dataset_eval
```

The first command prepares the subject-only splits using [`config/config.yaml`](config/config.yaml). The next two train models and report validation metrics. Cross-dataset evaluation also needs the artifacts produced by the full-text benchmark. The current cross-dataset script skips the subject-trained DistilBERT checkpoint because its search paths omit `results/distilbert/best_model`.

### Notebooks and tests

Generated results are excluded from Git. Run the corresponding scripts before rerunning the notebooks.

| Notebook | Contents |
| --- | --- |
| [01 — Dataset exploration](notebooks/01_dataset_exploration.ipynb) | Subject-only splits, class balance, and text lengths |
| [02 — Model comparison](notebooks/02_model_comparison.ipynb) | Subject-only validation metrics |
| [03 — Full-text benchmark](notebooks/03_full_dataset_benchmark.ipynb) | Shared test subset, full test set, cross-dataset evaluation, and error analysis |

```shell
python -m unittest discover -s tests -v
```

## Preliminary results

These values come from the saved run on the shared 300-email subset of the full-text test set.

| Model | Phishing-class F1 | Mean time per email (ms) | Stored size (MiB) |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.9774 | 0.354 | 0.89 |
| DistilBERT (fine-tuned) | 0.9680 | 36.710 | 256.11 |
| Qwen2.5 7B (zero-shot) | 0.8942 | 1,886.63 | 4,466.14 |
| Qwen2.5 7B (few-shot) | 0.8290 | 1,936.80 | 4,466.14 |
| URL features + Random Forest | 0.0847 | 0.223 | 9.01 |
| URL features + XGBoost | 0.0684 | 0.075 | 0.36 |

TF-IDF with Logistic Regression achieved the highest F1 on this subset and on the full test set of 2,498 emails. The URL-feature models had low recall across the corpus, where only 592 of the 2,498 test emails contained an explicit URL detected by the project’s pattern.

### Evaluation limitations

- The F1 scores measure email classification. Evaluating individual links requires separately labeled URLs.
- The benchmark reports average per-email time from batched CPU inference for the trained models and sequential local API calls for Qwen. A single-email benchmark would better represent interactive client latency. Stored size measures disk usage.
- The full-text split groups by normalized 1,500-character prefixes, but some train and test examples still match after DistilBERT tokenization and truncation. The subject-only split also has overlaps after case normalization. Final metrics should be recomputed after grouping by the effective model inputs.

## Proposed email-client integration

A mail client could send the subject, body, and extracted links to a local classifier when a message arrives. The classifier would return a message-level risk score, and the client could show a warning when it exceeds a threshold selected on validation data. The current benchmark supports choosing a candidate model. Highlighting specific suspicious links would require a separate URL-level predictor and evaluation.
