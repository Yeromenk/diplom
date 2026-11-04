import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import issparse
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, DistilBertConfig, DistilBertForSequenceClassification

from model.bert_finetune import train_distilbert
from src.data_loader import make_leakage_aware_split, preprocess_data
from src.features import TextFeatureExtractor, extract_url_features_for_df, extract_url_features_from_string, extract_urls


class DataTests(unittest.TestCase):
    def test_source_filename_is_excluded_and_empty_subject_is_removed(self):
        raw = pd.DataFrame({
            "email_subject": [" Hello world ", None],
            "email_source_file": ["phishing.csv", "legitimate.csv"],
            "label_int": [1, 0],
        })
        result = preprocess_data(raw)
        self.assertEqual(result["full_text"].tolist(), ["Hello world"])

    def test_unknown_labels_are_rejected(self):
        for column, value in [("label_int", 2), ("label_int", None), ("label_str", "unknown")]:
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                preprocess_data(pd.DataFrame({column: [value], "body": ["Message text"]}))

    def test_conflicting_labels_and_duplicates(self):
        raw = pd.DataFrame({
            "body": ["conflicting text", "conflicting text", "valid message", "valid message"],
            "label_str": ["phishing", "legitimate", " Legitimate ", "legitimate"],
        })
        result = preprocess_data(raw)
        self.assertEqual(result["full_text"].tolist(), ["valid message"])
        self.assertEqual(result["label"].tolist(), [0])

    def test_split_is_disjoint_stratified_and_repeatable(self):
        data = pd.DataFrame({"full_text": [f"message {i}" for i in range(100)], "label": [0, 1] * 50})
        splits = make_leakage_aware_split(data, 0.2, 0.2)
        again = make_leakage_aware_split(data, 0.2, 0.2)
        self.assertEqual([len(s) for s in splits], [60, 20, 20])
        seen = set()
        for split, repeated in zip(splits, again):
            self.assertFalse(seen.intersection(split["full_text"]))
            seen.update(split["full_text"])
            self.assertEqual(split["label"].mean(), 0.5)
            pd.testing.assert_frame_equal(split, repeated)
        with self.assertRaises(ValueError):
            make_leakage_aware_split(pd.concat([data, data.iloc[:1]]))
        with self.assertRaises(ValueError):
            make_leakage_aware_split(data, 0.8, 0.3)


class FeatureTests(unittest.TestCase):
    def test_url_parsing(self):
        self.assertEqual(extract_urls("<a href='HTTPS://example.org/login'>"), ["HTTPS://example.org/login"])
        cases = [
            ("www.example.org/path", len("www.example.org"), 0, 0),
            ("HTTPS://user:pass@example.org:443/path", len("example.org"), 0, 1),
            ("http://127.0.0.1/login", len("127.0.0.1"), 1, 0),
            ("http://127.0.0.1.example.org", len("127.0.0.1.example.org"), 0, 0),
            ("https://[::1]/", len("::1"), 1, 1),
        ]
        for url, hostname_length, has_ip, is_https in cases:
            with self.subTest(url=url):
                features = extract_url_features_from_string(url)
                self.assertEqual(features["hostname_length"], hostname_length)
                self.assertEqual(features["has_ip"], has_ip)
                self.assertEqual(features["is_https"], is_https)
        frame = pd.DataFrame({"full_text": [None, "https://example.org http://example.org"]}, index=[5, 9])
        features = extract_url_features_for_df(frame)
        self.assertEqual(features.index.tolist(), [5, 9])
        self.assertEqual(features["num_urls"].tolist(), [0, 2])
        self.assertEqual(features["is_https"].tolist(), [0, 0])

    def test_sparse_tfidf_and_classifier(self):
        extractor = TextFeatureExtractor()
        train = extractor.fit_transform(pd.Series(["normal message", "normal meeting", "verify account", "account login"]))
        vocabulary = dict(extractor.vectorizer.vocabulary_)
        validation = extractor.transform(pd.Series(["unseenword account", None]))
        self.assertTrue(issparse(train))
        self.assertEqual(train.dtype, np.float32)
        self.assertEqual(vocabulary, extractor.vectorizer.vocabulary_)
        self.assertNotIn("unseenword", vocabulary)
        model = LogisticRegression().fit(train, [0, 0, 1, 1])
        self.assertEqual(model.predict(validation).shape, (2,))


class TrainingTests(unittest.TestCase):
    def test_finetuning_evaluation_and_save_offline(self):
        # Exercise the actual pipeline with a tiny local model and variable-length inputs.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nverify\naccount\nmeeting\n", encoding="utf-8")
            tokenizer = BertTokenizer(vocab_file=str(root / "vocab.txt"))
            model = DistilBertForSequenceClassification(DistilBertConfig(
                vocab_size=len(tokenizer), n_layers=1, n_heads=2, dim=16, hidden_dim=32, num_labels=2,
            ))
            data_dir = root / "processed"
            data_dir.mkdir()
            data = pd.DataFrame({"full_text": ["hello", "hello meeting", "verify account", "verify"], "label": [0, 0, 1, 1]})
            for split in ("train", "val"):
                data.to_csv(data_dir / f"{split}.csv", index=False)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"data": {"processed_dir": str(data_dir), "random_seed": 7}, "models": {}}), encoding="utf-8")
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch("model.bert_finetune.AutoTokenizer.from_pretrained", return_value=tokenizer), patch(
                    "model.bert_finetune.AutoModelForSequenceClassification.from_pretrained", return_value=model
                ):
                    train_distilbert(str(config))
                self.assertTrue((root / "results/distilbert/best_model/config.json").exists())
                self.assertTrue((root / "results/distilbert/best_model/tokenizer_config.json").exists())
                metrics = pd.read_csv(root / "results/distilbert_metrics.csv")
                self.assertFalse(metrics[["accuracy", "precision", "recall", "f1"]].isna().any().any())
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
