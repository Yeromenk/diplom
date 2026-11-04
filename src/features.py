import re
import ipaddress
import urllib.parse
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

URL_REGEX = r'''https?://[^\s<>"']+|www\.[^\s<>"']+'''

def extract_urls(text: str) -> list[str]:
    """Extract HTTP(S) and www URLs from text."""
    if not isinstance(text, str):
        return []
    return re.findall(URL_REGEX, text, flags=re.IGNORECASE)

def extract_url_features_from_string(url: str) -> dict:
    """Extract numeric features from a single URL."""
    try:
        # A leading // lets urlparse recognize schemeless www URLs.
        parsed = urllib.parse.urlparse(url if "://" in url else "//" + url)
        hostname = parsed.hostname or ""
        is_https = int(parsed.scheme.lower() == "https")
    except ValueError:
        hostname = ""
        is_https = 0

    try:
        ipaddress.ip_address(hostname)
        has_ip = 1
    except ValueError:
        has_ip = 0

    suspicious_words = ['login', 'verify', 'update', 'account', 'banking', 'secure', 'signin', 'ebayisapi']
    has_suspicious_word = 1 if any(w in url.lower() for w in suspicious_words) else 0

    return {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "count_dots": url.count('.'),
        "count_hyphens": url.count('-'),
        "count_at": url.count('@'),
        "count_question": url.count('?'),
        "count_equal": url.count('='),
        "count_slash": url.count('/'),
        "count_digits": sum(c.isdigit() for c in url),
        "has_ip": has_ip,
        "is_https": is_https,
        "has_suspicious_word": has_suspicious_word
    }

def extract_url_features_for_df(df: pd.DataFrame, text_column: str = "full_text") -> pd.DataFrame:
    """Aggregate URL features for each email, preserving the input index."""
    feature_rows = []

    for text in df[text_column]:
        urls = extract_urls(text)
        num_urls = len(urls)

        if num_urls == 0:
            feats = {
                "num_urls": 0, "url_length": 0, "hostname_length": 0, "count_dots": 0,
                "count_hyphens": 0, "count_at": 0, "count_question": 0, "count_equal": 0,
                "count_slash": 0, "count_digits": 0, "has_ip": 0, "is_https": 0,
                "has_suspicious_word": 0
            }
        else:
            extracted = [extract_url_features_from_string(u) for u in urls]
            feats = {
                "num_urls": num_urls,
                "url_length": np.mean([f["url_length"] for f in extracted]),
                "hostname_length": np.mean([f["hostname_length"] for f in extracted]),
                "count_dots": np.max([f["count_dots"] for f in extracted]),
                "count_hyphens": np.max([f["count_hyphens"] for f in extracted]),
                "count_at": np.max([f["count_at"] for f in extracted]),
                "count_question": np.max([f["count_question"] for f in extracted]),
                "count_equal": np.max([f["count_equal"] for f in extracted]),
                "count_slash": np.max([f["count_slash"] for f in extracted]),
                "count_digits": np.mean([f["count_digits"] for f in extracted]),
                "has_ip": np.max([f["has_ip"] for f in extracted]),
                "is_https": np.min([f["is_https"] for f in extracted]),
                "has_suspicious_word": np.max([f["has_suspicious_word"] for f in extracted])
            }
        feature_rows.append(feats)

    return pd.DataFrame(feature_rows, index=df.index)

class TextFeatureExtractor:
    """Fit TF-IDF on training data and reuse its vocabulary for validation/test."""
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            dtype=np.float32
        )

    def fit_transform(self, train_texts: pd.Series) -> csr_matrix:
        # Preserve sparsity to avoid allocating a dense emails-by-vocabulary matrix.
        return self.vectorizer.fit_transform(train_texts.fillna(""))

    def transform(self, texts: pd.Series) -> csr_matrix:
        return self.vectorizer.transform(texts.fillna(""))
