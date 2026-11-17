import os
import re
from datasets import load_dataset

URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)


def fetch_and_inspect():
    os.makedirs("data/raw_full", exist_ok=True)

    # Load dataset with full email bodies from Hugging Face
    print("Downloading zefang-liu/phishing-email-dataset...")
    ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    df = ds.to_pandas()

    # Standardize columns
    df = df.dropna(subset=["Email Text", "Email Type"]).copy()
    df["label"] = (df["Email Type"] == "Phishing Email").astype(int)
    df["body"] = df["Email Text"].astype(str)
    df["has_url"] = df["body"].apply(lambda x: bool(URL_REGEX.search(x)))

    print(f"Total valid emails: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    print(f"Emails containing URLs: {df['has_url'].sum()} ({df['has_url'].mean() * 100:.1f}%)")
    print(f"Average body length (chars): {df['body'].str.len().mean():.0f}")

    output_path = os.path.join("data", "raw_full", "full_emails.csv")
    df[["body", "label"]].to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    fetch_and_inspect()