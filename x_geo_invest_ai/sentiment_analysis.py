from pathlib import Path

import pandas as pd
from textblob import TextBlob

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def process_file(path):
    df = pd.read_csv(path)
    df["content"] = df["content"].fillna("").astype(str)
    df["sentiment"] = df["content"].apply(get_sentiment)
    output_path = PROCESSED_DIR / f"{path.stem}_with_sentiment.csv"
    df.to_csv(output_path, index=False)
    print(f"Sentiment analysis completed for {path.name} -> {output_path.name}")


def main():
    for name in ("geopolitical_news.csv", "financial_news.csv", "combined_news.csv"):
        path = RAW_DIR / name
        if path.exists():
            process_file(path)

    combined_output = PROCESSED_DIR / "combined_news_with_sentiment.csv"
    if combined_output.exists():
        pd.read_csv(combined_output).to_csv(BASE_DIR / "news_with_sentiment.csv", index=False)


if __name__ == "__main__":
    main()
