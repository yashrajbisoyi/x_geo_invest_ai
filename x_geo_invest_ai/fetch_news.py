from datetime import datetime, timedelta, timezone
from pathlib import Path
import os

from dotenv import load_dotenv
import pandas as pd
import requests

load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

VERIFIED_SOURCE_DOMAINS = ["reuters.com", "bbc.com", "bloomberg.com"]

DATASET_CONFIG = {
    "geopolitical": {
        "query": '"geopolitics" OR war OR sanctions OR conflict OR diplomacy OR border OR military',
        "keywords": ["geopolitics", "war", "sanctions", "conflict", "diplomacy", "military"],
    },
    "financial": {
        "query": '"stock market" OR inflation OR interest rates OR banking OR equities OR finance',
        "keywords": ["stock market", "inflation", "interest rates", "banking", "equities", "finance"],
    },
}


def fetch_articles(query, from_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "domains": ",".join(VERIFIED_SOURCE_DOMAINS),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "from": from_date,
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "ok":
        raise RuntimeError(f"News API error: {payload}")

    return payload.get("articles", [])


def normalize_articles(articles, dataset_type, dataset_keywords):
    rows = []
    for article in articles:
        source = article.get("source", {}).get("name") or "Unknown"
        title = article.get("title") or ""
        description = article.get("description") or ""
        content = article.get("content") or description
        text = f"{title} {description} {content}".lower()

        rows.append(
            {
                "title": title,
                "source": source,
                "published": article.get("publishedAt"),
                "content": description,
                "url": article.get("url"),
                "dataset_type": dataset_type,
                "verified_source": any(domain.split(".")[0] in source.lower() for domain in VERIFIED_SOURCE_DOMAINS),
                "matched_keywords": ", ".join(
                    keyword for keyword in dataset_keywords if keyword.lower() in text
                ),
                "is_geopolitical": dataset_type == "geopolitical",
                "is_financial": dataset_type == "financial",
            }
        )
    return pd.DataFrame(rows)


def save_dataset(df, filename):
    output_path = RAW_DIR / filename
    df.to_csv(output_path, index=False)
    return output_path


def combine_datasets(datasets):
    combined = pd.concat(datasets, ignore_index=True)
    combined["matched_keywords"] = combined["matched_keywords"].fillna("")

    grouped = (
        combined.groupby(["url", "title"], dropna=False)
        .agg(
            {
                "source": "first",
                "published": "first",
                "content": "first",
                "verified_source": "max",
                "matched_keywords": lambda values: ", ".join(
                    sorted({item.strip() for value in values for item in str(value).split(",") if item.strip()})
                ),
                "is_geopolitical": "max",
                "is_financial": "max",
            }
        )
        .reset_index()
    )

    grouped["dataset_type"] = grouped.apply(
        lambda row: "combined"
        if row["is_geopolitical"] and row["is_financial"]
        else ("geopolitical" if row["is_geopolitical"] else "financial"),
        axis=1,
    )
    return grouped


def main():
    if not API_KEY:
        raise RuntimeError("NEWS_API_KEY is missing from the environment.")

    from_date = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
    generated_datasets = []

    for dataset_name, config in DATASET_CONFIG.items():
        articles = fetch_articles(config["query"], from_date)
        df = normalize_articles(articles, dataset_name, config["keywords"])
        output_path = save_dataset(df, f"{dataset_name}_news.csv")
        generated_datasets.append(df)
        print(f"Saved {len(df)} rows to {output_path}")

    combined_df = combine_datasets(generated_datasets)
    combined_path = save_dataset(combined_df, "combined_news.csv")

    combined_df.to_csv(BASE_DIR / "news_data.csv", index=False)
    print(f"Saved {len(combined_df)} rows to {combined_path}")
    print(combined_df[["source", "dataset_type", "title"]].head())


if __name__ == "__main__":
    main()
