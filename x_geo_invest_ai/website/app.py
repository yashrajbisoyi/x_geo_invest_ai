from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
import pandas as pd
import requests
from textblob import TextBlob
import yfinance as yf

load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RUNTIME_DATASET_PATH = PROCESSED_DIR / "runtime_combined_news.csv"
LIVE_DATASET_PATH = PROCESSED_DIR / "live_runtime_news.csv"
DATA_PATH_CANDIDATES = [
    RUNTIME_DATASET_PATH,
    PROCESSED_DIR / "final_combined_news.csv",
    BASE_DIR / "final_investment_recommendations.csv",
]
NEWS_PATH_CANDIDATES = [
    PROCESSED_DIR / "combined_news_with_risk.csv",
    BASE_DIR / "news_with_risk.csv",
]
CONFIDENCE_PATH_CANDIDATES = [
    BASE_DIR / "final_with_confidence.csv",
    ARTIFACTS_DIR / "final_with_confidence.csv",
]
EVALUATION_PATH = ARTIFACTS_DIR / "model_evaluation.json"

RISK_LEVELS = ["Low", "Medium", "High"]
SENTIMENT_LEVELS = ["Negative", "Neutral", "Positive"]
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
NEWSDATA_API_KEY = os.getenv("NEWSDATA_API_KEY") or os.getenv("NEWS_API_KEY")

market_cache = {"data": None, "timestamp": 0, "debug": {}}
x_news_cache = {"data": None, "timestamp": 0, "debug": {}}
country_news_cache = {}
MARKET_REQUEST_TIMEOUT = float(os.getenv("MARKET_REQUEST_TIMEOUT", "4"))
MARKET_CACHE_SECONDS = int(os.getenv("MARKET_CACHE_SECONDS", "300"))
LIVE_NEWS_CACHE_SECONDS = 120
COUNTRY_MONITOR_CACHE_SECONDS = 600

RISK_KEYWORDS = [
    "war", "conflict", "sanction", "missile", "nuclear", "invasion", "military",
    "border", "terror", "embargo", "blockade", "retaliation", "escalation",
    "tariff", "diplomacy", "unrest", "coup", "shipping disruption"
]

HIGH_RISK_KEYWORDS = [
    "war", "missile", "nuclear", "invasion", "terror", "blockade", "bombing", "airstrike"
]

COUNTRY_MONITOR_ALIASES = {
    "united states": ["united states", "united states of america", "usa", "u.s.", "us", "america"],
    "united kingdom": ["united kingdom", "uk", "u.k.", "britain", "great britain", "england"],
    "united arab emirates": ["united arab emirates", "uae", "u.a.e.", "emirates"],
    "south korea": ["south korea", "republic of korea"],
    "north korea": ["north korea", "dprk", "democratic people's republic of korea"],
    "russia": ["russia", "russian federation"],
    "iran": ["iran", "islamic republic of iran"],
    "china": ["china", "people's republic of china", "prc"],
    "taiwan": ["taiwan", "republic of china"],
    "turkey": ["turkey", "turkiye"],
    "syria": ["syria", "syrian arab republic"],
    "venezuela": ["venezuela", "bolivarian republic of venezuela"],
    "tanzania": ["tanzania", "united republic of tanzania"],
    "moldova": ["moldova", "republic of moldova"],
    "czechia": ["czechia", "czech republic"],
    "laos": ["laos", "lao pdr", "lao people's democratic republic"],
    "viet nam": ["viet nam", "vietnam"],
}

COUNTRY_ALIAS_LOOKUP = {}
for canonical_name, aliases in COUNTRY_MONITOR_ALIASES.items():
    normalized_canonical = " ".join(str(canonical_name).strip().lower().replace("-", " ").split())
    COUNTRY_ALIAS_LOOKUP[normalized_canonical] = normalized_canonical
    for alias in aliases:
        normalized_alias = " ".join(str(alias).strip().lower().replace("-", " ").split())
        if normalized_alias:
            COUNTRY_ALIAS_LOOKUP[normalized_alias] = normalized_canonical


def first_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


DATA_PATH = first_existing_path(DATA_PATH_CANDIDATES)
NEWS_PATH = first_existing_path(NEWS_PATH_CANDIDATES)


def get_evaluation():
    return load_json(EVALUATION_PATH)


def get_base_dataset_path():
    return first_existing_path(DATA_PATH_CANDIDATES)


def get_base_dataset_frame():
    return load_csv(get_base_dataset_path())


def get_news_dataset_path():
    return first_existing_path(NEWS_PATH_CANDIDATES)


def get_news_dataset_frame():
    return load_csv(get_news_dataset_path())


def get_confidence_dataset_path():
    return first_existing_path(CONFIDENCE_PATH_CANDIDATES)


def get_confidence_dataset_frame():
    return load_csv(get_confidence_dataset_path())


def text_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    if polarity < -0.1:
        return "Negative"
    return "Neutral"


def text_risk_level(text):
    text = str(text).lower()
    score = sum(1 for keyword in RISK_KEYWORDS if keyword in text)
    high_hits = sum(1 for keyword in HIGH_RISK_KEYWORDS if keyword in text)

    if high_hits >= 1 or score >= 3:
        return "High"
    if score >= 1:
        return "Medium"
    return "Low"


def recommendation_for_inputs(risk, sentiment):
    if risk == "High":
        return "Buy Gold & Defence Stocks | Avoid Equities"

    if risk == "Medium":
        if sentiment == "Positive":
            return "Selective Equity Buy | Hold Gold"
        return "Hold Gold | Avoid High-Risk Equities"

    if sentiment == "Positive":
        return "Buy Equities, Banking, IT"
    if sentiment == "Negative":
        return "Hold Cash | Defensive Stocks"
    return "Market Neutral | Hold Positions"


def confidence_for_inputs(risk, sentiment, prediction):
    df = get_base_dataset_frame()
    if df.empty:
        return None

    matches = df[
        (df["geo_risk_level"] == risk)
        & (df["sentiment"] == sentiment)
    ]

    if matches.empty:
        return None

    support = (matches["investment_recommendation"] == prediction).mean()
    return round(float(support) * 100, 2)


def build_news_records(limit=12):
    news_df = get_news_dataset_frame()
    if news_df.empty:
        return []

    latest = news_df.sort_values("published", ascending=False).head(limit).fillna("")
    records = []
    for _, row in latest.iterrows():
        records.append(
            {
                "title": row.get("title", "Untitled"),
                "source": row.get("source", "Unknown"),
                "published": row.get("published", ""),
                "content": row.get("content", ""),
                "sentiment": row.get("sentiment", "Unknown"),
                "risk": row.get("geo_risk_level", "Unknown"),
                "dataset_type": row.get("dataset_type", "combined"),
                "matched_keywords": row.get("matched_keywords", ""),
            }
        )
    return records


def build_live_recommendation_record(text, source, published, author=None, post_url=None):
    risk = text_risk_level(text)
    sentiment = text_sentiment(text)
    recommendation = recommendation_for_inputs(risk, sentiment)
    confidence = confidence_for_inputs(risk, sentiment, recommendation)

    return {
        "title": text[:160] + ("..." if len(text) > 160 else ""),
        "source": source,
        "published": published,
        "content": text,
        "author": author or "Unknown",
        "url": post_url,
        "sentiment": sentiment,
        "risk": risk,
        "recommendation": recommendation,
        "confidence": confidence,
        "dataset_type": "live-feed",
    }


def normalize_country_name(value):
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def text_matches_country_terms(text, terms):
    normalized_text = normalize_country_name(text)
    if not normalized_text or not terms:
        return False

    padded_text = f" {normalized_text} "
    for term in terms:
        normalized_term = normalize_country_name(term)
        if not normalized_term:
            continue
        if f" {normalized_term} " in padded_text:
            return True
    return False


def parse_numeric_confidence(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def country_monitor_terms(country_name):
    normalized = normalize_country_name(country_name)
    if not normalized:
        return []
    canonical = COUNTRY_ALIAS_LOOKUP.get(normalized, normalized)
    aliases = set(COUNTRY_MONITOR_ALIASES.get(canonical, []))
    aliases.add(canonical)
    aliases.add(normalized)
    return sorted({normalize_country_name(alias) for alias in aliases if alias}, key=len, reverse=True)


def fetch_newsdata_country_news(country_name, terms, size=8):
    if not NEWSDATA_API_KEY:
        raise RuntimeError("NEWSDATA_API_KEY is missing from the environment.")

    query_terms = [term for term in terms if term]
    if not query_terms:
        raise RuntimeError("No country terms available for query.")

    query = " OR ".join(f'"{term}"' if " " in term else term for term in query_terms[:6])
    response = requests.get(
        "https://newsdata.io/api/1/news",
        params={
            "apikey": NEWSDATA_API_KEY,
            "language": "en",
            "q": query,
            "size": size,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(payload.get("results", payload))

    records = []
    for item in payload.get("results", []):
        text_parts = [
            item.get("title") or "",
            item.get("description") or "",
            item.get("content") or "",
        ]
        text = " ".join(part for part in text_parts if part).strip()
        if not text_matches_country_terms(text, terms):
            continue

        record = build_live_recommendation_record(
            text=text,
            source=item.get("source_name") or "newsdata.io",
            published=item.get("pubDate", ""),
            author=item.get("creator", ["Unknown"])[0] if isinstance(item.get("creator"), list) and item.get("creator") else "Unknown",
            post_url=item.get("link"),
        )
        records.append(record)

    return records, {
        "status": "ok",
        "provider": "newsdata.io-country",
        "records": len(records),
        "query": query,
    }


def get_country_live_news(country_name, force_refresh=False):
    normalized = normalize_country_name(country_name)
    terms = country_monitor_terms(country_name)
    cached = country_news_cache.get(normalized)
    if (
        cached
        and not force_refresh
        and time.time() - cached["timestamp"] < COUNTRY_MONITOR_CACHE_SECONDS
    ):
        return cached["data"], cached["debug"], "hit"

    try:
        records, debug = fetch_newsdata_country_news(country_name, terms)
        country_news_cache[normalized] = {
            "data": records,
            "debug": debug,
            "timestamp": time.time(),
        }
        return records, debug, "miss"
    except Exception as exc:
        debug = {"status": "error", "provider": "country-live", "message": str(exc)}
        fallback = cached["data"] if cached else []
        if fallback:
            return fallback, cached["debug"], "stale"
        return [], debug, "empty"


def country_live_monitor(country_name):
    normalized = normalize_country_name(country_name)
    display_name = " ".join(word.capitalize() for word in normalized.split()) if normalized else "Unknown"
    terms = country_monitor_terms(country_name)
    general_records, general_debug, _ = get_live_news_records()
    live_country_records, country_debug, _ = get_country_live_news(country_name)

    matches = []
    for record in live_country_records:
        text = f"{record.get('title', '')} {record.get('content', '')}"
        if text_matches_country_terms(text, terms):
            matches.append(record)

    if not matches:
        for record in general_records:
            text = f"{record.get('title', '')} {record.get('content', '')}"
            if text_matches_country_terms(text, terms):
                matches.append(record)

    deduped_matches = []
    seen_live_keys = set()
    for item in matches:
        dedupe_key = (
            str(item.get("title", "")).strip().lower(),
            str(item.get("source", "")).strip().lower(),
            str(item.get("published", "")).strip().lower(),
        )
        if dedupe_key in seen_live_keys:
            continue
        seen_live_keys.add(dedupe_key)
        deduped_matches.append(item)
    matches = deduped_matches

    confidence_df = get_confidence_dataset_frame()
    if confidence_df.empty:
        confidence_df = get_base_dataset_frame()

    dataset_matches = []
    seen_dataset_keys = set()
    if not confidence_df.empty:
        searchable_columns = [column for column in ["title", "content", "source"] if column in confidence_df.columns]
        for _, row in confidence_df.fillna("").iterrows():
            text = " ".join(str(row.get(column, "")) for column in searchable_columns)
            if not text_matches_country_terms(text, terms):
                continue

            dedupe_key = (
                str(row.get("title", "")).strip().lower(),
                str(row.get("source", "")).strip().lower(),
                str(row.get("published", "")).strip().lower(),
            )
            if dedupe_key in seen_dataset_keys:
                continue
            seen_dataset_keys.add(dedupe_key)

            dataset_matches.append(
                {
                    "title": row.get("title", "Untitled"),
                    "source": row.get("source", "Dataset"),
                    "published": row.get("published", ""),
                    "risk": row.get("geo_risk_level") or row.get("risk") or "Unknown",
                    "sentiment": row.get("sentiment", "Unknown"),
                    "recommendation": row.get("investment_recommendation") or row.get("recommendation") or "Monitoring only",
                    "confidence": parse_numeric_confidence(
                        row.get("confidence_score_%") if "confidence_score_%" in row.index else row.get("confidence")
                    ),
                    "url": row.get("url") if "url" in row.index else None,
                }
            )

    combined_matches = [
        {
            "title": item.get("title", "Untitled"),
            "source": item.get("source", "Unknown"),
            "published": item.get("published", ""),
            "risk": item.get("risk", "Unknown"),
            "sentiment": item.get("sentiment", "Unknown"),
            "recommendation": item.get("recommendation", "Monitoring only"),
            "confidence": item.get("confidence"),
            "url": item.get("url"),
        }
        for item in matches
    ]

    merged_dataset_matches = []
    existing_keys = {
        (
            str(item.get("title", "")).strip().lower(),
            str(item.get("source", "")).strip().lower(),
            str(item.get("published", "")).strip().lower(),
        )
        for item in combined_matches
    }
    for item in dataset_matches:
        dedupe_key = (
            str(item.get("title", "")).strip().lower(),
            str(item.get("source", "")).strip().lower(),
            str(item.get("published", "")).strip().lower(),
        )
        if dedupe_key in existing_keys:
            continue
        merged_dataset_matches.append(item)

    combined_matches.extend(merged_dataset_matches)

    combined_matches = sorted(combined_matches, key=lambda item: str(item.get("published", "")), reverse=True)
    risk_counts = Counter(item.get("risk", "Unknown") for item in combined_matches if item.get("risk"))
    sentiment_counts = Counter(item.get("sentiment", "Unknown") for item in combined_matches if item.get("sentiment"))
    recommendation_counts = Counter(item.get("recommendation", "Unknown") for item in combined_matches if item.get("recommendation"))
    confidence_values = [item.get("confidence") for item in combined_matches if item.get("confidence") is not None]

    top_risk = risk_counts.most_common(1)[0][0] if risk_counts else None
    top_sentiment = sentiment_counts.most_common(1)[0][0] if sentiment_counts else None
    top_recommendation = recommendation_counts.most_common(1)[0][0] if recommendation_counts else None

    if top_risk and top_sentiment and not top_recommendation:
        top_recommendation = recommendation_for_inputs(top_risk, top_sentiment)

    average_confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else None
    if average_confidence is None and top_risk and top_sentiment and top_recommendation:
        average_confidence = confidence_for_inputs(top_risk, top_sentiment, top_recommendation)

    if matches and merged_dataset_matches:
        source_mode = "mixed"
    elif matches:
        source_mode = "live"
    elif dataset_matches:
        source_mode = "historical"
    else:
        source_mode = "unavailable"

    if matches:
        provider = country_debug.get("provider") or general_debug.get("provider")
    elif dataset_matches:
        provider = "historical-dataset"
    else:
        provider = country_debug.get("provider") or general_debug.get("provider")

    return {
        "country": display_name,
        "normalized_country": normalized,
        "provider": provider,
        "source_mode": source_mode,
        "records_found": len(combined_matches),
        "top_risk": top_risk or "No live signal",
        "top_sentiment": top_sentiment or "No live signal",
        "top_recommendation": top_recommendation or "Monitoring only",
        "average_confidence": average_confidence,
        "last_sync": int(x_news_cache["timestamp"]) if x_news_cache["timestamp"] else None,
        "headlines": [
            {
                "title": item.get("title", "Untitled"),
                "source": item.get("source", "Unknown"),
                "published": item.get("published", ""),
                "risk": item.get("risk", "Unknown"),
                "sentiment": item.get("sentiment", "Unknown"),
                "recommendation": item.get("recommendation", "Monitoring only"),
                "url": item.get("url"),
            }
            for item in combined_matches[:4]
        ],
    }


def live_records_to_dataframe(records):
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(
        [
            {
                "title": record.get("title", "Untitled"),
                "source": record.get("source", "Unknown"),
                "published": record.get("published", ""),
                "content": record.get("content", ""),
                "sentiment": record.get("sentiment", "Unknown"),
                "geo_risk_level": record.get("risk", "Unknown"),
                "investment_recommendation": record.get("recommendation", "Unknown"),
                "dataset_type": record.get("dataset_type", "live-feed"),
            }
            for record in records
        ]
    )

    frame["content"] = frame["content"].fillna("")
    frame["published"] = frame["published"].fillna("")
    return frame


def get_live_news_records(force_refresh=False):
    cache_is_fresh = (
        not force_refresh
        and x_news_cache["data"] is not None
        and time.time() - x_news_cache["timestamp"] < LIVE_NEWS_CACHE_SECONDS
    )
    if cache_is_fresh:
        return x_news_cache["data"], x_news_cache["debug"], "hit"

    try:
        try:
            records, debug = fetch_newsdata_live_news()
        except Exception as newsdata_exc:
            records, debug = fetch_x_live_news()
            debug["fallback_from"] = f"newsdata.io failed: {newsdata_exc}"

        x_news_cache["data"] = records
        x_news_cache["debug"] = debug
        x_news_cache["timestamp"] = time.time()
        persist_runtime_dataset(records)
        return records, debug, "miss"
    except Exception as exc:
        debug = {"status": "error", "message": str(exc)}
        fallback = x_news_cache["data"] if x_news_cache["data"] is not None else []
        if fallback:
            persist_runtime_dataset(fallback)
        return fallback, debug, "stale" if fallback else "empty"


def runtime_dataset_frame(include_live=True):
    base_df = get_base_dataset_frame()
    if not include_live:
        return base_df

    live_records, _, _ = get_live_news_records()
    live_df = live_records_to_dataframe(live_records)

    if base_df.empty:
        combined = live_df
    elif live_df.empty:
        combined = base_df
    else:
        combined = pd.concat([base_df, live_df], ignore_index=True, sort=False)

    if combined.empty:
        return combined

    dedupe_columns = [column for column in ["title", "source", "published"] if column in combined.columns]
    if dedupe_columns:
        combined = combined.drop_duplicates(subset=dedupe_columns, keep="first")
    return combined


def persist_runtime_dataset(records):
    live_df = live_records_to_dataframe(records)
    if not live_df.empty:
        LIVE_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        live_df.to_csv(LIVE_DATASET_PATH, index=False)

    runtime_df = runtime_dataset_frame(include_live=True)
    if not runtime_df.empty:
        RUNTIME_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        runtime_df.to_csv(RUNTIME_DATASET_PATH, index=False)


def build_home_metrics():
    live_debug = x_news_cache["debug"] if x_news_cache["data"] else {}
    summary = dataset_summary(include_live=True)
    last_live_sync = int(x_news_cache["timestamp"]) if x_news_cache["timestamp"] else None
    return {
        "dataset_rows": summary["total_articles"],
        "live_rows": summary.get("live_articles", 0),
        "live_cache": "warm" if x_news_cache["data"] else "cold",
        "last_live_sync": last_live_sync,
        "last_live_sync_label": time.strftime("%I:%M:%S %p", time.localtime(last_live_sync)) if last_live_sync else None,
        "live_provider": live_debug.get("provider"),
        "live_record_count": live_debug.get("records", 0),
    }


def run_training_pipeline():
    ml_model_path = BASE_DIR / "ml_model.py"
    confidence_path = BASE_DIR / "confidence_score.py"

    training = subprocess.run(
        [sys.executable, str(ml_model_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if training.returncode != 0:
        raise RuntimeError(training.stderr.strip() or training.stdout.strip() or "Training failed.")

    confidence = subprocess.run(
        [sys.executable, str(confidence_path)],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=300,
    )
    if confidence.returncode != 0:
        raise RuntimeError(confidence.stderr.strip() or confidence.stdout.strip() or "Confidence scoring failed.")

    return {
        "training_output": training.stdout.strip(),
        "confidence_output": confidence.stdout.strip(),
    }


def fetch_newsdata_live_news():
    if not NEWSDATA_API_KEY:
        raise RuntimeError("NEWSDATA_API_KEY is missing from the environment.")

    response = requests.get(
        "https://newsdata.io/api/1/news",
        params={
            "apikey": NEWSDATA_API_KEY,
            "language": "en",
            "q": "geopolitics OR war OR sanctions OR tariff OR oil OR inflation OR stock market OR interest rates",
            "category": "business,politics",
            "size": 10,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(payload.get("results", payload))

    records = []
    for item in payload.get("results", []):
        text = item.get("title") or item.get("description") or item.get("content") or ""
        records.append(
            build_live_recommendation_record(
                text=text,
                source=item.get("source_name") or "newsdata.io",
                published=item.get("pubDate", ""),
                author=item.get("creator", ["Unknown"])[0] if isinstance(item.get("creator"), list) and item.get("creator") else "Unknown",
                post_url=item.get("link"),
            )
        )

    return records, {
        "status": "ok",
        "provider": "newsdata.io",
        "records": len(records),
    }


def fetch_x_live_news():
    if not X_BEARER_TOKEN:
        raise RuntimeError("X_BEARER_TOKEN is missing from the environment.")

    query = (
        '(geopolitics OR war OR sanctions OR tariff OR oil OR inflation OR "stock market" '
        'OR "interest rates") lang:en -is:retweet'
    )
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"},
        params={
            "query": query,
            "max_results": 10,
            "tweet.fields": "created_at,author_id",
            "expansions": "author_id",
            "user.fields": "name,username",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    users = {
        user["id"]: user
        for user in payload.get("includes", {}).get("users", [])
    }

    records = []
    for tweet in payload.get("data", []):
        user = users.get(tweet.get("author_id"), {})
        username = user.get("username")
        post_url = f"https://x.com/{username}/status/{tweet['id']}" if username else None
        records.append(
            build_live_recommendation_record(
                text=tweet.get("text", ""),
                source="X",
                published=tweet.get("created_at", ""),
                author=user.get("name") or username,
                post_url=post_url,
            )
        )

    return records, {
        "status": "ok",
        "provider": "x",
        "records": len(records),
        "query": query,
    }


def dataset_summary(include_live=True):
    dataset_df = runtime_dataset_frame(include_live=include_live)

    if dataset_df.empty:
        return {
            "total_articles": 0,
            "live_articles": 0,
            "risk_counts": {},
            "sentiment_counts": {},
            "recommendation_counts": {},
            "source_counts": {},
            "dataset_type_counts": {},
            "combo_consistency": None,
            "verified_sources": [],
            "keyword_freq": {},
            "missing_content_pct": 0,
            "missing_title_pct": 0,
            "has_keywords_pct": 0,
            "date_range": {"from": "", "to": ""},
            "live_cache": "empty",
            "last_live_sync": None,
            "active_dataset_path": str(get_base_dataset_path()),
        }

    combo_recommendations = (
        dataset_df.groupby(["geo_risk_level", "sentiment"])["investment_recommendation"]
        .nunique()
        .tolist()
    )
    consistent_combos = sum(1 for count in combo_recommendations if count == 1)
    combo_consistency = round(
        100 * consistent_combos / len(combo_recommendations), 2
    ) if combo_recommendations else None

    source_counts = dataset_df["source"].fillna("Unknown").value_counts().head(8).to_dict()
    dataset_type_counts = (
        dataset_df["dataset_type"].fillna("combined").value_counts().to_dict()
        if "dataset_type" in dataset_df.columns
        else {"combined": int(len(dataset_df))}
    )

    # Keyword frequency
    keyword_freq = {}
    if "matched_keywords" in dataset_df.columns:
        all_keywords = []
        for val in dataset_df["matched_keywords"].fillna(""):
            for kw in str(val).split(","):
                kw = kw.strip()
                if kw:
                    all_keywords.append(kw)
        keyword_freq = dict(Counter(all_keywords).most_common(20))

    # Data quality metrics
    missing_content_pct = round(100 * dataset_df["content"].isna().mean(), 1) if "content" in dataset_df.columns else 0
    missing_title_pct = round(100 * dataset_df["title"].isna().mean(), 1) if "title" in dataset_df.columns else 0
    has_keywords_pct = round(100 * (dataset_df["matched_keywords"].fillna("").str.strip() != "").mean(), 1) if "matched_keywords" in dataset_df.columns else 0

    # Date range
    date_range = {"from": "", "to": ""}
    if "published" in dataset_df.columns:
        dates = pd.to_datetime(dataset_df["published"], errors="coerce").dropna()
        if not dates.empty:
            date_range = {
                "from": dates.min().strftime("%d %b %Y"),
                "to": dates.max().strftime("%d %b %Y"),
            }

    return {
        "total_articles": int(len(dataset_df)),
        "live_articles": int((dataset_df["dataset_type"].fillna("") == "live-feed").sum()) if "dataset_type" in dataset_df.columns else 0,
        "risk_counts": dataset_df["geo_risk_level"].value_counts().reindex(RISK_LEVELS, fill_value=0).to_dict(),
        "sentiment_counts": dataset_df["sentiment"].value_counts().reindex(SENTIMENT_LEVELS, fill_value=0).to_dict(),
        "recommendation_counts": dataset_df["investment_recommendation"].value_counts().to_dict(),
        "source_counts": source_counts,
        "dataset_type_counts": dataset_type_counts,
        "combo_consistency": combo_consistency,
        "verified_sources": sorted(dataset_df["source"].dropna().unique().tolist())[:10],
        "keyword_freq": keyword_freq,
        "missing_content_pct": missing_content_pct,
        "missing_title_pct": missing_title_pct,
        "has_keywords_pct": has_keywords_pct,
        "date_range": date_range,
        "live_cache": "warm" if x_news_cache["data"] else "cold",
        "last_live_sync": int(x_news_cache["timestamp"]) if x_news_cache["timestamp"] else None,
        "active_dataset_path": str(get_base_dataset_path()),
    }


def build_performance_metrics():
    summary = dataset_summary()
    dataset_df = runtime_dataset_frame(include_live=True)
    evaluation = get_evaluation()
    confusion_matrix_path = ARTIFACTS_DIR / "confusion_matrix.png"
    base_metrics = {
        "total_articles": summary["total_articles"],
        "combo_count": 0,
        "combo_consistency": summary["combo_consistency"],
        "top_recommendation": None,
        "best_model": None,
        "accuracy": None,
        "f1_weighted": None,
        "cv_f1_mean": None,
        "cv_f1_std": None,
        "training_time_seconds": None,
        "prediction_time_seconds": None,
        "model_comparison": [],
        "improvement_notes": [],
        "evaluation_dataset_path": None,
        "trained_at": None,
        "confusion_matrix_url": None,
    }

    if not dataset_df.empty:
        combos = (
            dataset_df.groupby(["geo_risk_level", "sentiment"]).size().reset_index(name="count")
        )
        top_recommendation = Counter(dataset_df["investment_recommendation"]).most_common(1)[0]
        base_metrics["combo_count"] = int(len(combos))
        base_metrics["top_recommendation"] = {
            "label": top_recommendation[0],
            "count": int(top_recommendation[1]),
        }

    if evaluation:
        base_metrics.update(
            {
                "best_model": evaluation.get("best_model"),
                "accuracy": evaluation.get("accuracy"),
                "f1_weighted": evaluation.get("f1_weighted"),
                "cv_f1_mean": evaluation.get("cv_f1_mean"),
                "cv_f1_std": evaluation.get("cv_f1_std"),
                "training_time_seconds": evaluation.get("training_time_seconds"),
                "prediction_time_seconds": evaluation.get("prediction_time_seconds"),
                "model_comparison": evaluation.get("model_comparison", []),
                "improvement_notes": evaluation.get("improvement_notes", []),
                "evaluation_dataset_path": evaluation.get("dataset_path"),
                "trained_at": evaluation.get("trained_at"),
                "confusion_matrix_url": "/artifacts/confusion_matrix.png" if confusion_matrix_path.exists() else None,
            }
        )

    return base_metrics


def fetch_yfinance_price(ticker, attempts, periods=()):
    try:
        ticker_obj = yf.Ticker(ticker)
        fast_info = getattr(ticker_obj, "fast_info", None)
        if fast_info:
            candidate = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
            if candidate is not None:
                attempts.append("yfinance.fast_info")
                return round(float(candidate), 2)
            attempts.append("yfinance.fast_info returned no price")
    except Exception as exc:
        attempts.append(f"yfinance.fast_info failed: {exc}")
        if "Too Many Requests" in str(exc):
            return None

    for period in periods:
        try:
            history = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            if not history.empty and "Close" in history and not history["Close"].dropna().empty:
                attempts.append(f"yfinance.history({period},1d)")
                return round(float(history["Close"].dropna().iloc[-1]), 2)
            attempts.append(f"yfinance.history({period}) returned empty")
        except Exception as exc:
            attempts.append(f"yfinance.history({period}) failed: {exc}")
            if "Too Many Requests" in str(exc):
                break

    return None


def fetch_yahoo_quote_price(ticker, attempts):
    try:
        response = requests.get(
            "https://query1.finance.yahoo.com/v7/finance/quote",
            params={"symbols": ticker},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=MARKET_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json().get("quoteResponse", {}).get("result", [])
        if result:
            candidate = result[0].get("regularMarketPrice")
            if candidate is not None:
                attempts.append("yahoo.quote")
                return round(float(candidate), 2)
        attempts.append("yahoo.quote returned no price")
    except Exception as exc:
        attempts.append(f"yahoo.quote failed: {exc}")
    return None


def fetch_alpha_vantage_price(label, attempts):
    if not ALPHA_VANTAGE_API_KEY or label not in {"gold", "oil"}:
        return None

    try:
        if label == "gold":
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "GOLD_SILVER_SPOT",
                    "symbol": "GOLD",
                    "apikey": ALPHA_VANTAGE_API_KEY,
                },
                timeout=MARKET_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            candidate = payload.get("price")
            if candidate is not None:
                attempts.append("alpha_vantage.gold_spot")
                return round(float(candidate), 2)
            attempts.append("alpha_vantage.gold_spot returned no price")
            return None

        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "BRENT",
                "interval": "daily",
                "apikey": ALPHA_VANTAGE_API_KEY,
            },
            timeout=MARKET_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data", [])
        if rows:
            candidate = rows[0].get("value")
            if candidate is not None:
                attempts.append("alpha_vantage.brent")
                return round(float(candidate), 2)
        attempts.append("alpha_vantage.brent returned no price")
    except Exception as exc:
        attempts.append(f"alpha_vantage failed: {exc}")

    return None


def fetch_nse_nifty_price(attempts):
    try:
        response = requests.get(
            "https://www.nseindia.com/api/allIndices",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
                "Referer": "https://www.nseindia.com/",
            },
            timeout=MARKET_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        for item in payload.get("data", []):
            if item.get("index") == "NIFTY 50":
                candidate = item.get("last")
                if candidate is not None:
                    attempts.append("nse.allIndices")
                    return round(float(candidate), 2)
                break
        attempts.append("nse.allIndices returned no NIFTY price")
    except Exception as exc:
        attempts.append(f"nse.allIndices failed: {exc}")

    return None


def fetch_market_symbol(label, ticker):
    attempts = []
    price = fetch_yfinance_price(ticker, attempts)
    if price is None:
        price = fetch_yahoo_quote_price(ticker, attempts)
    if price is None:
        price = fetch_alpha_vantage_price(label, attempts)
    if price is None and label == "nifty":
        price = fetch_nse_nifty_price(attempts)

    return label, price if price is not None else "Unavailable", {
        "ticker": ticker,
        "status": "ok" if price is not None else "unavailable",
        "attempts": attempts,
    }


def fetch_market_snapshot():
    symbols = {
        "gold": "GC=F",
        "oil": "BZ=F",
        "nifty": "^NSEI",
        "sensex": "^BSESN",
    }
    data = {}
    debug = {}

    with ThreadPoolExecutor(max_workers=len(symbols)) as executor:
        futures = [executor.submit(fetch_market_symbol, label, ticker) for label, ticker in symbols.items()]
        for future in futures:
            label, price, details = future.result()
            data[label] = price
            debug[label] = details

    return data, debug


def fetch_usdinr_rate():
    attempts = []
    rate = None

    try:
        ticker_obj = yf.Ticker("USDINR=X")
        fast_info = getattr(ticker_obj, "fast_info", None)
        if fast_info:
            candidate = fast_info.get("lastPrice") or fast_info.get("regularMarketPrice")
            if candidate is not None and float(candidate) > 10:
                rate = round(float(candidate), 2)
                attempts.append(f"yfinance.fast_info(USDINR=X) -> {rate}")
            else:
                attempts.append(f"yfinance.fast_info(USDINR=X) returned bad value: {candidate}")
    except Exception as exc:
        attempts.append(f"yfinance.fast_info(USDINR=X) failed: {exc}")

    if rate is None:
        rate = 84.0
        attempts.append("hardcoded fallback rate 84.0")

    return rate, attempts


# ── ROUTES ──

@app.route("/artifacts/<path:filename>")
def serve_artifact(filename):
    return send_from_directory(str(ARTIFACTS_DIR), filename)


@app.route("/")
@app.route("/landing")
def landing():
    return render_template("landing.html")


@app.route("/home", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    selection = {"risk": "Medium", "sentiment": "Neutral"}
    evaluation = get_evaluation()
    home_metrics = build_home_metrics()

    if request.method == "POST":
        risk = request.form.get("risk", "Medium")
        sentiment = request.form.get("sentiment", "Neutral")

        prediction = recommendation_for_inputs(risk, sentiment)
        confidence = confidence_for_inputs(risk, sentiment, prediction)
        selection = {"risk": risk, "sentiment": sentiment}

    return render_template(
        "home.html",
        prediction=prediction,
        confidence=confidence,
        selection=selection,
        evaluation=evaluation,
        home_metrics=home_metrics,
        risks=RISK_LEVELS,
        sentiments=SENTIMENT_LEVELS,
    )


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", summary=dataset_summary())


@app.route("/performance")
def performance():
    return render_template("performance.html", metrics=build_performance_metrics())


@app.route("/world-monitor")
def world_monitor():
    return render_template("world_monitor.html")


@app.route("/news")
def news():
    return render_template("news.html", articles=build_news_records())


@app.route("/api/live-news")
def live_news():
    records, debug, cache_state = get_live_news_records()
    return jsonify({"articles": records, "debug": debug, "cache": cache_state})


@app.route("/api/country-monitor")
def api_country_monitor():
    country = request.args.get("country", "").strip()
    force_refresh = request.args.get("refresh", "").strip().lower() in {"1", "true", "yes"}
    if not country:
        return jsonify({"status": "error", "message": "country is required"}), 400
    if force_refresh:
        get_live_news_records(force_refresh=True)
    payload = country_live_monitor(country)
    return jsonify({"status": "ok", **payload})


@app.route("/api/dataset-summary")
def api_dataset_summary():
    force_refresh = request.args.get("refresh", "").strip().lower() in {"1", "true", "yes"}
    if force_refresh:
        get_live_news_records(force_refresh=True)
    summary = dataset_summary(include_live=True)
    summary["live_provider"] = x_news_cache["debug"].get("provider") if x_news_cache["data"] else None
    summary["live_record_count"] = x_news_cache["debug"].get("records", 0) if x_news_cache["data"] else 0
    return jsonify(summary)


@app.route("/api/refresh-runtime-model", methods=["POST"])
def refresh_runtime_model():
    try:
        records, debug, cache_state = get_live_news_records(force_refresh=True)
        persist_runtime_dataset(records)
        pipeline_result = run_training_pipeline()
        summary = dataset_summary(include_live=True)
        evaluation = get_evaluation()
        return jsonify(
            {
                "status": "ok",
                "cache": cache_state,
                "debug": debug,
                "summary": {
                    **summary,
                    "live_provider": debug.get("provider"),
                    "live_record_count": debug.get("records", len(records)),
                },
                "evaluation": evaluation,
                "pipeline": pipeline_result,
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/live-data")
def live_data():
    try:
        if market_cache["data"] is not None and time.time() - market_cache["timestamp"] < MARKET_CACHE_SECONDS:
            cached_response = dict(market_cache["data"])
            cached_response["debug"] = dict(market_cache["debug"])
            cached_response["cache"] = "hit"
            return jsonify(cached_response)

        data, debug = fetch_market_snapshot()
        usdinr_rate, fx_attempts = fetch_usdinr_rate()

        if usdinr_rate is not None:
            if data.get("gold") != "Unavailable":
                data["gold"] = round(float(data["gold"]) * usdinr_rate, 2)
            if data.get("oil") != "Unavailable":
                data["oil"] = round(float(data["oil"]) * usdinr_rate, 2)

        data["usdinr"] = usdinr_rate if usdinr_rate is not None else "Unavailable"
        debug["fx"] = {
            "ticker": "USDINR=X",
            "status": "ok" if usdinr_rate is not None else "unavailable",
            "attempts": fx_attempts,
        }
        has_any_success = any(value != "Unavailable" for value in data.values())

        if has_any_success:
            market_cache["data"] = data
            market_cache["debug"] = debug
            market_cache["timestamp"] = time.time()
            response = dict(data)
            response["debug"] = debug
            response["cache"] = "miss"
            return jsonify(response)

        if market_cache["data"] is not None:
            response = dict(market_cache["data"])
            response["debug"] = debug
            response["cache"] = "stale-success"
            return jsonify(response)

        market_cache["data"] = data
        market_cache["debug"] = debug
        market_cache["timestamp"] = time.time()
        response = dict(data)
        response["debug"] = debug
        response["cache"] = "miss"
        return jsonify(response)

    except Exception as exc:
        print("Market API Error:", exc)
        fallback = market_cache["data"] or {
            "gold": "Unavailable",
            "oil": "Unavailable",
            "nifty": "Unavailable",
            "sensex": "Unavailable",
        }
        debug = market_cache["debug"] or {}
        for label in ("gold", "oil", "nifty", "sensex"):
            debug.setdefault(
                label,
                {"ticker": label, "status": "error", "attempts": [f"route failure: {exc}"]},
            )

        response = dict(fallback)
        response["debug"] = debug
        response["cache"] = "stale" if market_cache["data"] is not None else "empty"
        response["route_error"] = str(exc)
        return jsonify(response)
    
    
@app.route("/api/home-metrics")
def api_home_metrics():
    return jsonify(build_home_metrics())



if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=debug_mode)
