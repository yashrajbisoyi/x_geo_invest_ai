from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RISK_KEYWORDS = [
    "war",
    "warfare",
    "armed conflict",
    "conflict",
    "hostilities",
    "invasion",
    "incursion",
    "occupation",
    "military action",
    "military operation",
    "airstrike",
    "missile",
    "drone strike",
    "bombing",
    "shelling",
    "naval clash",
    "nuclear",
    "nuclear threat",
    "atomic",
    "arms race",
    "weapons system",
    "geopolitical tension",
    "political tension",
    "instability",
    "unrest",
    "coup",
    "regime change",
    "martial law",
    "sanction",
    "embargo",
    "blockade",
    "trade restriction",
    "diplomatic crisis",
    "terrorism",
    "terrorist attack",
    "militant",
    "insurgent",
    "border dispute",
    "territorial dispute",
    "annexation",
    "oil disruption",
    "pipeline sabotage",
    "shadow fleet",
    "shipping disruption",
    "cyber attack",
    "hybrid warfare",
    "retaliation",
    "escalation",
    "security concern",
]


def calculate_risk(row):
    text = (str(row.get("title", "")) + " " + str(row.get("content", ""))).lower()
    return sum(1 for word in RISK_KEYWORDS if word in text)


def risk_level(score):
    if score >= 2:
        return "High"
    if score == 1:
        return "Medium"
    return "Low"


def process_file(path):
    df = pd.read_csv(path)
    df["geo_risk_score"] = df.apply(calculate_risk, axis=1)
    df["geo_risk_level"] = df["geo_risk_score"].apply(risk_level)
    output_path = PROCESSED_DIR / f"{path.stem.replace('_with_sentiment', '')}_with_risk.csv"
    df.to_csv(output_path, index=False)
    print(f"Geopolitical risk scoring completed for {path.name} -> {output_path.name}")


def main():
    for name in (
        "geopolitical_news_with_sentiment.csv",
        "financial_news_with_sentiment.csv",
        "combined_news_with_sentiment.csv",
    ):
        path = PROCESSED_DIR / name
        if path.exists():
            process_file(path)

    combined_output = PROCESSED_DIR / "combined_news_with_risk.csv"
    if combined_output.exists():
        pd.read_csv(combined_output).to_csv(BASE_DIR / "news_with_risk.csv", index=False)


if __name__ == "__main__":
    main()
