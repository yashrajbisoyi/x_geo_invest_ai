from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "final_combined_news.csv"
EVAL_PATH = BASE_DIR / "artifacts" / "model_evaluation.json"


def main():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {PROCESSED_PATH}")

    df = pd.read_csv(PROCESSED_PATH)

    df["sentiment"].value_counts().plot(kind="bar", title="Sentiment Distribution")
    plt.tight_layout()
    plt.show()

    df["geo_risk_level"].value_counts().plot(kind="bar", title="Geopolitical Risk Levels")
    plt.tight_layout()
    plt.show()

    df["investment_recommendation"].value_counts().plot(
        kind="barh", title="Investment Recommendations"
    )
    plt.tight_layout()
    plt.show()

    if EVAL_PATH.exists():
        with open(EVAL_PATH, "r", encoding="utf-8") as file:
            evaluation = json.load(file)

        comparison = pd.DataFrame(evaluation.get("model_comparison", []))
        if not comparison.empty:
            comparison.set_index("name")[["accuracy", "f1_weighted"]].plot(
                kind="bar", title="Model Comparison"
            )
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
