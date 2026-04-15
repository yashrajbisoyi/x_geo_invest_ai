from pathlib import Path
import pickle

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATA_PATH_CANDIDATES = [
    PROCESSED_DIR / "runtime_combined_news.csv",
    PROCESSED_DIR / "final_combined_news.csv",
]
MODEL_PATH = ARTIFACTS_DIR / "best_model.pkl"


def first_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


DATA_PATH = first_existing_path(DATA_PATH_CANDIDATES)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model artifact: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["text"] = df["title"].fillna("").astype(str) + " " + df["content"].fillna("").astype(str)

    # Add missing columns with defaults if not present
    if "dataset_type" not in df.columns:
        df["dataset_type"] = "combined"
    if "is_geopolitical" not in df.columns:
        df["is_geopolitical"] = True
    if "is_financial" not in df.columns:
        df["is_financial"] = True

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The saved model does not support predict_proba.")

    expected_columns = ["text"]
    preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
    if preprocessor is not None:
        for name, transformer, columns in getattr(preprocessor, "transformers", []):
            if name == "text":
                continue
            if isinstance(columns, list):
                expected_columns.extend(columns)

    default_values = {
        "source": "Unknown",
        "dataset_type": "combined",
        "geo_risk_level": "Low",
        "sentiment": "Neutral",
        "is_geopolitical": True,
        "is_financial": True,
        "energy_risk": False,
    }

    for column in expected_columns:
        if column not in df.columns:
            df[column] = default_values.get(column, "")

    feature_frame = df[expected_columns]

    probabilities = model.predict_proba(feature_frame).max(axis=1)
    df["confidence_score_%"] = (probabilities * 100).round(2)

    output_path = ARTIFACTS_DIR / "final_with_confidence.csv"
    df.to_csv(output_path, index=False)

    print(f"Confidence scores added -> {output_path}")
    print(df[["investment_recommendation", "confidence_score_%"]].head())


if __name__ == "__main__":
    main()
