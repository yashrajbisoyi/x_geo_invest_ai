from pathlib import Path
import subprocess
import sys
import time

BASE_DIR = Path(__file__).resolve().parent

STEPS = [
    "fetch_news.py",
    "sentiment_analysis.py",
    "geopolitical_risk.py",
    "investment_engine.py",
    "ml_model.py",
    "confidence_score.py",
]


def main():
    overall_start = time.perf_counter()

    for step in STEPS:
        step_start = time.perf_counter()
        print(f"Running {step}...")
        subprocess.run([sys.executable, str(BASE_DIR / step)], check=True)
        print(f"Completed {step} in {time.perf_counter() - step_start:.2f}s")

    print(f"Full workflow completed in {time.perf_counter() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
