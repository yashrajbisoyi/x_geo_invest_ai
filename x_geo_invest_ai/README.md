# GeoFinance AI

GeoFinance AI is a source-aware pipeline and Flask app for studying geopolitical and financial news relationships and converting them into investment guidance.

## Scope alignment

The repository now supports:

- verified-source extraction through NewsAPI domain filtering for Reuters, BBC, and Bloomberg
- separate geopolitical, financial, and combined datasets
- preprocessing for sentiment and geopolitical risk
- recommendation generation for the combined workflow
- ML training and evaluation with accuracy, weighted F1, and execution-time reporting
- a web app for analytics, news inspection, performance review, and prediction demos
- optional live news feed via `newsdata.io` or X with on-the-fly recommendation tags

See `docs/project_scope_mapping.md` for the bullet-by-bullet mapping.

## Repository structure

- `data/raw/` stores extracted geopolitical, financial, and combined news datasets
- `data/processed/` stores sentiment, risk, and final recommendation datasets
- `artifacts/` stores trained-model outputs and evaluation reports
- `website/` contains the Flask app and templates
- `docs/project_scope_mapping.md` tracks scope coverage against your report bullets

## Workflow

1. Data extraction: `fetch_news.py`
2. Sentiment preprocessing: `sentiment_analysis.py`
3. Geopolitical risk scoring: `geopolitical_risk.py`
4. Recommendation generation: `investment_engine.py`
5. Model training and evaluation: `ml_model.py`
6. Confidence scoring: `confidence_score.py`
7. Full pipeline runner: `run_pipeline.py`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key
X_BEARER_TOKEN=your_x_bearer_token
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWSDATA_API_KEY=your_newsdata_key
FLASK_DEBUG=true
```

## Run the full project workflow

```bash
python run_pipeline.py
```

This creates:

- `data/raw/geopolitical_news.csv`
- `data/raw/financial_news.csv`
- `data/raw/combined_news.csv`
- `data/processed/final_combined_news.csv`
- `artifacts/model_evaluation.json`
- `artifacts/best_model.pkl`
- `artifacts/final_with_confidence.csv`

## Run the web app

```bash
python website/app.py
```

Then open `http://127.0.0.1:5000`.

## Important limitations

- historical depth depends on your NewsAPI plan and may not cover the full research horizon you want
- the current labels are still rule-generated, so the ML stage is evaluating pattern recovery rather than expert-annotated market truth
- the repository now has a place for scope mapping, but report citations must still be added by the team for the final paper
- the live news section prefers `NEWSDATA_API_KEY` and can fall back to `X_BEARER_TOKEN`
- gold and oil market fallback works better if you provide `ALPHA_VANTAGE_API_KEY`
