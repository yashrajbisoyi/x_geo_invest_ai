# Project Scope Mapping

This file maps the intended project bullet points to the current repository implementation.

| Project bullet | Current status | Implementation notes |
| --- | --- | --- |
| Data extracted from verified E-News sources using APIs | Implemented with constraints | `fetch_news.py` now filters to Reuters, BBC, and Bloomberg domains through NewsAPI domain filtering. |
| Recent and historical periods collected | Partially implemented | The current extractor pulls a rolling 90-day window. Longer historical coverage depends on API plan limits and should be extended in future runs. |
| Focus areas include Geopolitics and Finance | Implemented | Separate geopolitical and financial queries are used. |
| Separate datasets for geopolitical, financial, and combined news | Implemented | The pipeline now writes `data/raw` and `data/processed` variants for all three dataset types. |
| Optimized keyword-based extraction | Implemented | Query and matching keyword sets are encoded in `fetch_news.py`. |
| Keywords selected based on research papers and prior studies | Partially implemented | The repository now isolates keyword lists for traceability, but the exact literature citations still need to be added to the report. |
| Research papers properly cited in the report | Pending documentation | A report citation section still needs to be filled with the team's selected papers. |
| Data cleaned and preprocessed before training | Implemented | Sentiment and risk preprocessing steps now run across the separated datasets. |
| ML models identify patterns and relationships | Implemented | `ml_model.py` trains and compares multiple ML models on the combined processed dataset. |
| Geopolitical-financial relationships analyzed | Implemented at summary level | Structured features and analytics summarize cross-signal relationships. |
| Suitable ML/DL models selected for prediction | Partially implemented | ML comparison is implemented; DL models are not yet included. |
| Prediction performance and execution time evaluated | Implemented | The training script writes accuracy, weighted F1, training time, and prediction time to `artifacts/model_evaluation.json`. |
| Existing patterns and possible improvements discussed | Implemented | Improvement notes are included in the evaluation artifact and can be used in the report. |
| Complete workflow: Data extraction -> Training -> Prediction | Implemented | `run_pipeline.py` executes the end-to-end workflow. |
