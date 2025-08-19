# Recommendation System (MovieLens) — End‑to‑End

**Goal:** Build and deploy a hybrid recommender (popularity, collaborative filtering, content-based) with a Streamlit app and simple pipeline.

## Start
```bash
# 1) Install deps (ideally in a new venv)
pip install -r requirements.txt

# 2) (Optional) Download MovieLens latest datasets and place ratings.csv, movies.csv in data/raw/
#    https://grouplens.org/datasets/movielens/

# 3) Prepare data (will auto-generate tiny sample if MovieLens not found)
python src/data_prep.py

# 4) Train models (popularity + matrix factorization via Surprise)
python src/train.py

# 5) Evaluate
python src/evaluate.py

# 6) Launch the app
streamlit run app.py

# 7) (Optional) Run end-to-end pipeline
python run_pipeline.py
```

## Structure
```
data/
  raw/        # place MovieLens CSVs here (ratings.csv, movies.csv)
  processed/  # cleaned matrices and artifacts
models/       # saved model files
src/
  data_prep.py
  models.py
  train.py
  evaluate.py
  recommend.py
  pipeline.py
app.py
requirements.txt
```

## Notes
- If `data/raw/ratings.csv` isn't present, scripts create a **toy dataset** so everything runs.
- The Streamlit app lets you pick a movie and shows top-N recommendations.
