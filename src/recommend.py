import pandas as pd, numpy as np, joblib

PROC='data/processed'
MODEL='models/svd.joblib'

def top_n_for_item(movie_id, n=5):
    movies = pd.read_parquet(f'{PROC}/movies.parquet')
    ratings = pd.read_parquet(f'{PROC}/ratings.parquet')
    # Simple: recommend by average rating of other movies (toy) — placeholder for item-item similarity
    mean_scores = ratings.groupby('movieId').rating.mean().sort_values(ascending=False)
    recs = mean_scores.drop(labels=[movie_id], errors='ignore').head(n).index.tolist()
    return movies[movies.movieId.isin(recs)][['movieId','title']].to_dict(orient='records')

def top_n_for_user(user_id, n=5):
    algo = joblib.load(MODEL)
    movies = pd.read_parquet(f'{PROC}/movies.parquet')
    seen = set(pd.read_parquet(f'{PROC}/ratings.parquet').query('userId==@user_id').movieId.tolist())
    preds = []
    for mid in movies.movieId:
        if mid in seen: 
            continue
        preds.append((mid, algo.predict(user_id, mid).est))
    preds.sort(key=lambda x: x[1], reverse=True)
    mids = [m for m,_ in preds[:n]]
    return movies[movies.movieId.isin(mids)][['movieId','title']].to_dict(orient='records')
