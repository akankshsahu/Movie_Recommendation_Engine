import os, pandas as pd, numpy as np, json

RAW = 'data/raw'
PROC = 'data/processed'
os.makedirs(PROC, exist_ok=True)

ratings_fp = os.path.join(RAW, 'ratings.csv')
movies_fp  = os.path.join(RAW, 'movies.csv')

if not (os.path.exists(ratings_fp) and os.path.exists(movies_fp)):
    # Create toy data
    users = [1,2,3,4,5]
    items = [101,102,103,104,105]
    ratings = []
    rng = np.random.default_rng(42)
    for u in users:
        for i in items:
            ratings.append([u,i,int(rng.integers(3,6))])
    ratings_df = pd.DataFrame(ratings, columns=['userId','movieId','rating'])
    movies_df  = pd.DataFrame({
        'movieId': items,
        'title': ['Toy Story', 'Matrix', 'Inception', 'Interstellar', 'Spirited Away'],
        'genres': ['Animation|Adventure','Action|Sci-Fi','Thriller|Sci-Fi','Sci-Fi|Drama','Animation|Fantasy']
    })
else:
    ratings_df = pd.read_csv(ratings_fp)
    movies_df  = pd.read_csv(movies_fp)

# Basic clean
ratings_df = ratings_df[['userId','movieId','rating']].dropna()
movies_df  = movies_df[['movieId','title','genres']].dropna()

ratings_df.to_parquet(os.path.join(PROC, 'ratings.parquet'), index=False)
movies_df.to_parquet(os.path.join(PROC,  'movies.parquet'), index=False)
print('Saved processed ratings & movies.')
