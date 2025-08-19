import pandas as pd
from surprise import SVD, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- USER-BASED SVD --------------------
def train_svd(data, n_factors=50, n_epochs=30, lr_all=0.005, reg_all=0.02):
    """
    Train an SVD model on the given dataset.
    """
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    algo.fit(trainset)
    return algo


def get_top_n(algo, user_id, movies, ratings, n=10):
    """
    Return top-N movie recommendations for a user.
    """
    all_movie_ids = movies['movieId'].unique()
    rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()

    predictions = []
    for mid in all_movie_ids:
        if mid not in rated_movies:
            pred = algo.predict(user_id, mid)
            predictions.append((mid, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movie_ids = [mid for mid, _ in predictions[:n]]
    top_movies = movies[movies['movieId'].isin(top_movie_ids)]['title'].tolist()
    return top_movies

# -------------------- ITEM-BASED SIMILARITY --------------------
def build_item_similarity_matrix(ratings):
    """
    Build a cosine similarity matrix between movies.
    """
    movie_user_matrix = ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    movie_similarity = cosine_similarity(movie_user_matrix)
    movie_similarity_df = pd.DataFrame(
        movie_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )
    return movie_similarity_df


def get_similar_movies(movie_id, movies, movie_similarity_df, n=10):
    """
    Return top-N similar movies to the given movie_id using cosine similarity.
    """
    if movie_id not in movie_similarity_df.index:
        return []  # fallback if movie_id not found
    sim_scores = movie_similarity_df[movie_id].drop(movie_id)  # exclude itself
    top_ids = sim_scores.sort_values(ascending=False).head(n).index
    return movies[movies['movieId'].isin(top_ids)]['title'].tolist()
