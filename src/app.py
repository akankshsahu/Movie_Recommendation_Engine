import sys
import os

# Add project root (folder above src/) to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
from surprise import Dataset, Reader
from src.models import train_svd, get_top_n, build_item_similarity_matrix, get_similar_movies

# Load data
@st.cache_data
def load_data():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
    ratings = pd.read_parquet(os.path.join(base_path, "ratings.parquet"))
    movies = pd.read_parquet(os.path.join(base_path, "movies.parquet"))
    return ratings, movies

ratings, movies = load_data()

# Train SVD model with caching
@st.cache_resource
def train_svd_model(ratings):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    svd_model = train_svd(data)
    return svd_model

svd_model = train_svd_model(ratings)

# Build item similarity matrix with caching
@st.cache_resource
def build_similarity_matrix(ratings):
    return build_item_similarity_matrix(ratings)

item_sim_matrix = build_similarity_matrix(ratings)

# Streamlit UI
st.title("🎬 Movie Recommender System")
mode = st.radio("Recommendation mode:", ["User-based", "Item-based"])

if mode == "User-based":
    user_id = st.number_input("Enter a user ID:", min_value=1, max_value=int(ratings['userId'].max()), step=1)
    if st.button("Get Recommendations"):
        top_recs = get_top_n(svd_model, user_id, movies, ratings, n=10)
        st.write(f"Top movies for user {user_id}:")
        for i, rec in enumerate(top_recs, 1):
            st.write(f"{i}. {rec}")

elif mode == "Item-based":
    movie_choice = st.selectbox("Pick a movie you like:", movies['title'].values)
    if st.button("Get Similar Movies"):
        movie_id = movies[movies['title'] == movie_choice]['movieId'].values[0]
        similar_movies = get_similar_movies(movie_id, movies, item_sim_matrix, n=10)
        st.write(f"Movies similar to **{movie_choice}**:")
        for i, rec in enumerate(similar_movies, 1):
            st.write(f"{i}. {rec}")
