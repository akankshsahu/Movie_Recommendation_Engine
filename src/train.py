from src.models import train_svd
import pandas as pd
from surprise import Dataset, Reader

def main():
    ratings = pd.read_parquet("data/processed/ratings.parquet")
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    svd_model = train_svd(data)
    print("Trained SVD model successfully.")

if __name__ == "__main__":
    main()
