from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from src.models import train_svd
import pandas as pd

def main():
    # Load processed ratings
    ratings = pd.read_parquet("data/processed/ratings.parquet")

    # Build Surprise dataset
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    # Train/test split
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train SVD
    algo = train_svd(data)
    algo.fit(trainset)

    # Make predictions on test set
    predictions = algo.test(testset)

    # Compute RMSE and MAE
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)

    print(f"Evaluation results - RMSE: {rmse:.4f}, MAE: {mae:.4f}")

if __name__ == "__main__":
    main()
