# src/feature_engineering.py
import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_and_engineer_features(db_path="shadow_ranking_project/data/marketplace_data.db"):
    # establish connection to the database
    conn = duckdb.connect(db_path)
    
    # xtract all rows from search listings table into a dataframe
    query = "select * from search_listings;"
    df = conn.execute(query).fetch_df()
    conn.close()
    
    # define structural target variable (top 15 percent of listings within category)
    # in an engine audit, predicting premium positioning exposes visibility drivers
    df["is_top_tier"] = (df["organic_search_rank"] <= 1500).astype(int)
    
    # process categorical attributes using one hot encoding vectors
    df_encoded = pd.get_dummies(df, columns=["category"], drop_first=False)
    
    # split features and target matrices, exclude identity values
    # we explicitly drop tracking labels that would cause data leakage
    feature_columns = [col for col in df_encoded.columns if col not in [
        "listing_id", "organic_search_rank", "is_top_tier"
    ]]
    
    x = df_encoded[feature_columns]
    y = df_encoded["is_top_tier"]
    
    # 80/20 train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print("feature engineering module executed.")
    print(f"training matrix shape: {x_train.shape} | testing matrix shape: {x_test.shape}")
    
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    extract_and_engineer_features()
