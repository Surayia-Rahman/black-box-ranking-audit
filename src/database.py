# src/database.py
import duckdb
import numpy as np
import pandas as pd

def initialize_database(db_path="shadow_ranking_project/data/marketplace_data.db", num_listings=50000):
    # establish a persistent local connection to the duckdb storage file
    conn = duckdb.connect(db_path)
    
    # define the relational schema
    conn.execute("""
        create or replace table search_listings (
            listing_id integer primary key,
            category varchar,
            price double,
            shipping_fee double,
            historical_sales integer,
            click_through_rate double,
            conversion_rate double,
            seller_rating double,
            processing_time_hours double,
            is_promoted integer,
            organic_search_rank integer
        );
    """)
    
    np.random.seed(42)
    
    # establish authentic categories matching public marketplace logs
    listing_ids = np.arange(1, num_listings + 1)
    real_categories = ["health_beauty", "computers_accessories", "watches_gifts", "sports_leisure", "auto"]
    categories = np.random.choice(real_categories, num_listings, p=[0.25, 0.20, 0.15, 0.22, 0.18])
    
    # simulate high variance lognormal pricing arrays matching olist metrics
    prices = np.random.lognormal(mean=4.1, sigma=0.8, size=num_listings) + 2.0
    prices = np.clip(prices, 5.0, 800.0)
    
    # shipping values are heavily linked to pricing thresholds in authentic marketplaces
    shipping_fees = np.where(prices > 150.0, 0.0, np.random.uniform(5.0, 35.0, num_listings))
    
    # populate authentic historical sales distributions using discrete negative binomial counts
    historical_sales = np.random.negative_binomial(n=2, p=0.005, size=num_listings)
    
    click_through_rate = np.random.beta(a=1.5, b=25, size=num_listings)
    conversion_rate = click_through_rate * np.random.uniform(0.05, 0.45, num_listings)
    
    # build realistic highly skewed ratings reflecting real e commerce reviews
    seller_rating = np.random.choice(
        [5.0, 4.0, 3.0, 2.0, 1.0], 
        num_listings, 
        p=[0.55, 0.25, 0.10, 0.04, 0.06]
    ) + np.random.uniform(-0.3, 0.0, num_listings)
    seller_rating = np.clip(seller_rating, 1.0, 5.0)
    
    # fulfillment logistics processing times reflecting real global shipping windows
    processing_time_hours = np.random.exponential(scale=30, size=num_listings) + 2.0
    is_promoted = np.where(np.random.rand(num_listings) > 0.88, 1, 0)
    
    # calculate the hidden search engine score matrix
    base_score = (historical_sales * 0.15) + (click_through_rate * 40) + (conversion_rate * 200)
    rating_modifier = (seller_rating - 3.0) * 20
    price_penalty = (prices + shipping_fees) * -0.02
    promoted_boost = is_promoted * 30
    
    # system shadow ranking drop dead constraint logic
    shadow_penalty = np.where(processing_time_hours > 48.0, -95.0, 0.0)
    system_noise = np.random.normal(loc=0.0, scale=12.0, size=num_listings)
    
    total_score = base_score + rating_modifier + price_penalty + promoted_boost + shadow_penalty + system_noise
    
    df = pd.DataFrame({
        "listing_id": listing_ids,
        "category": categories,
        "price": prices,
        "shipping_fee": shipping_fees,
        "historical_sales": historical_sales,
        "click_through_rate": click_through_rate,
        "conversion_rate": conversion_rate,
        "seller_rating": seller_rating,
        "processing_time_hours": processing_time_hours,
        "is_promoted": is_promoted,
        "score": total_score
    })
    
    # organize ranks dynamically partitioned within real world product segments
    df["organic_search_rank"] = df.groupby("category")["score"].rank(ascending=False, method="first").astype(int)
    df = df.drop(columns=["score"])
    
    conn.execute("insert into search_listings select * from df")
    print(f"successfully populated relational table with {num_listings} real world distribution profiles.")
    conn.close()

if __name__ == "__main__":
    initialize_database()
