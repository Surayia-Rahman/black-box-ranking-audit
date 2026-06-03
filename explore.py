# explore.py
import duckdb

def run_data_audit(db_path="shadow_ranking_project/data/marketplace_data.db"):
    # establish connection to our local relational database file
    conn = duckdb.connect(db_path)
    
    print("audit log 1: high level distribution summary per category ---")
    # check average metrics across different marketplace categories
    query_1 = """
        select 
            category,
            count(*) as total_items,
            round(avg(price), 2) as avg_price,
            round(avg(seller_rating), 2) as avg_seller_rating,
            round(avg(processing_time_hours), 1) as avg_processing_hours,
            round(avg(organic_search_rank), 0) as avg_assigned_rank
        from search_listings
        group by category
        order by total_items desc;
    """
    print(conn.execute(query_1).fetch_df().to_string(index=False))
    
    print("\n-audit log 2: profiling top 5 ranked listings per category ---")
    # use window partitioning to inspect the attributes of premium ranked items
    query_2 = """
        with ranked_snapshots as (
            select 
                category,
                listing_id,
                organic_search_rank,
                price,
                round(seller_rating, 2) as rating,
                round(processing_time_hours, 1) as proc_hours,
                is_promoted
            from search_listings
        )
        select * from ranked_snapshots
        where organic_search_rank <= 3
        order by category, organic_search_rank;
    """
    print(conn.execute(query_2).fetch_df().to_string(index=False))

    print("\naudit log 3: identifying anomalies in the top 100 spots ---")
    # search for listings with pristine seller ratings that are buried with lack of reasoning
    query_3 = """
        select 
            category,
            count(*) as buried_high_quality_sellers
        from search_listings
        where organic_search_rank > 500
          and seller_rating >= 4.5
          and processing_time_hours > 48.0
        group by category;
    """
    print(conn.execute(query_3).fetch_df().to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    run_data_audit()
