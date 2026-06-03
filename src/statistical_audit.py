# src/statistical_audit.py
import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def perform_statistical_audit(db_path="shadow_ranking_project/data/marketplace_data.db", output_dir="shadow_ranking_project/data"):
    print("initiating formal mathematical and statistical audit...")
    
    # establish database connection and pull target tracking variables
    conn = duckdb.connect(db_path)
    query = "select organic_search_rank, processing_time_hours, seller_rating from search_listings;"
    df = conn.execute(query).fetch_df()
    conn.close()
    
    # segment data into pre threshold and post threshold processing windows
    under_48 = df[df["processing_time_hours"] <= 48.0]["organic_search_rank"]
    over_48 = df[df["processing_time_hours"] > 48.0]["organic_search_rank"]
    
    print("\n--- statistical validation: two sample welch t test ---")
    # execute welchs t test to account for unequal sample sizes and variances
    t_stat, p_val = stats.ttest_ind(under_48, over_48, equal_var=False)
    print(f"calculated t-statistic: {t_stat:.4f}")
    print(f"empirical p-value: {p_val:.4e} (significance threshold alpha = 0.01)")
    

    plt.figure(figsize=(9, 5))
    
    # plot the kernel density or histogram distribution for compliant vs late windows
    plt.hist(under_48, bins=50, alpha=0.6, label="compliant processing window (<= 48h)", color="#1f77b4", density=True)
    plt.hist(over_48, bins=50, alpha=0.6, label="penalized shadow window (> 48h)", color="#d62728", density=True)
    
    plt.title("audit distribution plot: systemic rank displacement profile", fontsize=11, pad=15)
    plt.xlabel("assigned organic search rank (lower position number implies top visibility)", fontsize=9)
    plt.ylabel("probability density profile", fontsize=9)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "distributional_rank_audit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\ndistributional plot generated successfully. saved to: {plot_path}")
    return t_stat, p_val

if __name__ == "__main__":
    import sys
    sys.path.append("shadow_ranking_project")
    perform_statistical_audit()
