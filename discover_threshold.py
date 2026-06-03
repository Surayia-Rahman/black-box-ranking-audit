# discover_threshold.py
import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os

def run_automated_threshold_discovery():
    print("starting automated black-box discovery")
    
    db_path = "shadow_ranking_project/data/marketplace_data.db"
    output_dir = "shadow_ranking_project/data"
    
    # initialize database connection and fetch target arrays
    conn = duckdb.connect(db_path)
    query = "select organic_search_rank, processing_time_hours from search_listings;"
    df = conn.execute(query).fetch_df()
    conn.close()
    
    # define candidate search space thresholds
    candidate_thresholds = np.arange(12, 86, 2)
    effect_sizes = []
    
    print("sweeping processing latency windows and calculating cohen's d effect sizes...")
    
    for cutoff in candidate_thresholds:
        under_split = df[df["processing_time_hours"] <= cutoff]["organic_search_rank"]
        over_split = df[df["processing_time_hours"] > cutoff]["organic_search_rank"]
        
        if len(under_split) < 100 or len(over_split) < 100:
            effect_sizes.append(0.0)
            continue
            
        # compute cohen's d to quantify population separation magnitude
        mean_diff = under_split.mean() - over_split.mean()
        pooled_sd = np.sqrt((under_split.var() + over_split.var()) / 2.0)
        cohens_d = mean_diff / pooled_sd
        effect_sizes.append(abs(cohens_d))
        
    # locate the threshold that maximizes population variance
    optimal_idx = np.argmax(effect_sizes)
    discovered_hours = candidate_thresholds[optimal_idx]
    peak_effect_size = effect_sizes[optimal_idx]
    
    print("\n--- discovery report ---")
    print(f"identified system fracture point: {discovered_hours} hours latency")
    print(f"maximal practical effect size (cohen's d): {peak_effect_size:.2f}")
    
    # execute conditional verification checks at the optimal split node
    under_final = df[df["processing_time_hours"] <= discovered_hours]["organic_search_rank"]
    over_final = df[df["processing_time_hours"] > discovered_hours]["organic_search_rank"]
    
    t_stat, p_val = stats.ttest_ind(under_final, over_final, equal_var=False)
    print(f"welch's t-statistic at discovery point: {t_stat:.4f}")
    print(f"empirical p-value profile: {p_val:.4e}")
    
    # plot empirical density distributions at the discovered cutoff
    plt.figure(figsize=(9, 5))
    plt.hist(under_final, bins=50, alpha=0.6, label=f"compliant window (<={discovered_hours}h)", color="#1f77b4", density=True)
    plt.hist(over_final, bins=50, alpha=0.6, label=f"penalized shadow window (>{discovered_hours}h)", color="#d62728", density=True)
    
    plt.title(f"systemic rank displacement profile (discovered cutoff: {discovered_hours}h | d = {peak_effect_size:.2f})", fontsize=11, pad=15)
    plt.xlabel("assigned organic search rank (lower position number implies top visibility)", fontsize=9)
    plt.ylabel("probability density profile", fontsize=9)
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "discovered_change_point_audit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nstatistical validation complete")

if __name__ == "__main__":
    run_automated_threshold_discovery()
