# src/explainability.py
import shap
import matplotlib.pyplot as plt
import os

def generate_engine_explanations(model, x_test, output_dir="shadow_ranking_project/data"):
    print("initializing shapley dependency auditing... (calculating local feature impacts)")
    
    # tree explainer optimized for forest structures
    explainer = shap.TreeExplainer(model)
    
    # compute shap values on a representative validation sampling
    sample_size = min(1000, len(x_test))
    x_sample = x_test.sample(sample_size, random_state=42)
    shap_values = explainer(x_sample)
    
    # isolate the target top tier prediction array (class 1 index)
    if len(shap_values.shape) == 3:
        shap_values_target = shap_values[:, :, 1]
    else:
        shap_values_target = shap_values
        
    plt.figure(figsize=(8, 5))
    
    # relationship between processing duration and localized log odds push
    shap.plots.scatter(
        shap_values_target[:, "processing_time_hours"], 
        show=False
    )
    
    plt.title("auditing system shadow rankings: processing time threshold impact", fontsize=11, pad=15)
    plt.xlabel("processing duration (hours)", fontsize=9)
    plt.ylabel("shap value (impact on top tier visibility probability)", fontsize=9)
    plt.axvline(x=48.0, color="red", linestyle="--", alpha=0.7, label="suspected shadow penalty threshold")
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "shadow_threshold_audit.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"visualization saved to: {plot_path}")
    return plot_path

if __name__ == "__main__":
    import sys
    sys.path.append("shadow_ranking_project")
    from src.feature_engineering import extract_and_engineer_features
    from src.audit_model import train_and_audit_engine
    
    x_tr, x_te, y_tr, y_te = extract_and_engineer_features()
    fitted_model, _ = train_and_audit_engine(x_tr, x_te, y_tr, y_te)
    generate_engine_explanations(fitted_model, x_te)
