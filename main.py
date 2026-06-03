# main.py
import sys
import os

# append the project root directory path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.database import initialize_database
from src.feature_engineering import extract_and_engineer_features
from src.audit_model import train_and_audit_engine
from src.explainability import generate_engine_explanations

def run_system_audit_pipeline():
    print("starting...")
    
    # database verification and row generation pipeline phase
    db_path = "shadow_ranking_project/data/marketplace_data.db"
    if not os.path.exists(db_path):
        print("local database storage file not discovered. initializing fallback...")
        initialize_database(db_path=db_path)
    else:
        print(f"verified database instance at connection endpoint: {db_path}")
        
    # extract records and execute training matrix transformations
    x_train, x_test, y_train, y_test = extract_and_engineer_features(db_path=db_path)
    
    # execute forest model training loops and extract global score arrays
    fitted_model, feature_rankings = train_and_audit_engine(x_train, x_test, y_train, y_test)
    
    # calculate localized game theoretic impact metrics and export plot visual
    plot_output_path = generate_engine_explanations(fitted_model, x_test)
    
    print("algorithmic shadow ranking pipeline completed")

if __name__ == "__main__":
    run_system_audit_pipeline()
