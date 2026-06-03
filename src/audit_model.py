# src/audit_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

def train_and_audit_engine(x_train, x_test, y_train, y_test):
    print("initiating random forest audit model training...")
    
    # initialize the model
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=12, 
        random_state=42, 
        n_jobs=-1
    )
    
    model.fit(x_train, y_train)
    
    # compute target predictions on unseen validation matrix
    y_pred = model.predict(x_test)
    
    print("\n--- model verification: classification matrix report ---")
    print(classification_report(y_test, y_pred, digits=4))
    
    # isolate feature importance weights to rank structural contributors
    importances = model.feature_importances_
    feature_names = x_train.columns
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance_weight": importances
    }).sort_values(by="importance_weight", ascending=False)
    
    print("--- model verification: extracted feature importance rankings ---")
    print(importance_df.to_string(index=False))
    
    return model, importance_df

if __name__ == "__main__":
    import sys
    sys.path.append("shadow_ranking_project")
    from src.feature_engineering import extract_and_engineer_features
    x_tr, x_te, y_tr, y_te = extract_and_engineer_features()
    train_and_audit_engine(x_tr, x_te, y_tr, y_te)
