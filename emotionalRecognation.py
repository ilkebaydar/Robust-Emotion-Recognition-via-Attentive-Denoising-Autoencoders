import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import random
import os
import traceback

TOP_N_FEATURES = 50 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Safe Core Calculation
total_cores = os.cpu_count()
SAFE_CORES = max(2, int(total_cores / 2))

print(f"--- SYSTEM CHECK ---")
print(f"Using Safe Cores: {SAFE_CORES}")
print(f"Target: Using Top {TOP_N_FEATURES} Features (Optimized)")

def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def prepare_data(train_df, test_df):
    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    groups = train_df.iloc[:, -1].values 
    X_test = test_df.values
    return X, y, groups, X_test

def select_top_n_features(X, y, X_test, n):
    """
    Selects the top N features based on a quick LightGBM check.
    """
    print(f"\n--- Selecting Top {n} Features ---")
    
    # Quick model for feature importance
    model = lgb.LGBMClassifier(
        random_state=SEED, 
        verbose=-1, 
        n_estimators=100,
        n_jobs=SAFE_CORES
    )
    model.fit(X, y)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n]
    
    print(f"Selection complete. Reducing from {X.shape[1]} to {n} features.")
    
    return X[:, top_indices], X_test[:, top_indices]

def train_and_evaluate(X, y, groups, X_test):
    
    test_predictions_accumulated = np.zeros((X_test.shape[0], 4))
    
    print(f"\n--- Starting Training (Group CV) with {X.shape[1]} Features ---")
    
    group_k_fold = GroupKFold(n_splits=5)
    f1_scores_group = []
    
    for fold_index, (train_idx, val_idx) in enumerate(group_k_fold.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Main Model (Tuned settings)
        classifier = lgb.LGBMClassifier(
            random_state=SEED, 
            verbose=-1,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=SAFE_CORES
        )
        
        classifier.fit(X_train, y_train)
        
        val_preds = classifier.predict(X_val)
        score = f1_score(y_val, val_preds, average='macro')
        f1_scores_group.append(score)
        
        print(f"Fold {fold_index+1} F1-Macro: {score:.4f}")
        
        test_predictions_accumulated += classifier.predict_proba(X_test)

    avg_score = np.mean(f1_scores_group)
    print(f"Average Group CV Score: {avg_score:.4f}")
    
    return test_predictions_accumulated, avg_score

def create_submission_file(test_probabilities):
    final_class_predictions = np.argmax(test_probabilities, axis=1)
    
    submission_df = pd.DataFrame({
        'ID': range(len(final_class_predictions)), 
        'Predicted': final_class_predictions
    })
    
   
    filename = 'submission_feat_50.csv'
    submission_df.to_csv(filename, index=False)
    print(f"\nSubmission file ready: '{filename}'")

if __name__ == "__main__":
    try:
        train_df, test_df = load_data()
        X, y, groups, X_test = prepare_data(train_df, test_df)
        
        # 1. Feature Selection (Using the hardcoded best number: 50)
        X_reduced, X_test_reduced = select_top_n_features(X, y, X_test, TOP_N_FEATURES)
        
        # 2. Train & Evaluate
        test_probs, score = train_and_evaluate(X_reduced, y, groups, X_test_reduced)
        
        # 3. Create File
        create_submission_file(test_probs)
        
        print(f"\nFINAL RESULT: {score:.5f}")
        
    except Exception:
        print("\nCRITICAL ERROR:")
        traceback.print_exc()