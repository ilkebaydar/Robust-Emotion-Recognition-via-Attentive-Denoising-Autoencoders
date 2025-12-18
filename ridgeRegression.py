import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import random
import os
import traceback

# --- CONFIGURATION ---
# C=0.01 implies strong regularization (Ridge).
# Helps to handle noise in 1793 features.
C_VALUE = 0.01 
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# Safe Core Calculation
total_cores = os.cpu_count()
SAFE_CORES = max(2, int(total_cores / 2))

print(f"--- SYSTEM CHECK ---")
print(f"Model: Logistic Regression (LBFGS) | Features: ALL | C={C_VALUE}")
print(f"Using Safe Cores: {SAFE_CORES}")

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def prepare_data(train_df, test_df):
    print("Preparing data...")
    # Selecting ALL features
    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    groups = train_df.iloc[:, -1].values 
    X_test = test_df.values
    return X, y, groups, X_test

def train_linear_model(X, y, groups, X_test):
    
    test_probs_accum = np.zeros((X_test.shape[0], 4))
    
    print(f"\n--- Starting Linear Training (Group CV) ---")
    
    group_k_fold = GroupKFold(n_splits=5)
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(group_k_fold.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 1. SCALING (MANDATORY FOR LINEAR MODELS)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        

        clf = LogisticRegression(
            C=C_VALUE,
            penalty='l2',
            solver='lbfgs',      
            max_iter=3000,       # High iteration count to ensure convergence
            random_state=SEED,
            n_jobs=SAFE_CORES,   # lbfgs supports parallel processing!
            verbose=0
        )
        
        clf.fit(X_train_scaled, y_train)
        
        val_preds = clf.predict(X_val_scaled)
        score = f1_score(y_val, val_preds, average='macro')
        f1_scores.append(score)
        
        print(f"Fold {fold+1} Linear F1-Macro: {score:.4f}")
        
        test_probs_accum += clf.predict_proba(X_test_scaled)

    avg_score = np.mean(f1_scores)
    print(f"Average Linear Group CV Score: {avg_score:.4f}")
    
    return test_probs_accum, avg_score

def create_submission_file(test_probabilities):
    final_preds = np.argmax(test_probabilities, axis=1)
    
    submission_df = pd.DataFrame({
        'ID': range(len(final_preds)), 
        'Predicted': final_preds
    })
    
    filename = 'submission_linear_all.csv'
    submission_df.to_csv(filename, index=False)
    print(f"\nFile ready: '{filename}'")

if __name__ == "__main__":
    try:
        train_df, test_df = load_data()
        X, y, groups, X_test = prepare_data(train_df, test_df)
        
        test_probs, score = train_linear_model(X, y, groups, X_test)
        
        create_submission_file(test_probs)
        
        print(f"\nFINAL LINEAR SCORE (Local): {score:.5f}")
        
    except Exception:
        print("\nCRITICAL ERROR DETAILS:")
        traceback.print_exc()