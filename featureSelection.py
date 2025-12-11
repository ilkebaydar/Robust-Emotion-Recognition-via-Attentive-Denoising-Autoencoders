import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score
import random
import os
import traceback
#import matplotlib.pyplot as plt


SEED = 35
random.seed(SEED)
np.random.seed(SEED)

#SAFE CORE CALCULATION (prevent overheating and halting unexpectedly)
total_cores = os.cpu_count()
SAFE_CORES = max(2, int(total_cores /2))

print(f"SYSTEM CHECK")
print(f"Total Cores Detected: {total_cores}")
print(f"Using Safe Cores: {SAFE_CORES}")

#1st step: load both train and test dataframes
def load_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

#2nd step: prepare data with separating features, labels and groups
def prepare_data(train_df, test_df):
    #all feature columns except label and person_id
    X =train_df.iloc [:, :-2].values #X: features (excepts last two column)
    y=train_df.iloc[:, -2].values.astype(int) # y:labels (only last two column)
    groups = train_df.iloc[:, -1].values #groups:person_id

    X_test = test_df.values

    return X, y, groups, X_test
def get_feature_importance_ranking(X, y):
    print("\n CALCULATE FEATURE IMPORTANCE")
    model = lgb.LGBMClassifier(
        random_state=SEED,
        verbose= -1,
        n_estimators= 100,
        n_jobs= SAFE_CORES
    )
    model.fit(X, y)

    importances = model.feature_importances_
    ranked_indices = np.argsort(importances)[::-1]

    return ranked_indices

def evaluate_feature_count(X, y, groups, X_test, ranked_indices, top_n):
    print(f"\nTesting Top {top_n} Features...")
    
    # 1. Slice Data
    if top_n == "ALL":
        current_indices = ranked_indices # Use all
        print(f"Using ALL {len(ranked_indices)} features.")
    else:
        current_indices = ranked_indices[:top_n]
    
    X_subset = X[:, current_indices]
    X_test_subset = X_test[:, current_indices]
    
    # 2. Run Group CV
    group_k_fold = GroupKFold(n_splits=5)
    f1_scores = []
    test_preds_accum = np.zeros((X_test.shape[0], 4))
    
    for train_idx, val_idx in group_k_fold.split(X_subset, y, groups=groups):
        X_train, X_val = X_subset[train_idx], X_subset[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf = lgb.LGBMClassifier(
            random_state=SEED, 
            verbose=-1,
            n_estimators=300,    # Kept moderate for speed during search
            learning_rate=0.05,
            n_jobs=SAFE_CORES
        )
        clf.fit(X_train, y_train)
        
        val_preds = clf.predict(X_val)
        score = f1_score(y_val, val_preds, average='macro')
        f1_scores.append(score)
        
        test_preds_accum += clf.predict_proba(X_test_subset)
        
    avg_score = np.mean(f1_scores)
    print(f"-> Result for Top {top_n}: {avg_score:.5f}")
    
    return avg_score, test_preds_accum

   

if __name__ == "__main__":
    try: 
        train_df, test_df = load_data()
        X, y, groups, X_test = prepare_data(train_df, test_df)

        ranked_indices = get_feature_importance_ranking(X, y)
       
        candidates = [50, 100, 200, 300, 350, 400, 500, 750, 1000, 1200, 1793]
        best_score = -1.0
        best_n = 0
        best_preds=None

        print(f"\START GRID SEARCH FOR OPTIMAL FEATURE COUNT")
        for n in candidates:
            if n == 1793:
                score, preds = evaluate_feature_count(X, y, groups, X_test, ranked_indices, "ALL")
            else:
                score, preds = evaluate_feature_count(X, y, groups, X_test, ranked_indices, n)
            if score >best_score:
                best_score = score
                best_n = n
                best_preds =preds
                print(f"NEW BEST FOUND! (N={n}, Score={score:.5f})")

        print(f"Best Number of Features: {best_n}")
        print(f"Best Group CV Score:     {best_score:.5f}")

     
    except Exception:
        print("\n CRITICAL ERROR DETAILS!")
        traceback.print_exc()
