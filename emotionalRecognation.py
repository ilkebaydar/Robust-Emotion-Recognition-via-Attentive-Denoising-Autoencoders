import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import random
import os
import traceback
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
SAFE_CORES = max(2, int(os.cpu_count() / 2))

# Denenecek Tüm Adaylar
C_CANDIDATES = [0.05, 0.1, 0.5, 1.0]

print(f"--- SYSTEM CHECK ---")
print(f"Strategy: LASSO FACTORY (Generating file for EVERY C value)")

def load_data():
    print("Loading datasets...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df

def prepare_data(train_df, test_df):
    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    groups = train_df.iloc[:, -1].values 
    X_test = test_df.values
    return X, y, groups, X_test

def create_submission_file(test_probabilities, c_val):
    final_preds = np.argmax(test_probabilities, axis=1)
    df = pd.DataFrame({'ID': range(len(final_preds)), 'Predicted': final_preds})
    
    filename = f'submission_lasso_C{c_val}.csv'
    df.to_csv(filename, index=False)
    print(f" -> SAVED: '{filename}'")

def scan_and_generate(X, y, groups, X_test):
    
    group_k_fold = GroupKFold(n_splits=5)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n--- Starting Production Line ---")
    
    for c_val in C_CANDIDATES:
        fold_scores = []
        print(f"\n[Processing C={c_val}]")
        print(f"1. Calculating CV Score...", end=" ", flush=True)
        
        # 1. Aşama: Cross Validation (Skoru görmek için)
        for i, (train_idx, val_idx) in enumerate(group_k_fold.split(X, y, groups=groups)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Hızlı hesaplama için OneVsRest + Liblinear
            base_model = LogisticRegression(
                C=c_val, penalty='l1', solver='liblinear', 
                max_iter=200, random_state=SEED, verbose=0
            )
            clf = OneVsRestClassifier(base_model, n_jobs=1)
            
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            fold_scores.append(f1_score(y_val, val_preds, average='macro'))
            print(".", end="", flush=True)
        
        avg_score = np.mean(fold_scores)
        print(f" Done. (CV Score: {avg_score:.4f})")
        
        # 2. Aşama: Full Eğitim ve Dosya Oluşturma
        print(f"2. Retraining on FULL Data & Saving...", end=" ", flush=True)
        
        # Final dosya için en kaliteli çözücüyü (SAGA) kullanıyoruz
        final_clf = LogisticRegression(
            C=c_val,
            penalty='l1',
            solver='saga', 
            max_iter=2000,
            random_state=SEED,
            n_jobs=SAFE_CORES
        )
        final_clf.fit(X_scaled, y)
        
        test_probs = final_clf.predict_proba(X_test_scaled)
        create_submission_file(test_probs, c_val)

if __name__ == "__main__":
    try:
        train_df, test_df = load_data()
        X, y, groups, X_test = prepare_data(train_df, test_df)
        
        scan_and_generate(X, y, groups, X_test)
        
        print(f"\nALL JOBS COMPLETED. Check your folder!")
        
    except Exception:
        traceback.print_exc()