import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

# --- FINAL PUSH CONFIG ---
# Increasing C to allow more features (~850)
LASSO_C = 0.012 
RANDOM_STATE = 42
warnings.filterwarnings("ignore")

def run_hybrid_ensemble():
    print("--- 🎯 FINAL PUSH FOR 0.42+ ---")

    # 1. Load Data
    train_path, test_path = None, None
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            if "train.csv" in filename: train_path = os.path.join(dirname, filename)
            elif "test.csv" in filename: test_path = os.path.join(dirname, filename)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    X_test = test_df.values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)

    # 2. Expanded Feature Selection
    print(f"[1] Selecting more features with C={LASSO_C}...")
    lasso = OneVsRestClassifier(LogisticRegression(C=LASSO_C, penalty='l1', solver='liblinear', random_state=RANDOM_STATE))
    lasso.fit(X_scaled, y)
    mask = np.any(np.abs(np.vstack([est.coef_ for est in lasso.estimators_])) > 1e-5, axis=0)
    
    X_subset = X_scaled[:, mask]
    X_test_subset = X_test_scaled[:, mask]
    print(f"   ✅ Features used: {np.sum(mask)}")

    # 3. Weighted Ensemble
    print("[2] Training Weighted Hybrid SVMs...")
    
    # Model 1: Stable
    clf1 = SVC(C=10, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
    # Model 2: The Champion (Weighted higher)
    clf2 = SVC(C=50, kernel='rbf', gamma='scale', probability=True, random_state=RANDOM_STATE)
    # Model 3: High-Gamma (Captures tighter patterns)
    clf3 = SVC(C=25, kernel='rbf', gamma='auto', probability=True, random_state=RANDOM_STATE)
    
    ensemble = VotingClassifier(
        estimators=[('stable', clf1), ('champ', clf2), ('tight', clf3)],
        voting='soft',
        weights=[1, 2, 1], # Champion model gets double the vote
        n_jobs=-1
    )
    
    ensemble.fit(X_subset, y)
    
    # 4. Save
    print("[3] Exporting result...")
    preds = ensemble.predict(X_test_subset)
    sub_file = 'submission_Hybrid.csv'
    pd.DataFrame({'ID': range(len(preds)), 'Predicted': preds}).to_csv(sub_file, index=False)
    
    print("✅ DONE! Final Hybrid approach complete.")

if __name__ == "__main__":
    run_hybrid_ensemble()