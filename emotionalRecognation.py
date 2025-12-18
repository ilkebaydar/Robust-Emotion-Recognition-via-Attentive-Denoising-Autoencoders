import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

# --- FINAL SURGICAL OPERATION ---
# C=0.015 gave 944 features (Too many).
# We are dropping it to 0.003 to force it down to ~200.
STRICT_LASSO_C = 0.003

warnings.filterwarnings("ignore")

print(f"--- OPERATION: SVM HYBRID (SURGICAL MODE) ---")
print(f"Step 1: Using Surgical Lasso (C={STRICT_LASSO_C})...")

def run_strict_hybrid():
    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    X_test = test_df.values
    
    # 2. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. LASSO FEATURE SELECTION
    print("Selecting features with Lasso...")
    
    base_lasso = LogisticRegression(
        C=STRICT_LASSO_C, 
        penalty='l1', 
        solver='liblinear', 
        random_state=42
    )
    
    lasso = OneVsRestClassifier(base_lasso)
    lasso.fit(X_scaled, y)
    
    # Extract coefficients
    all_coefs = []
    for estimator in lasso.estimators_:
        all_coefs.append(estimator.coef_)
    
    coef_matrix = np.vstack(all_coefs)
    mask = np.any(np.abs(coef_matrix) > 1e-5, axis=0)
    n_features = np.sum(mask)
    
    print(f"✅ Lasso Selection Complete!")
    print(f"   Original Features: {X.shape[1]}")
    print(f"   Selected Features: {n_features} (Hoping for ~200)")
    
    if n_features == 0:
        print("⚠️ WARNING: Too strict! No features selected.")
        return

    # Filter dataset
    X_subset = X_scaled[:, mask]
    X_test_subset = X_test_scaled[:, mask]
    
    # 4. SVM ENGAGED
    print("\nStep 2: Training SVM on selected features...")
    
    # Optimized Param Grid (Quick but effective)
    param_grid = {
        'C': [1, 10, 50], 
        'gamma': ['scale', 0.01],
        'kernel': ['rbf'] 
    }
    
    svm = SVC(random_state=42)
    grid = GridSearchCV(svm, param_grid, cv=5, scoring='f1_macro', n_jobs=-1, verbose=1)
    
    grid.fit(X_subset, y)
    
    print(f"\n🏆 Best SVM Parameters: {grid.best_params_}")
    print(f"🚀 Best Local CV Score: {grid.best_score_:.5f}")
    
    # 5. Generate Submission File
    final_model = grid.best_estimator_
    final_preds = final_model.predict(X_test_subset)
    
    filename = f'submission_SVM_Surgical_C{STRICT_LASSO_C}.csv'
    df = pd.DataFrame({'ID': range(len(final_preds)), 'Predicted': final_preds})
    df.to_csv(filename, index=False)
    print(f"✅ FINAL FILE READY: '{filename}'")

if __name__ == "__main__":
    run_strict_hybrid()