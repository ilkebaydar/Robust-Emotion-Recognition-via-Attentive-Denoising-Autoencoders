import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import random
import os
import warnings


BEST_C = 0.12  # found from liblinear solving with lasso

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
SAFE_CORES = max(2, int(os.cpu_count() / 2))


print(f"Using Best C: {BEST_C}")
print(f"Solver: SAGA")

def train_and_submit():
    print("Loading data...")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    X = train_df.iloc[:, :-2].values
    y = train_df.iloc[:, -2].values.astype(int)
    X_test = test_df.values
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training SAGA model...")
   
    
    
    clf = LogisticRegression(
        C=BEST_C,
        penalty='l1',
        solver='saga',             
        # multi_class='multinomial', 
        max_iter=3000,             
        random_state=SEED,
        n_jobs=SAFE_CORES,
        verbose=1
    )
    
    clf.fit(X_scaled, y)
    
    n_selected = np.sum(np.abs(clf.coef_) > 1e-5) / 4
    print(f"\nTraining Complete!")
    print(f"Average features kept per class: {int(n_selected)}")
    
    
    test_probs = clf.predict_proba(X_test_scaled)
    final_preds = np.argmax(test_probs, axis=1)
    
    df = pd.DataFrame({'ID': range(len(final_preds)), 'Predicted': final_preds})
    filename = f'submission_SAGA_C{BEST_C}.csv'
    df.to_csv(filename, index=False)
    print(f"✅ FINAL FILE READY: '{filename}'")

if __name__ == "__main__":
    train_and_submit()