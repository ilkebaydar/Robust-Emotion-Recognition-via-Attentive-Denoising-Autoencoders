import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import random
import os
import warnings

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#COARSE-TO-FINE SEARCH
#C_CANDIDATES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0] #COARSE-> after 0.1 score trend: downward
#C_CANDIDATES = [0.07, 0.08] #FINE: score of 0.07 is less than 0.1, no need to try others less than 0.1
C_CANDIDATES = [0.13, 0.15, 0.17]  #FINE: to find better solution than 0.1

print(f"--- FAST SCANNER & GENERATOR  ---")

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

def create_submission_file(model, X_test_scaled, c_val):
    # prediction
    test_probs = model.predict_proba(X_test_scaled)
    final_preds = np.argmax(test_probs, axis=1)
    
    #submission file creation
    df = pd.DataFrame({'ID': range(len(final_preds)), 'Predicted': final_preds})
    filename = f'submission_fast_C{c_val}.csv'
    df.to_csv(filename, index=False)
    return filename

def scan_and_generate(X, y, groups, X_test):
    
    # data scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test)
    
    group_k_fold = GroupKFold(n_splits=5)
    
    print(f"\n{'C VALUE':<10} | {'CV SCORE':<10} | {'STATUS'}")
    print("-" * 50)
    
    best_score = -1
    best_c = None
    
    for c_val in C_CANDIDATES:
        # calculate CV score
        fold_scores = []
        
        # Model: Liblinear + OneVsRest -> for fast evaluation
        base_model = LogisticRegression(
            C=c_val, penalty='l1', solver='liblinear', 
            max_iter=300, random_state=SEED, verbose=0
        )
        clf = OneVsRestClassifier(base_model, n_jobs=1)
        
        for train_idx, val_idx in group_k_fold.split(X, y, groups=groups):
            clf.fit(X_scaled[train_idx], y[train_idx])
            val_preds = clf.predict(X_scaled[val_idx])
            fold_scores.append(f1_score(y[val_idx], val_preds, average='macro'))
        
        avg_score = np.mean(fold_scores)
        
        # training and write
        clf.fit(X_scaled, y)
        filename = create_submission_file(clf, X_test_scaled, c_val)
        
        # for keeping track score trends
        status = f"Saved: {filename}"
        if avg_score > best_score:
            best_score = avg_score
            best_c = c_val
            status += " (NEW BEST!)"
            
        print(f"{c_val:<10} | {avg_score:.5f}   | {status}")

    print("-" * 50)
    print(f"🏆 BEST LOCAL SCORE: {best_score:.5f} with C={best_c}")

if __name__ == "__main__":
    train_df, test_df = load_data()
    X, y, groups, X_test = prepare_data(train_df, test_df)
    
    scan_and_generate(X, y, groups, X_test)