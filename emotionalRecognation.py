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

   #select most important 350 features
def select_important_features(X, y, X_test, top_n = 350):
    print(f"START FEATURE SELECTION (Target: Top {top_n} features)")

    temp_model= lgb.LGBMClassifier(random_state=SEED, verbose= -1, n_estimators=100, n_jobs=SAFE_CORES)
    temp_model.fit(X, y)

    importances = temp_model.feature_importances_

    #find indices descending sorted
    indices = np.argsort(importances)[::-1]

    #select only top_n features
    top_indices = indices[:top_n]
    print(f"Selected top {top_n} features out of 1793")

    X_selected = X[: , top_indices]
    X_test_selected = X_test[:, top_indices]

    return X_selected, X_test_selected


# 5-Fold Cross Validation (both stratified and group cv implementation)
def train_and_evaluate (X, y, groups, X_test):
    test_preds_accum =np.zeros((X_test.shape[0], 4)) #4 classes are given

    """
    #hold place for test predictions (I took average preds from holds)
    print("\n START STANDART 5-FOLD CROSS VALIDATION")
    stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    f1_scores_standard = []

    for fold_index, (train_idx, val_idx) in enumerate(stratified_kf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        #Light Gradient Boosting model
        classifier = lgb.LGBMClassifier(random_state=SEED, verbose=-1)
        classifier.fit(X_train, y_train)

        val_preds= classifier.predict(X_val)
        score = f1_score(y_val, val_preds, average='macro') #from important note 3 in project pdf
        f1_scores_standard.append(score)

        print(f"Standard Fold {fold_index+1} F1-Macro: {score: .4f}")

    avg_standard_score = np.mean(f1_scores_standard)
    print(f"Average Standard 5-Fold F1 Score: {avg_standard_score: .4f}")
    """

    print("\n START 5-FOLD GROUP CROSS VALIDATION(Person-independent)")
    
    group_kf = GroupKFold(n_splits=5)
    f1_scores_group = []

    #Split wrt groups
    for fold_index, (train_idx, val_idx) in enumerate(group_kf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx] , y[val_idx]

    
        classifier = lgb.LGBMClassifier(
                    random_state=SEED,
                    verbose= -1,
                    n_estimators=500, #increased for better learning
                    learning_rate=0.05,#for robustness
                    num_leaves=31, #for controlling complexity
                    n_jobs= SAFE_CORES #uses 50% CPU power 
                    )
        classifier.fit(X_train, y_train)

        val_preds = classifier.predict(X_val)
        score = f1_score(y_val, val_preds, average='macro')
        f1_scores_group.append(score)

        print(f"Group Fold {fold_index+1} Macro: {score: .4f}")

        #accumulate preds
        test_preds_accum += classifier.predict_proba(X_test)

    avg_group_score = np.mean(f1_scores_group)
            
    print(f"Aveage Group 5-Fold F1 Score: {avg_group_score: .4f}")
    return test_preds_accum, None, avg_group_score #instead of None avg_standard_score will add when calculating standard one


#generate submission file
def create_submission_file(test_probs):
    final_class_preds= np.argmax(test_probs, axis= 1)

    submission_df = pd.DataFrame({
        'ID': range(len(final_class_preds)),
        'Predicted': final_class_preds
    })

    submission_df.to_csv('submission_with_fs.csv', index=False)
    print("\n submission.csv created successfully.")

if __name__ == "__main__":
    try: 
        train_df, test_df = load_data()
        X, y, groups, X_test = prepare_data(train_df, test_df)

        #select best 350 feature
        print("Step 1: Feature Selection")
        X_reduced, X_test_reduced = select_important_features(X, y, X_test, top_n=350)
        print(f"Feature Selection is Done. New shape: {X_reduced.shape}")

        print("Step 2: Model Training with Group K-Fold")
        test_probs, score_std, score_grp = train_and_evaluate(X_reduced, y, groups, X_test_reduced)
        create_submission_file(test_probs)

        print("\n SUMMARY:")
       # print(f"Standard Cross Validaiton F1: {score_std: .4f}")
        print(f"Group Cross Validation F1: {score_grp: .4f} (for  Kaggle)")

    except Exception:
        print("\n CRITICAL ERROR DETAILS!")
        traceback.print_exc()
