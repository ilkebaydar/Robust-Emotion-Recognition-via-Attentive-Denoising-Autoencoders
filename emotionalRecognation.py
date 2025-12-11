import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import f1_score
import random
import os


SEED = 1
random.seed(SEED)
np.random.seed(SEED)

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

    #test data only has features
        #check test.csv, if it has id colum, drop it

    X_test = test_df.values

    return X, y, groups, X_test

# 5-Fold Cross Validation (both stratified and group cv implementation)
def train_and_evaluate (X, y, groups, X_test):

    #hold place for test predictions (I took average preds from holds)
    test_preds_accum = np.zeros((X_test.shape[0], 4)) #4 classes are given
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

    print("\n START 5-FOLD GROUP CROSS VALIDATION(Person-independent)")
    
    group_kf = GroupKFold(n_splits=5)
    f1_scores_group = []

    #Split wrt groups
    for fold_index, (train_idx, val_idx) in enumerate(group_kf.split(X, y, groups=groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx] , y[val_idx]

        classifier = lgb.LGBMClassifier(random_state=SEED, verbose= -1)
        classifier.fit(X_train, y_train)

        val_preds = classifier.predict(X_val)
        score = f1_score(y_val, val_preds, average='macro')
        f1_scores_group.append(score)

        print(f"Group Fold {fold_index+1} Macro: {score: .4f}")

        #accumulate preds
        test_preds_accum += classifier.predict_proba(X_test)

    avg_group_score = np.mean(f1_scores_group)
            
    print(f"Aveage Group 5-Fold F1 Score: {avg_group_score: .4f}")
    return test_preds_accum, avg_standard_score, avg_group_score

#generate submission file
def create_submission_file(test_probs):
    final_class_preds= np.argmax(test_probs, axis= 1)

    submission_df = pd.DataFrame({
        'ID': range(len(final_class_preds)),
        'Predicted': final_class_preds
    })

    submission_df.to_csv('submission.csv', index=False)
    print("\n submission.csv created successfully.")

if __name__ == "__main__":
    try: 
        train_df, test_df = load_data()
        X, y, groups, X_test = prepare_data(train_df, test_df)
        test_probs, score_std, score_grp = train_and_evaluate(X, y, groups, X_test)
        create_submission_file(test_probs)

        print("\n SUMMARY:")
        print(f"Standars Cross Validaiton F1: {score_std: .4f}")
        print(f"Group Cross Validation F1: {score_grp: .4f} (for  Kaggle)")

    except FileNotFoundError:
        print("Error: train.csv and/or test.csv not found")

    except Exception:
        print(f"An error occured: {Exception}")
        