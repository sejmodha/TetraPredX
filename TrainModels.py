"""
Created on Wed 24 Mar 15:37:43 GMT 2021.

@author: sejmodha
"""

import joblib
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


np.random.seed(235487)

"""Functions for prokaryotic sequence prediction models."""


def set_vars():
    """Set var_list required for the module."""
    parser = argparse.ArgumentParser(description='This script can be used to train new random forest models')
    parser.add_argument('-i', '--infile', help='Input a comma separated data file (.csv)', required=True)
    parser.add_argument('-n', '--indexColumn', help='Specify index column name (default = "id")', required=False, default='id')
    parser.add_argument('-y', '--targetColumn', help='Specify target/label column name (default = "class")', required=False, default='class')
    parser.add_argument('-t', '--threads', help='Specify number of threads (default = 2)', required=False, default=2, type=int)
    parser.add_argument('-o', '--outputPrefix', help='Specify an output prefix', required=False)
    parser.add_argument('-path', '--modelpath', help='Provide locations to model files (default = models/)', required=False, default='models')
    parser.add_argument('-cv', '--crossvalidation', help='Cross-validation generator (default = 5)', required=False, default=5, type=int)

    args = parser.parse_args()

    datafile = args.infile
    index_column = args.indexColumn
    target_column = args.targetColumn
    cpu = args.threads
    path = args.modelpath
    out = args.outputPrefix
    cv = args.crossvalidation

    df = pd.read_csv(datafile, index_col=index_column)

    return (df, target_column, cpu, out, path, cv)


def get_train_test(input_df, label_col, test_size, k, n_features):
    """Generate train/test set for each class."""
    train_test_dict = {}
    for i in sorted(input_df[label_col].unique()):
        # print(i)
        try:
            pos_df = input_df[input_df[label_col] == i].drop(columns=[label_col])
            pos_df[i] = 1
            # print(pos_df.shape)
            # print(pos_df.head())
            neg_df = input_df[input_df[label_col] != i].sample(n=pos_df.shape[0]).drop(columns=[label_col])
            neg_df[i] = 0
            # print(neg_df.shape)
            # print(neg_df.head())
            data_df = pd.concat([pos_df, neg_df]).sample(frac=1)
            # print(data_df.shape)
            # get X and Y
            X = data_df.iloc[:, 0:n_features]
            y = data_df.iloc[:, n_features]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
            train_test_dict[i] = [X_train, X_test, y_train, y_test, X, y]
            # print(X_train.shape)
            # print(X_test.shape)
            # print(y_train.shape)
            # print(y_test.shape)
        except(ValueError):
            print('Please make sure that data contains positive and negative datapoints for each class.\n')

    return train_test_dict


def get_best_model(X, y, cpu):
    """Run GriSearchCV to identity the best model parameters."""
    for i in y.columns:
        rfc = RandomForestClassifier(random_state=123, n_estimators=100, class_weight='balanced', bootstrap=False)
        steps = [('scaler', StandardScaler()), ('rf', rfc)]
        pipe = Pipeline(steps)
        params = [{'rf__bootstrap': [True, False],
                   'rf__n_estimators': [50, 100, 150, 200, 300, 400, 500],
                   'rf__criterion': ['gini', 'entropy']}]
        gs = GridSearchCV(pipe, params, scoring='f1', n_jobs=cpu)
        gs_clf = gs.fit(X, y[i])
        best_params = gs_clf.best_params_
        best_pipe = gs_clf.best_estimator_
        rf_clf = best_pipe.named_steps.rf
        model = best_pipe

        print(model, rf_clf)


def train_models_rfc(data_dict, out, path, cpu, cv):
    """Run the Random forest classifier and saves models."""
    for k, v in data_dict.items():
        print(f'Building ML models for {k}')
        X_train, X_test, y_train, y_test, X, y = v

        rfc = RandomForestClassifier(n_estimators=1000, class_weight='balanced', bootstrap=True, n_jobs=cpu)

        rfc.fit(X_train, y_train)
        y_pred_rfc = rfc.predict(X_test)
        y_pred_proba_rfc = rfc.predict_proba(X_test)
        print('====================================================')
        print('Raw random forest classifier results below:')
        print('====================================================')
        print(f'Model accuracy is: {metrics.accuracy_score(y_test, y_pred_rfc)}')
        print(metrics.classification_report(y_test, y_pred_rfc))
        titles_options = [("Confusion matrix, without normalization: "+k, None),
                          ("Normalized confusion matrix: "+k, 'true')]
        for title, normalise in titles_options:
            print(title)
            print(confusion_matrix(y_test, y_pred_rfc, normalize=normalise))

        # Try calibrating the classifiers

        clf_sigmoid = CalibratedClassifierCV(rfc, method='sigmoid', n_jobs=cpu, cv=cv)
        clf_sigmoid.fit(X_train, y_train)

        y_pred_proba_sigmoid = clf_sigmoid.predict_proba(X_test)

        print("Brier scores: (smaller is better)")

        clf_score = brier_score_loss(y_test, y_pred_proba_rfc[:, 1])
        print("No calibration: %1.3f" % clf_score)

        clf_sigmoid_score = brier_score_loss(y_test, y_pred_proba_sigmoid[:, 1])
        print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)
        print('====================================================')
        print('Sigmoid calibrated pipeline results below:')
        print('====================================================')
        y_pred_sigmoid = clf_sigmoid.predict(X_test)

        print(f'Model accuracy is: {metrics.accuracy_score(y_test, y_pred_sigmoid)}')
        print(metrics.classification_report(y_test, y_pred_sigmoid))
        for title, normalise in titles_options:
            print(title)
            print(confusion_matrix(y_test, y_pred_sigmoid, normalize=normalise))
        print('====================================================')
        print(f'{cv}-fold Cross validations performed below')
        print('====================================================')
        kfold = KFold(n_splits=cv)
        results_kfold = cross_val_score(rfc, X, y, cv=kfold)
        print(results_kfold)
        print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

        # save the model to disk
        if out is None:
            filename = f'{path}/rfc_{k}.model'
        else:
            filename = f'{path}/{out}_{k}.model'
        print(f'\nSaving calibrated model. \nModel is saved in: {filename}\n')
        joblib.dump(clf_sigmoid, filename, compress=3)
        # save non-calibrated model below
        # joblib.dump(rfc, filename, compress=3)
        plt.close('all')
    return


def main():
    """Run the module as a script."""
    df, target_column, cpu, out, path, cv = set_vars()
    data_dict = get_train_test(df, target_column, 0.3, 4, 256)

    train_models_rfc(data_dict, out, path, cpu, cv)


if __name__ == "__main__":
    main()
