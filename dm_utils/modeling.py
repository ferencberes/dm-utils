import numpy as np
import pandas as pd
import sklearn.metrics as ms
from model_preparation import *
from model_wrappers.scikit_model import get_lreg_coefficients


def get_auc(train_df, test_df, clf, feature_columns, target):
    train_auc = ms.roc_auc_score(train_df[target], [l[1] for l in clf.predict_proba(train_df[feature_columns])])
    test_auc = ms.roc_auc_score(test_df[target], [l[1] for l in clf.predict_proba(test_df[feature_columns])])
    return train_auc, test_auc


def print_auc_(train_auc, test_auc):
    print("train auc:", train_auc)
    print("test  auc:", test_auc)


def print_auc(train_df, test_df, clf, feature_columns, target):
    train_auc, test_auc = get_auc(train_df, test_df, clf, feature_columns, target)
    print_auc_(train_auc, test_auc)


def smape(y_true, y_pred):
    return 2 * np.mean( np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)) )


def rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def run_model(train_df, test_df, feature_columns, target, clf, verbose=True):
    clf.fit(train_df[feature_columns], train_df[target])
    train_auc, test_auc = get_auc(train_df, test_df, clf, feature_columns, target)
    if verbose:
        print_auc_(train_auc, test_auc)
    return train_auc, test_auc


def run_model_n_times(features_df, feature_columns, target, cut, clf, n_times, verbose=False):
    train_aucs = []
    test_aucs = []
    for i in range(n_times):
        if verbose:
            print("n=%i:" % (i+1))
        train_df, test_df = get_random_train_test(features_df, cut, target, seed=None)
        train_auc, test_auc = run_model(train_df, test_df, feature_columns, target, clf, verbose=verbose)
        if verbose:
            print_auc_(train_auc, test_auc)
        train_aucs.append(train_auc)
        test_aucs.append(test_auc)
    if verbose:
        print("--------------------------------")
    train_aucs = pd.Series(train_aucs)
    test_aucs = pd.Series(test_aucs)
    train_auc = train_aucs.mean()
    test_auc = test_aucs.mean()
    print_auc_(train_auc, test_auc)
    print("deviation:")
    print("train:", train_aucs.std())
    print("test: ", test_aucs.std())


def get_aggregated_coef_df(features_df, feature_columns, target, cut, clf, n_times):
    feature_coefficients = dict([(col, []) for col in feature_columns])
    for i in range(100):
        train_df, test_df = get_random_train_test(features_df, cut, target, seed=None)
        clf.fit(train_df[feature_columns], train_df[target])
        coef_df = get_lreg_coefficients(clf, feature_columns)
        for feat in feature_columns:
            feature_coefficients[feat].append( coef_df.set_index("name").ix[feat]["value"] )
    feature_coefficients_ser = pd.Series(feature_coefficients)
    aggregated_coef_df = pd.DataFrame(index=feature_coefficients.keys())
    for col, func in [("mean", np.mean), ("std", np.std), ("min", min), ("max", max)]:
        aggregated_coef_df[col] = feature_coefficients_ser.apply(func)
    return aggregated_coef_df.sort_values("mean")
