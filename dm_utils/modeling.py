import numpy as np
import pandas as pd
import sklearn.metrics as sm


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

def get_aggregated_coef_df(features_df, feature_columns, cut, clf, get_lreg_coefficients, n_times):
    feature_coefficients = dict([(col, []) for col in feature_columns])
    for i in range(100):
        train_df, test_df = get_train_test(features_df, cut=cut)
        clf.fit(train_df[feature_columns], train_df["TARGET"])
        coef_df = get_lreg_coefficients(clf, feature_columns)
        for feat in feature_columns:
            feature_coefficients[feat].append( coef_df.set_index("name").ix[feat]["value"] )
    feature_coefficients_ser = pd.Series(feature_coefficients)
    aggregated_coef_df = pd.DataFrame(index=feature_coefficients.keys())
    for col, func in [("mean", np.mean), ("std", np.std), ("min", min), ("max", max)]:
        aggregated_coef_df[col] = feature_coefficients_ser.apply(func)
    return aggregated_coef_df.sort_values("mean")
    