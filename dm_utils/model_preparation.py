import numpy as np
import pandas as pd


def get_random_train_test(features_df, cut, target, seed=None):
    pos_target = features_df[features_df[target] == 1]
    neg_target = features_df[features_df[target] == 0]
    np.random.seed(seed)
    pos_train_index = pos_target.iloc[np.random.permutation(len(pos_target))][:round(cut*len(pos_target))].index
    neg_train_index = neg_target.iloc[np.random.permutation(len(neg_target))][:round(cut*len(neg_target))].index
    train_index = pd.Index(np.random.permutation(list(pos_train_index) + list(neg_train_index)))
    train_df = features_df.ix[train_index]
    test_df = features_df.ix[~features_df.index.isin(train_index)]
    return train_df, test_df

def get_train_test_cut_by_col(features_df, column, cut):
    train_index = features_df[column] < cut 
    train_df = features_df[train_index]
    test_df = features_df[~test_index]
    return train_df, test_df
