import numpy as np
import pandas as pd


def print_badrate(target_col):
    print(target_col.value_counts())
    print("badrate:")
    print(target_col.value_counts() / len(target_col))
    
def print_badrate_train_test(train_df, test_df, TARGET):
    df = pd.DataFrame(index=[0, 1])
    df["train"] = train_df[TARGET].value_counts() / len(train_df)
    df["test"] = test_df[TARGET].value_counts() / len(test_df)
    sum_len = len(train_df) + len(test_df)
    cut_df = pd.DataFrame({"train": len(train_df) / sum_len, "test": len(test_df) / sum_len}, index=["cut"])
    print(df.append(cut_df))
    
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
