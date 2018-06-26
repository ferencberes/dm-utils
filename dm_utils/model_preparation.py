import numpy as np
import pandas as pd


def print_badrate(target_col):
    print(target_col.value_counts())
    print("badrate:")
    print(target_col.value_counts() / len(target_col))

def print_badrate_train_test(train_df, test_df, TARGET):
    target_values = sorted(set(list(train_df[TARGET].unique()) + list(test_df[TARGET].unique())))
    df = pd.DataFrame(index=target_values)
    df["train"] = train_df[TARGET].value_counts() / len(train_df)
    df["test"] = test_df[TARGET].value_counts() / len(test_df)
    sum_len = float(len(train_df) + len(test_df))
    cut_df = pd.DataFrame({"train": len(train_df) / sum_len, "test": len(test_df) / sum_len}, index=["cut"])
    print(df.append(cut_df))

def get_random_train_test(features_df, cut, target, seed=None):
    np.random.seed(seed)
    train_index = []
    for target_value in features_df[target].unique():
        temp_df = features_df[features_df[target] == target_value]
        train_index += list(temp_df.iloc[np.random.permutation(len(temp_df))][:int(round(cut*len(temp_df)))].index)
    train_df = features_df.ix[train_index]
    test_df = features_df.ix[~features_df.index.isin(train_index)]
    return train_df, test_df

def get_train_test_cut_by_col(features_df, column, cut):
    train_index = features_df[column] < cut 
    train_df = features_df[train_index]
    test_df = features_df[~train_index]
    return train_df, test_df

def onehot_column(dataframe, column):
    new_columns = []
    for value in dataframe[column].unique():
        new_col = column + "_-_ONEHOT_-_" + str(value)
        new_columns.append(new_col)
        dataframe[new_col] = dataframe[column].apply(lambda col: int(col == value))
    return new_columns

def onehot_columns(dataframe, columns):
    new_columns = []
    for column in columns:
        new_columns += onehot_column(dataframe, column)
    return new_columns
