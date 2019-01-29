import pandas as pd
import numpy as np
import sys, os

import turicreate as tc
import turicreate.aggregate as agg

### General SFrame functions ###

def dtypes(sf):
    return list(zip(sf.column_names(), sf.dtype))

def replace(sf, replace_dict, cols=None):
    '''Replace values in the selected columns of the turicreate.SFrame'''
    sf_tmp = sf.copy()
    if cols == None:
        cols = sf_tmp.column_names()
    for c in cols:
        sf_tmp[c] = sf_tmp[c].apply(lambda x: replace_dict.get(x,x))
    return sf_tmp

def fillna(sf, value, cols=None):
    '''Replace missing values in the selected columns of the turicreate.SFrame'''
    if cols == None:
        cols = sf.column_names()
    for c in cols:
        sf = sf.fillna(c, value)
    return sf

def concatenate(s1, s2):
    '''Concatenate two turicreate.SFrame object with even different shapes'''
    diff_1 = list(set(dtypes(s1)) - set(dtypes(s2)))
    #print(diff_1)
    for col, t in diff_1:
        s2[col] = None
        s2[col] = s2[col].astype(t)
    diff_2 = list(set(dtypes(s2)) - set(dtypes(s1)))
    #print(diff_2)
    for col, t in diff_2:
        s1[col] = None
        s1[col] = s1[col].astype(t)
    union = list(set(s2.column_names()).union(set(s1.column_names())))
    #print(union)
    return s1[union].append(s2[union])
        
def get_dummies(sf, columns, keep=False, selected_columns=None):
    '''Create binary columns from categorical feature (OneHotEncoding)'''
    sf_tmp = sf.copy()
    for col in columns:
        unique_values = list(sf[col].unique())
        for col_val in unique_values:
            new_col = col + "_ONEHOT_" + str(col_val)
            if selected_columns == None or new_col  in selected_columns:
                sf_tmp[new_col] = sf_tmp.apply(lambda x: 1 if x[col] == col_val else 0)
        if not keep:
            sf_tmp = sf_tmp.remove_column(col)
        print(col)
    return sf_tmp

def num_missing(sf, cols=None):
    '''Extract the number of missing values from the selected of the turicreate.SFrame'''
    if cols == None:
        cols = sf.column_names()
    missing_info = []
    for c in cols:
        if None in sf[c]:
            cnt_info = sf[c].value_counts()
            num_missing = cnt_info[cnt_info["value"] == None]["count"][0]
            missing_info.append((c, num_missing))
            print(c)
    missing_df = pd.DataFrame(missing_info, columns=["name","num_missing"]).sort_values("num_missing", ascending=False)
    missing_df["frac_missing"] = missing_df["num_missing"] / len(sf)
    return missing_df

def batch_join(left, right, keys, how="left", batch_size=10000000):
    '''Join two turicreate.SFrame if the default join operation cannot fit into memory'''
    if len(left) <= batch_size:
        print("default join is applied!")
        return left.join(right, on=keys, how=how)
    else:
        print("batch join is applied!")
        index_splits = list(range(0, len(left)+batch_size, batch_size))
        is_first = True
        for i in range(1, len(index_splits)):
            from_idx, to_idx = index_splits[i-1], index_splits[i]
            partial_left = left[from_idx:to_idx]
            if is_first:
                joined = partial_left.join(right, on=keys, how=how)
                is_first = False
            else:
                joined = joined.append(partial_left.join(right, on=keys, how=how))
            print(i, len(joined))
        return joined

