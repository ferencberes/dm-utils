import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import operator

import matplotlib
import matplotlib.pyplot as plt

### Oridnal categorical features & replacement with PD or WOE ###

def get_pd(df, TRG):
    "Calculate PD for a dataframe."
    return len(df[df[TRG] == 1]) / float(len(df))

def get_stats_for_categorical(df,col,trg,verbose=False,keep_these_cols=["all_count","def_count","PD","WOE","IV_member"]):
    """Compute PD, IV, WOE for binned values"""
    freqs = df[col].value_counts()
    freqs_pos = df[df[trg]==1][col].value_counts()
    stats_df = pd.DataFrame({"all_count":freqs,"def_count":freqs_pos}).fillna(0)
    # handle no-default categories
    for idx, row in stats_df.iterrows():
        if row["def_count"] == 0:
            print("%s: no default record in category with index %s: %s" % (col, str(idx), row))
            stats_df.set_value(idx, "all_count", row["all_count"] + 1)
            stats_df.set_value(idx, "def_count", 1)
    stats_df["non_def_count"] = stats_df.apply(lambda x: x["all_count"] - x["def_count"], axis=1)
    # type conversion
    for col in ["all_count","def_count","non_def_count"]:
        stats_df[col] = stats_df[col].astype('float')
    sums = stats_df.sum()
    stats_df["PD"] = stats_df.apply(lambda x: x["def_count"]/x["all_count"], axis=1)
    stats_df["default"] = stats_df.apply(lambda x: x["def_count"]/sums["def_count"], axis=1)
    stats_df["non_default"] = stats_df.apply(lambda x: x["non_def_count"]/sums["non_def_count"], axis=1)    
    stats_df["WOE"] = stats_df.apply(lambda x: np.log(x["non_default"]/x["default"]), axis=1)
    stats_df["IV_member"] = stats_df.apply(lambda x: (x["non_default"]-x["default"]) * x["WOE"], axis=1)
    if verbose:
        print(stats_df)
    stats_df = stats_df.sort_index()
    IV = stats_df["IV_member"].sum()
    return stats_df[keep_these_cols], IV

def get_stat_per_col(df, col, trg, stat="PD", verbose=False):
    "Calculate 'PD' or 'WOE' for a categorical feature in a dataframe."
    if not stat in ["PD","WOE"]:
        raise RuntimeError("Choose 'stat' from 'PD' or 'WOE'!")
    stats, iv = get_stats_for_categorical(df,col,trg)
    if verbose:
        print()
        print("IV for %s: %f" % (col,iv))
    return stats.sort_values(stat,ascending=(True if stat=="PD" else False))[stat], iv

#def get_ordinal_categories(data_df,train_df,columns,replace_value="INDEX",trg="TARGET",verbose=False):
def get_ordinal_categories(data_df,columns,replace_value="WOE",trg="TARGET",verbose=False):
    """Generate categorical features for 'data_df'. PD or WOE values of the categories will be used based on 'replace_value' argument.
    """
    train_df = data_df.copy()
    if not replace_value in ["INDEX","PD","WOE"]:
        raise RuntimeError("Choose 'replace_value' from 'WOE', 'PD' or 'INDEX'!")
    iv_map = {}
    for col in columns:
        stats, iv = get_stat_per_col(train_df, col, trg, stat=("PD" if replace_value=="INDEX" else replace_value),verbose=False)
        iv_map[col] = iv
        if replace_value == "INDEX":
            # note that stats are ordered according to the stat!!!
            cat_to_ord_dict = dict(zip(stats.index, range(len(stats))))
            if verbose:
                print(sorted(cat_to_ord_dict.items(), key=operator.itemgetter(1)))
            data_df[col+"_ORDINAL"] = data_df[col].replace(cat_to_ord_dict)
        else:
            if verbose:
                print(stats)
            train_df[col+"_"+replace_value] = train_df[col].replace(stats)
            data_df[col+"_"+replace_value] = data_df[col].replace(stats)
    #return iv_map

### Categorization of continuous features ###

def extract_splits_from_gbt(values,gbt,splits={}):
    """Extract all split values from a GBT classifier. Provide splits dictionary if you have pre-collected
    splits already."""
    for i in range(gbt.n_estimators):
        for val in gbt.estimators_[i,0].tree_.threshold:
            if val != -2:
                if not val in splits:
                    splits[val] = 0.0
                splits[val] = splits[val] + 1.0
    return splits

def get_auto_categories(values,splits,num_splits=2,label_with_indeces=True,verbose=False):
    """Categorize data based on auto-filtered cutting values. You can define the number of cutting values by 'num_splits' argument."""
    sorted_splits = sorted(splits.items(), key=operator.itemgetter(1), reverse=True)
    #print(sorted_splits)
    cuts, selected_splits = splits.keys(), {}
    k = min(num_splits, len(splits))
    for split_val, support_val in sorted_splits[:k]:
            selected_splits[split_val] = support_val
    splits = selected_splits
    splits[max(values)+0.01] = 0.0
    splits[min(values)-0.01] = 0.0
    split_values = sorted(splits.keys())
    if verbose:
        print("Interval bound frequencies:")
        for val in split_values:
            print("%f: %f" % (val,splits[val]))
        print()
    if label_with_indeces:
        categorized_values = pd.cut(values, split_values, labels=False)
    else:
        categorized_values = pd.cut(values, split_values, retbins=True)
    if len(splits) < num_splits:
        print("Warning: the number of optimal cutting values is less then your predefined 'num_splits=%i' value!" % num_splits)
    return categorized_values, splits

def calculate_stats_for_bins(binned_values,labels,sorted_split_values,verbose=False):
    """Compute PD, IV, WOE for binned values"""
    binned_df = pd.DataFrame()
    binned_df["value"] = binned_values
    binned_df["label"] = labels    
    stats_df, IV = get_stats_for_categorical(binned_df,"value","label")
    if verbose:
        print(stats_df)
    stats_df = stats_df.sort_index()
    lower_bounds, upper_bounds = sorted_split_values[0:-1], sorted_split_values[1:]
    if len(lower_bounds) != len(stats_df) or len(upper_bounds) != len(stats_df):
        print(len(lower_bounds), len(upper_bounds), len(stats_df))
        raise RuntimeError("TRY AGAIN: bagging found invalid number of cutting values!")
    stats_df["lower_bound"] = lower_bounds
    stats_df["upper_bound"] = upper_bounds
    return stats_df, IV

def plotter(original_values,binned_values,splits,plot_bin_freqs=True,figure_file=None):
    """Visualize original and categorized feature value distributions with category splits"""
    split_values = sorted(splits.keys())
    values_df = pd.DataFrame({"value":original_values,"bin":binned_values})
    medians = values_df.groupby("bin").median()
    values_df["bin_median"] = values_df["bin"].map(medians["value"])
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.hist(values_df["value"],range=[split_values[0],split_values[-2]],bins=30)
    if plot_bin_freqs:
        plt.hist(values_df["bin_median"],range=[split_values[0],split_values[-2]],color="g",alpha=0.4)
    for val in split_values[1:-1]:
        plt.axvline(val, color='r', linestyle='dashed', linewidth=splits[val])
    plt.subplot(1,2,2)
    plt.hist(values_df["value"],bins=40)
    if plot_bin_freqs:
        plt.hist(values_df["bin_median"],bins=40,range=[split_values[0],split_values[-1]],color="g",alpha=0.4)
    for val in split_values[1:-1]:
        plt.axvline(val, color='r', linestyle='dashed', linewidth=splits[val])
    plt.show()
    if figure_file != None:
        plt.savefig(figure_file)

def get_data_sample(df, feat, trg, ratio):
    """Sample records from original data for bagging"""
    df_fraction = df[[feat,trg]]
    non_defaults = df_fraction[df_fraction[trg] == 0].sample(frac=ratio)
    defaults = df_fraction[df_fraction[trg] == 1].sample(frac=ratio)
    df_sample = pd.concat([non_defaults,defaults],axis=0) # TODO: should we use random permutation???
    return np.array(df_sample[feat]).reshape(len(df_sample),1), df_sample[trg]
    
def show_opt_categories(df, feat, gbt, num_splits, sample_ratio = 0.9, bagging_rounds = 10, trg = "TARGET", verbose=False, figure_file=None):
    """Discover important splits for a continuous feature. Splits are extracted through GBT with bagging."""
    splits = {}
    auc_list = []
    for i in range(bagging_rounds):
        tr_arr, labels = get_data_sample(df,feat,trg,sample_ratio)
        gbt.fit(tr_arr,labels)
        pred = gbt.predict_proba(tr_arr)[:,1]
        auc_list.append(roc_auc_score(labels,pred))
        splits = extract_splits_from_gbt(tr_arr,gbt,splits)
    for val in splits:
        splits[val] = float(splits[val]) / bagging_rounds
    binned_values, splits = get_auto_categories(df[feat],splits,num_splits,verbose=verbose)
    stats, iv = calculate_stats_for_bins(binned_values,df[trg],sorted(splits.keys()),verbose=False)
    print(stats[["lower_bound","upper_bound","all_count","def_count","PD","WOE"]])
    #print("AUC: %f, IV:%f" % (np.mean(auc_list),iv))
    plotter(df[feat],binned_values,splits,figure_file=figure_file)
    
def get_categories(df,feat,split_values,value_type="index",trg="TARGET",figure_file=None, verbose=False):
    """Get categories based on split_values. 'split_values' must be sorted! 
    Choose with 'value_type' what to replace into category indices (e.g.: 'pd' or 'woe')."""
    all_bounds = [df[feat].min()-0.01] + split_values + [df[feat].max()+0.01]
    binned_values = pd.cut(df[feat], all_bounds, labels=False)
    stats, iv = calculate_stats_for_bins(binned_values,df[trg],all_bounds,verbose=verbose)
    if value_type == "index":
        # plots and statistics only visualized if index is used as value
        print(stats[["lower_bound","upper_bound","all_count","def_count","PD","WOE"]])
        #print("IV: %f" % iv)
        plotter(df[feat],binned_values,dict(zip(all_bounds,np.ones(len(all_bounds)))),figure_file=figure_file)
    elif value_type == "pd":
        binned_values = binned_values.replace(stats["PD"])
    elif value_type == "woe":
        binned_values = binned_values.replace(stats["WOE"])
    else:
        raise RuntimeError("Invalid 'value_type'! Choose from 'pd', 'woe' or 'index'.")
    return binned_values