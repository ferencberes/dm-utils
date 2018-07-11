import sys#, importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#importlib.reload(sys)
#sys.setdefaultencoding("utf-8")
sys.path.insert(0,"../")
import evaluation_utils.eval_utils as eu

### toplist highlight ###

def get_color(bool_tuple, colors=["green","red","yellow",""]):
    """Get color for toplist item."""
    is_daily, is_next_or_prev, is_player = bool_tuple
    if is_daily:
        color = colors[0]
    elif is_next_or_prev:
        color = colors[1]   
    elif is_player:
        color = colors[2]
    else:
        color = colors[3]
    return color

def highlight_toplist_items(s, daily_players, next_or_prev_players, tennis_account_ids):
    """Highlight toplist items with proper colors."""
    is_daily = s.isin(daily_players)
    is_next_or_prev = s.isin(next_or_prev_players)
    is_player = s.isin(tennis_account_ids)
    bool_vector = list(zip(is_daily, is_next_or_prev, is_player))
    return ['background-color: %s' % get_color(b) for b in bool_vector]

def show_toplist_with_accounts(cm_postfix, snapshot_id, joined_id_df, path_variables, tennis_account_variables, top_k=50, is_binary=True, is_restricted=False, fname=None):
    """Show toplist for a given centrality measure. Relevant player types can be highlighted as well."""
    score_prefix, player_output_dir = path_variables
    tw_tennis_accounts, tennis_account_ids = tennis_account_variables
    players_prefix = "%s/players" % player_output_dir
    day_idx = snapshot_id // 24
    player_labels = eu.load_score_map(players_prefix,day_idx,epsilon=0.0).reset_index()
    # looking for infinite values
    player_labels = player_labels.replace([np.inf, -np.inf], np.nan)
    num_nan_scores = player_labels.isnull()["score"].sum()
    if num_nan_scores > 0:
        raise RuntimeError("%s Inf score occured in this file!!!")
    #print(player_labels.head())
    if is_binary:
        daily_players = list(player_labels[player_labels["score"]>0.0]["id"])
        prev_players = []
    else:
        daily_players = list(player_labels[player_labels["score"]==2.0]["id"])
        next_or_prev_players = list(player_labels[player_labels["score"]==1.0]["id"])
    print("Number of found daily players: %i" % len(daily_players))
    if is_restricted:
        scores = eu.load_score_map(score_prefix + cm_postfix, snapshot_id, restricted_indices=tennis_account_ids).reset_index()
    else:
        scores = eu.load_score_map(score_prefix + cm_postfix, snapshot_id).reset_index()
    print("Length of full toplist: %i" % len(scores))
    scores_with_account = scores.merge(joined_id_df, left_on="id", right_on="generated_id", how="inner")
    sorted_scores_with_account = scores_with_account.sort_values("score", ascending=False).reset_index(drop=True)
    sorted_scores_with_account["is_player"] = sorted_scores_with_account["screen_name"].isin(tw_tennis_accounts)
    out_df = sorted_scores_with_account[["generated_id","id_y","screen_name","is_player","name","score"]].head(top_k)
    out_df = out_df.merge(player_labels,how="left",left_on="generated_id",right_on="id").fillna(0.0)
    out_df = out_df.rename(columns={"score_x":"score","score_y":"relevance","screen_name":"account_name(@)"})
    out_df = out_df[["generated_id","name","account_name(@)","is_player","relevance","score"]]
    if fname != None:
        write_toplist_to_latex(fname, out_df)
    else:
        return out_df.style.apply(lambda x : highlight_toplist_items(x, daily_players, next_or_prev_players, tennis_account_ids), subset=["generated_id"])
    
def write_toplist_to_latex(fname, toplist_df):
    """Write toplist dataframe to file in latex format"""
    latex_out = "\\begin{tabular}{||c l l c||}\n"
    latex_out += "\hline\n"
    for idx, row in toplist_df.iterrows():
        bool_rec = (row["relevance"] == 2.0, row["relevance"] == 1.0, row["is_player"])
        color = get_color(bool_rec, colors=["yellow","orange","gray","white"])
        latex_out += "\\rowcolor{%s} %i & %s & @%s & %i \\\ \n" % (color, idx+1, row["name"].replace("&","\&"), row["account_name(@)"], row["relevance"])
    latex_out += "\hline\n"
    latex_out += "\end{tabular}\n"
    latex_out = latex_out.replace("_","\_").replace("\t","")
    with open(fname,'w') as f:
        f.write(latex_out)
    print("Toplist written to file: %s" % fname)
        
### correlations and visualization ###

def self_corr_olr(input_pref, metric_id, intervals, betas, windows, norm=24, offset=0):
    spearmans, legends, dframes = [], [], []
    if norm == 1:
        snapshot_indices, day_indices = intervals, intervals
    else:
        snapshot_indices = [((j+offset) % norm) - norm for j in intervals]
        day_indices = [(j+offset) // norm for j in intervals]
    for b in betas:
        for w in windows:
            legends.append("oc beta=%0.2f window=%0.2f" % (b,w))
            score_id = "olr_a0.05_b%0.2f_Exp(b:0.500,n:%0.3f)" % (b,w*3600)
            print(score_id)
            score_paths = "%s/%s/olr" % (input_pref, score_id)
            # correlation
            corrs = eu.calculate_measure_for_days(days=intervals,input_prefix=score_paths,measure_type=metric_id,is_sequential=True,n_threads=8)
            spearmans.append(corrs)
            # create dataframe
            df = pd.DataFrame()
            df["day"] = day_indices
            df["snapshot"] = snapshot_indices
            df[metric_id] = corrs
            df["time_window"] = w
            df["olr_beta"] = b
            df["score"] = score_id
            dframes.append(df[["score","day","snapshot",metric_id,"time_window","olr_beta"]])
    return spearmans, legends, pd.concat(dframes,axis=0)

def get_mean_correlations(corrs, params):
    mean = []
    for i, p in enumerate(params):
        mean.append((p, np.mean(corrs[i])))
    return mean

def draw_time_series(corrs, legends, intervals, is_xticks_hour=True):
    hour_of_day = [j % 24 for j in intervals]
    x = range(len(intervals))
    plt.figure(figsize=(20,8))
    for i, item in enumerate(corrs):
        plt.plot(x,item,label=legends[i])
    if is_xticks_hour:
        plt.xticks(x, hour_of_day,rotation="vertical")
    else:
        plt.xticks(x, intervals,rotation="vertical")
    plt.legend()
    
def draw_hist(corrs, legends, intervals):
    #plt.figure(figsize=(20,8))
    plt.figure(figsize=(10,10))
    x = range(len(intervals))
    for i, item in enumerate(corrs):
        plt.hist(item,label=legends[i],bins=20,alpha=0.5,cumulative=True)
    plt.legend()

def draw_daily_mean(dfs, metric_id, param_name):
    sns.tsplot(data=dfs, time="snapshot",unit="day",value=metric_id,condition=param_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
