import os
import re
import pandas as pd
import xgboost as xgb



def get_python_trees(model, SHIFT, deal_with_missing=False, num_trees=None):
    """Converts XGBoost model file to python script."""
    def get_decision_line(line, deal_with_missing):
        condition = line.split("[")[1].split("]")[0]
        feat_and_cut = condition.split("<")
        feat = feat_and_cut[0]
        cut = float(feat_and_cut[1])
        missing_str = ''
        if deal_with_missing:
            yes_branch = int(tmp_line.split("yes=")[1].split(",")[0])
            missing_branch = int(line.split("missing=")[1])
            if yes_branch == missing_branch:
                missing_str = ' or pd.isnull(row["%s"])' % feat
        return '%s:if row["%s"] < %f%s:\n' % (line.split(":[")[0], feat, cut, missing_str)

    def get_indent(times):
        my_indent = "    "
        return my_indent * times

    if isinstance(model, xgb.XGBModel):
        trees = model.booster().get_dump()
    else:
        raise RuntimeError("Wrong model type! Use  XGBModel instead.")
    num_pattern = re.compile("\s*(\d+:)")
    
    max_trees = len(trees) if num_trees is None else num_trees
    model_str = ""
    tree_counter = 0
    prev_indent_size = SHIFT - 1
    curr_indent_size = SHIFT
    for tree_str in trees:
        tree_counter += 1
        model_str += "\n%s#tree_%i\n" % (get_indent(SHIFT), tree_counter)
        is_new_tree = True
        for line in tree_str.rstrip("\n").split("\n"):
            num_finder = re.search(num_pattern, line)
            if "leaf" in line:
                tmp_line = "%s:score += %f\n" % (line.split(":")[0], float(line.split("=")[1]))
            elif "<" in line:
                is_decision = True 
                tmp_line = get_decision_line(line, deal_with_missing)
                
            if num_finder:
                node_idx = num_finder.group(1)
                indent = tmp_line.split(node_idx)[0]
                curr_indent_size = len(indent) + SHIFT
                tmp_line = tmp_line.replace(indent+node_idx, '')
            if curr_indent_size <= prev_indent_size:
                if not is_new_tree:
                    if not (is_decision and curr_indent_size == SHIFT):
                        model_str += "%selse:\n" % get_indent(curr_indent_size-1)
            model_str += get_indent(curr_indent_size) + tmp_line
            prev_indent_size = curr_indent_size
            is_new_tree = False
        if tree_counter >= max_trees:
            break
    return model_str


def get_gbt_model_row(model, deal_with_missing=False, num_trees=None):
    func_str = """
import pandas as pd

def score_by_gbt_tree_rule(row):
    \"""GBT model generated by XGBoost\"""
    score = 0.0           
    """
    func_str += get_python_trees(model, 1, deal_with_missing, num_trees)
    func_str += """
    return score
"""
    return func_str

def get_gbt_model_df(model, deal_with_missing=False, num_trees=None):
    func_str = """
import pandas as pd

def score_gbt(eval_df, model_name):
    \"""Score dataframe by gbt rule\"""

    def score_by_gbt_tree_rule(row):
        # GBT model generated by XGBoost
        score = 0.0            
"""
    func_str += get_python_trees(model, 2, deal_with_missing, num_trees)
    func_str += """
        return score
    eval_df['SCORE_'+model_name] = eval_df.apply(score_by_gbt_tree_rule, axis=1)
    return eval_df  
"""
    return func_str


def print_gbt_rules(model, deal_with_missing=False, num_trees=None):
    print(get_gbt_model_row(model, deal_with_missing, num_trees=None))

def save_gbt_model(model, model_dir_path, model_name, deal_with_missing=False, one_row=False):
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    f = open(model_dir_path+"/__init__.py", "w")
    f.close()

    if one_row:
        func_str = get_gbt_model_row(model, deal_with_missing)
    else:
        func_str = get_gbt_model_df(model, deal_with_missing)
    file_name = model_dir_path + "/model_rules_" + model_name + ".py"
    with open(file_name, "w") as f:
        f.write(func_str)
    print("GBT model was SAVED")


def get_importance(model):
    return pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
