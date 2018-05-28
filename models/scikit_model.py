import numpy as np
import pandas as pd
import os


def get_gbt_feature_importance(gbt_model, column_names):
    gbt_all_importances = pd.Series(gbt_model.feature_importances_, index=column_names, name="feature importance").sort_values(ascending=False)
    return gbt_all_importances[gbt_all_importances > 0]

def get_gbt_model_rules(gbt_model, column_names, k=None):
    """Export scikit learn GBT model rules to function"""
    if len(column_names) == 1:
        # duplication of the same feature name is needed if only 1 was used for GBT training!
        column_names = column_names*3
    INDENT = str("    ")
    num_of_trees = len(gbt_model.estimators_)
    if k!=None:
        num_of_trees = min(k, num_of_trees)
    
    func_str = ["def score_gbt(eval_df, model_name):"]
    func_str += [INDENT + '"""Score segment by gbt rule"""']
    func_str += []
    func_str += [INDENT + "def score_by_gbt_tree_rule(row):"] 
    func_str += [INDENT + '# GBT model generated by Scikit-Learn']
    func_str += [INDENT + INDENT + "score = 0.0",]
    for i in range(num_of_trees):
        func_str += [INDENT + INDENT + '### tree_%i ###' % (i+1)]
        func_str += print_regression_tree_with_names(gbt_model.estimators_[i,0].tree_, column_names, INDENT)
        func_str += ['']
    func_str += [INDENT + INDENT + "return score"]
    func_str += []
    func_str += [INDENT + "eval_df['SCORE_'+model_name] = eval_df.apply(score_by_gbt_tree_rule, axis=1)"]
    func_str += [INDENT + "return eval_df"]
    return func_str


def print_regression_tree_with_names(tree, column_names, indent):
    """Convert scikit learn GBT model tree into readable code"""
    return print_tree_with_names(tree, column_names, indent, regression=True)

def print_decision_tree_with_names(tree, column_names, indent):
    """Convert scikit learn DT model tree into readable code"""
    return print_tree_with_names(tree, column_names, indent, regression=False)

def print_tree_with_names(tree, column_names, indent, regression):
    left      = tree.children_left
    right     = tree.children_right
    threshold = tree.threshold
    features  = [column_names[i] for i in tree.feature]
    value = tree.value
    depth = 0
    
    def get_indent(depth, indent):   
        s = ""
        for j in range(depth):
            s += indent
        return s
    
    def recurse(left, right, threshold, features, node, depth, regression):
        depth+=1
        text = []
        line_indent = get_indent(depth, indent)
        if (threshold[node] != -2):
            text += [line_indent + 'if row["%s"] <= %f:' % (features[node], threshold[node])]
            if left[node] != -1:
                text += recurse(left, right, threshold, features, left[node], depth, regression)
            text += [line_indent + "else:"]
            if right[node] != -1:
                text += recurse(left, right, threshold, features, right[node], depth, regression)
        else:
            if regression:
                text += [line_indent + 'score += %f' % value[node][0][0]]
            else:
                text += [line_indent + '[neg = %f pos = %f]' % (value[node][0][0], value[node][0][1])]
        return text
    
    return recurse(left, right, threshold, features, 0, 1, regression)

def print_gbt_rules(gbt_model, column_names, k=None):
    """Print rules of the provided 'gbt_model'. You must provide the name of the columns you used for training the model in a list  'column_names'."""
    func_str = get_gbt_model_rules(gbt_model, column_names, k=k)
    for line in func_str:
        print(line)


def save_gbt_model(gbt_model, column_names, model_dir_path, model_name):
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    f = open(model_dir_path+'/__init__.py', 'w')
    f.close()

    func_str = get_gbt_model_rules(gbt_model, column_names, k=None)
    file_name = model_dir_path + "/model_rules_" + model_name + ".py"
    with open(file_name, 'w') as f:
        for line in func_str:
            f.write(line + '\n')
    print("GBT model was SAVED")
    

def get_lreg_coefficients(lreg_model, column_names):
    intercept_df = pd.DataFrame({"name":"(intercept)", "value": lreg_model.intercept_})
    coef_df = pd.DataFrame({"name": column_names, "value": lreg_model.coef_[0]})
    return intercept_df.append(coef_df, ignore_index=True)


def save_lreg_model(lreg_model, column_names, model_dir_path, model_name, is_normalized=False):
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    
    lreg_coeffs = get_lreg_coefficients(lreg_model, column_names)
    lreg_coeffs.to_csv(model_dir_path+"/"+model_name+"_"+"coefficients.csv", index=False, sep="|")

    with open(model_dir_path+"/model_rules_lreg.py", 'w') as f:
        f.write("""\

import pandas as pd
import numpy as np
import os

def score_lreg(eval_df, model_name, normalize_before_scoring=%s):
    \"""Score logistic regression model. You can normalize the data before scoring.\"""
    
    def normalize(df):
       return (df - df.min()) / (df.max() - df.min())

    def get_teta(row):
        teta = intercept
        for feat in coeff_dict:
            teta += row[feat] * coeff_dict[feat]
        return teta

    model_dir_path = os.path.dirname(os.path.realpath(__file__))
    coeff = pd.read_csv(model_dir_path+'/'+model_name+'_coefficients.csv', sep="|").set_index("name")
    cols = list(coeff.index)
    cols.remove("(intercept)")
    if normalize_before_scoring:
       relevant_df = normalize(eval_df[cols])
    else:
       relevant_df = eval_df[cols]
    #print(relevant_df.head())
    intercept = coeff["value"]["(intercept)"]
    coeff = coeff.drop("(intercept)")
    coeff_dict = coeff.to_dict()["value"]
    eval_df["SCORE_"+model_name] = relevant_df.apply(lambda row: 1/(1+np.exp(-get_teta(row))), axis=1)
    return eval_df

""" % is_normalized)
    print("LREG model was SAVED")
        
