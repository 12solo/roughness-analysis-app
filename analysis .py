import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

def get_stats_summary(df, group_col, value_col):
    summary = df.groupby(group_col)[value_col].agg([
        'mean', 'std', 'count', 'min', 'max'
    ]).reset_index()
    
    # Calculate 95% CI
    summary['ci_95'] = 1.96 * (summary['std'] / np.sqrt(summary['count']))
    return summary

def perform_anova(df, value_col, group_col):
    """Performs One-Way ANOVA across groups"""
    groups = [group[value_col].values for name, group in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

def perform_ttest(df, value_col, group_col, val1, val2):
    """Independent t-test between two specific groups"""
    group1 = df[df[group_col] == val1][value_col]
    group2 = df[df[group_col] == val2][value_col]
    t_stat, p_val = stats.ttest_ind(group1, group2)
    return t_stat, p_val