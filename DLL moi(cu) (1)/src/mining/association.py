import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_association_mining(df_disc, min_support=0.05, min_threshold=0.5):
    """
    Run Association Rule Mining on discretized data.
    """
    # Filter columns to only those discretized + target 'left'
    cols_to_use = ['jobsatisfaction', 'performancerating', 'monthlyincome', 
                   'totalworkingyears', 'distancefromhome', 'overtime', 
                   'businesstravel', 'department', 'educationfield', 'left']
    
    df_mining = df_disc[cols_to_use].copy()
    # Pre-process 'left' to string for encoding if it's not already
    df_mining['left'] = df_mining['left'].map({1: 'Resigned', 0: 'Stayed'})
    
    # One-hot encode for mlxtend
    df_onehot = pd.get_dummies(df_mining, columns=df_mining.columns)
    
    # Frequent itemsets
    frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
    
    # Association rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    
    # Filter rules where consequent is 'left_Resigned'
    resigned_rules = rules[rules['consequents'].apply(lambda x: 'left_Resigned' in x)]
    stayed_rules = rules[rules['consequents'].apply(lambda x: 'left_Stayed' in x)]
    
    return resigned_rules, stayed_rules, frequent_itemsets
