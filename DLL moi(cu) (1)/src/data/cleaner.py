import pandas as pd
import numpy as np

def clean_data(df):
    """
    Apply basic cleaning to the HR dataset.
    """
    # Create a copy to avoid side effects
    df_clean = df.copy()
    
    # Check for duplicates
    df_clean = df_clean.drop_duplicates()
    
    # Rename columns to standard snake_case if necessary 
    # Clean column names (strip whitespace and lowercase)
    df_clean.columns = [col.strip().lower() for col in df_clean.columns]
    
    if 'sales' in df_clean.columns:
        df_clean = df_clean.rename(columns={'sales': 'department'})
        
    # Map Attrition (or similar) to binary if it exists
    # Look for Attrition regardless of case
    attrition_col = [c for c in df_clean.columns if c.lower() == 'attrition']
    if attrition_col:
        df_clean['left'] = df_clean[attrition_col[0]].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    elif 'left' not in df_clean.columns:
        # If neither Attrition nor left is found, we might have an issue
        pass

    # Check for missing values (usually none in this dataset)
    # Fill with median for numeric columns, mode for categorical
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    return df_clean

def handle_outliers(df, column):
    """
    Handle outliers using IQR method for a specific column.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    
    return df
