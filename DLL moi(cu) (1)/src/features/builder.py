import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def encode_categorical(df):
    """
    Encode categorical features for IBM Dataset.
    """
    df_encoded = df.copy()
    
    # Identify categorical columns (excluding some we might want to drop or handle separately)
    # Using 'object' dtype as a proxy
    cat_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Drop EmpID as it's just an identifier
    if 'EmpID' in cat_cols:
        df_encoded = df_encoded.drop(columns=['EmpID'])
        cat_cols.remove('EmpID')
    
    # One-hot encode the rest
    df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
    
    return df_encoded

def scale_features(df, exclude_cols=['left']):
    """
    Scale numeric features using StandardScaler.
    """
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    cols_to_scale = [col for col in df.columns if col not in exclude_cols]
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])
    
    return df_scaled, scaler

def discretize_features(df):
    """
    Discretize features for Association Rule Mining (IBM Dataset version).
    """
    df_disc = df.copy()
    
    # Map JobSatisfaction (1-4) to labels
    if 'JobSatisfaction' in df_disc.columns:
        df_disc['JobSatisfaction'] = df_disc['JobSatisfaction'].map({1: 'Low_Sat', 2: 'Mid_Sat', 3: 'High_Sat', 4: 'VeryHigh_Sat'})
    
    # Map PerformanceRating (3-4) to labels
    if 'PerformanceRating' in df_disc.columns:
        df_disc['PerformanceRating'] = df_disc['PerformanceRating'].map({3: 'Good_Perf', 4: 'Excellent_Perf'})
    
    # MonthlyIncome: low, medium, high
    if 'MonthlyIncome' in df_disc.columns:
        df_disc['MonthlyIncome'] = pd.qcut(df['MonthlyIncome'], 
                                           q=3, 
                                           labels=['Low_Income', 'Mid_Income', 'High_Income'])
    
    # TotalWorkingYears: junior, senior
    if 'TotalWorkingYears' in df_disc.columns:
        df_disc['TotalWorkingYears'] = pd.cut(df['TotalWorkingYears'], 
                                              bins=[-1, 5, 15, 100], 
                                              labels=['Junior', 'Experienced', 'Senior'])
    
    return df_disc
