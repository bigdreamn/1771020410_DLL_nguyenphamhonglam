import matplotlib.pyplot as plt
import seaborn as sns

def plot_target_distribution(df, target_col='left'):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title('Distribution of Employee Attrition (Left)')
    plt.show()

def plot_correlation_heatmap(df):
    plt.figure(figsize=(12, 10))
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

def plot_categorical_vs_target(df, cat_col, target_col='left'):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=cat_col, hue=target_col, data=df, palette='magma')
    plt.title(f'{cat_col} vs {target_col}')
    plt.xticks(rotation=45)
    plt.show()

def plot_numerical_distributions(df, num_cols):
    fig, axes = plt.subplots(nrows=len(num_cols)//2 + 1, ncols=2, figsize=(15, 20))
    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()
