
import sys
import os
import pandas as pd

# Add the root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_config, load_data
from src.data.cleaner import clean_data
from src.features.builder import encode_categorical, scale_features, discretize_features
from src.mining.association import run_association_mining
from src.mining.clustering import run_clustering
from src.models.supervised import train_supervised_models
from src.models.semi_supervised import run_semi_supervised_experiment

def main():
    print("Starting HR Analytics Data Mining Pipeline...")
    # Ensure output directories exist
    for d in ["outputs/reports", "outputs/figures", "outputs/tables"]:
        os.makedirs(d, exist_ok=True)

    # 1. Load Config
    config = load_config('configs/params.yaml')

    # 2. Load and Clean Data
    print("Loading and cleaning data...")
    df_raw = load_data(config['raw_data_path'])
    df = clean_data(df_raw)
    # Đồng bộ tên cột: viết hoa chữ cái đầu, bỏ khoảng trắng
    df.columns = [c.strip().capitalize() for c in df.columns]

    # Save EDA summary
    eda_summary = df.describe(include='all')
    eda_summary.to_csv('outputs/reports/eda_summary.csv')
    # Save a sample for preview
    df.head(20).to_csv('outputs/reports/eda_sample.csv', index=False)

    # --- EDA Plots ---
    import matplotlib.pyplot as plt
    import seaborn as sns
    from src.visualization.plots import plot_target_distribution, plot_correlation_heatmap, plot_categorical_vs_target, plot_numerical_distributions
    # Target distribution
    plt.figure()
    plot_target_distribution(df, target_col=config.get('target_col', 'Attrition'))
    plt.savefig('outputs/figures/eda_target_distribution.png'); plt.close()
    # Correlation heatmap
    plt.figure()
    plot_correlation_heatmap(df)
    plt.savefig('outputs/figures/eda_correlation_heatmap.png'); plt.close()
    # Categorical vs target (ví dụ: Department)
    if 'Department' in df.columns:
        plt.figure()
        plot_categorical_vs_target(df, 'Department', target_col=config.get('target_col', 'Attrition'))
        plt.savefig('outputs/figures/eda_department_vs_target.png'); plt.close()
    # Numerical distributions
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if num_cols:
        plt.figure()
        plot_numerical_distributions(df, num_cols[:min(6, len(num_cols))])
        plt.savefig('outputs/figures/eda_numerical_distributions.png'); plt.close()

    # 3. Feature Engineering
    print("Performing feature engineering...")
    df_encoded = encode_categorical(df)
    df_scaled, _ = scale_features(df_encoded)
    df_disc = discretize_features(df)

    # Save processed data
    df_scaled.to_parquet(config['processed_data_path'])
    df_disc.to_csv('data/processed/HR_Discretized.csv', index=False)

    # 4. Association Rules
    print("Running association rule mining...")
    resigned_rules, stayed_rules, frequent_itemsets = run_association_mining(df_disc)
    resigned_rules.to_csv('outputs/reports/association_rules_resigned.csv', index=False)
    stayed_rules.to_csv('outputs/reports/association_rules_stayed.csv', index=False)
    frequent_itemsets.to_csv('outputs/reports/association_frequent_itemsets.csv', index=False)

    # 5. Clustering
    print("Running employee clustering...")
    df_clustered, kmeans, silhouette = run_clustering(df_scaled)
    df_clustered.to_csv('outputs/reports/clustering_results.csv', index=False)
    # Save clustering summary
    cluster_summary = df_clustered.groupby('cluster').mean(numeric_only=True)
    cluster_summary.to_csv('outputs/reports/clustering_summary.csv')
    # Cluster scatter plot (nếu có 2D features)
    if df_scaled.shape[1] >= 2:
        plt.figure()
        plt.scatter(df_scaled.iloc[:,0], df_scaled.iloc[:,1], c=df_clustered['cluster'], cmap='viridis', alpha=0.6)
        plt.title('Clustering Scatter Plot (first 2 features)')
        plt.xlabel(df_scaled.columns[0]); plt.ylabel(df_scaled.columns[1])
        plt.savefig('outputs/figures/clustering_scatter.png'); plt.close()
    # Silhouette score bar
    plt.figure()
    plt.bar(['Silhouette'], [silhouette])
    plt.title('Clustering Silhouette Score')
    plt.savefig('outputs/figures/clustering_silhouette.png'); plt.close()

    # 6. Supervised Modeling
    print("Training supervised models...")
    from src.evaluation.metrics import get_classification_metrics, plot_confusion_matrix, plot_pr_curve
    supervised_res = train_supervised_models(df_scaled)
    # Save supervised metrics if available
    if isinstance(supervised_res, dict):
        pd.DataFrame(supervised_res).T.to_csv('outputs/reports/supervised_metrics.csv')
        # Nếu có y_test, y_pred, y_prob thì vẽ confusion matrix, PR curve
        if 'y_test' in supervised_res and 'xgb' in supervised_res and 'prob' in supervised_res['xgb']:
            y_test = supervised_res['y_test']
            y_prob = supervised_res['xgb']['prob']
            # Classification metrics
            metrics = get_classification_metrics(y_test, y_prob)
            # Save classification report
            import json
            with open('outputs/reports/supervised_classification_report.json', 'w') as f:
                json.dump(metrics['report'], f, indent=2)
            # Confusion matrix
            plt.figure()
            plot_confusion_matrix(metrics['cm'])
            plt.savefig('outputs/figures/supervised_confusion_matrix.png'); plt.close()
            # PR curve
            plt.figure()
            plot_pr_curve(y_test, y_prob, label='XGBoost')
            plt.savefig('outputs/figures/supervised_pr_curve.png'); plt.close()
            # SHAP summary plot (nếu có model và X_test)
            try:
                import shap
                if 'model' in supervised_res['xgb'] and 'X_test' in supervised_res:
                    explainer = shap.TreeExplainer(supervised_res['xgb']['model'])
                    shap_values = explainer.shap_values(supervised_res['X_test'])
                    shap.summary_plot(shap_values, supervised_res['X_test'], show=False)
                    plt.savefig('outputs/figures/supervised_shap_summary.png'); plt.close()
            except Exception as e:
                print(f"SHAP plot error: {e}")

    # 7. Semi-supervised Experiment
    print("Running semi-supervised experiments...")
    semi_supervised_res = run_semi_supervised_experiment(df_scaled)
    pd.DataFrame(semi_supervised_res).T.to_csv('outputs/reports/semi_supervised_comparison.csv')

    print("Pipeline completed successfully! Results are in 'outputs/reports/'.")

if __name__ == "__main__":
    # Ensure CWD is project root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    main()
