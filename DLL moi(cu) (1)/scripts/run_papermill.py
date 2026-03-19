import papermill as pm
import os

def run_notebooks():
    notebooks = [
        "01_eda.ipynb",
        "02_preprocess_feature.ipynb",
        "03_mining_or_clustering.ipynb",
        "04_modeling.ipynb",
        "04b_semi_supervised.ipynb",
        "05_evaluation_report.ipynb"
    ]
    
    nb_dir = "notebooks"
    output_dir = "outputs/reports"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for nb in notebooks:
        print(f"Executing {nb}...")
        input_path = os.path.join(nb_dir, nb)
        output_path = os.path.join(output_dir, f"executed_{nb}")
        pm.execute_notebook(input_path, output_path)
        print(f"Finished {nb}. Output saved to {output_path}")

if __name__ == "__main__":
    # Ensure CWD is project root
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    run_notebooks()
