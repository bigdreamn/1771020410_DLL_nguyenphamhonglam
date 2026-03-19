import pandas as pd
import yaml
import os

def load_config(config_path="configs/params.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Parquet.")

if __name__ == "__main__":
    config = load_config()
    df = load_data(config["raw_data_path"])
    print(df.head())
    print(df.info())
