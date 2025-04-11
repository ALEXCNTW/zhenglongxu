#This is a preprocess part. Read all the datas and get features in them. Save as .pkl .

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Win-Draw-Loss label encoding
def encode_result(result_str):
    return {"H": 0, "D": 1, "A": 2}[result_str]

# Over-Under label encoding
def encode_over_under(home_goals, away_goals, threshold=2.5):
    return 1 if (home_goals + away_goals) > threshold else 0

# Load all CSV files
def load_all_data():
    seasons = [
        "C:/Users/10699/Desktop/PythonProject/data/2020-2021.csv",
        "C:/Users/10699/Desktop/PythonProject/data/2021-2022.csv",
        "C:/Users/10699/Desktop/PythonProject/data/2022-2023.csv",
        "C:/Users/10699/Desktop/PythonProject/data/2023-2024.csv"
    ]
    dfs = []
    for season in seasons:
        path = os.path.join("data", season)
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# Process Win-Draw-Loss data（William Hill）
def preprocess_result(df):
    df = df.dropna(subset=["WHH", "WHD", "WHA", "FTR"])
    df["label"] = df["FTR"].map(encode_result)
    X = df[["WHH", "WHD", "WHA"]].values
    y = df["label"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    os.makedirs("data_preprocessing", exist_ok=True)
    np.savez("data_preprocessing/processed_result.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    joblib.dump(scaler, "data_preprocessing/scaler_result.pkl")
    print("Win-Draw-Loss data preprocessing completed.")

# Process Over-Under data（Bet365）
def preprocess_over_under(df):
    df = df.dropna(subset=["B365>2.5", "B365<2.5", "FTHG", "FTAG"])
    df["over_under"] = df.apply(lambda row: encode_over_under(row["FTHG"], row["FTAG"]), axis=1)
    X = df[["B365>2.5", "B365<2.5"]].values
    y = df["over_under"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    os.makedirs("data_preprocessing", exist_ok=True)
    np.savez("data_preprocessing/processed_overunder.npz", X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    joblib.dump(scaler, "data_preprocessing/scaler_overunder.pkl")
    print("Over-Under data preprocessing completed.")

if __name__ == "__main__":
    df_all = load_all_data()
    preprocess_result(df_all.copy())
    preprocess_over_under(df_all.copy())

