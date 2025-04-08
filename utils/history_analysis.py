import pandas as pd
import os

def load_all_seasons(data_path="data"):
    seasons = ["2020-2021.csv", "2021-2022.csv", "2022-2023.csv", "2023-2024.csv"]
    all_matches = []
    for season_file in seasons:
        file_path = os.path.join(data_path, season_file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_matches.append(df)
        else:
            print(f"File not found:{file_path}")
    if all_matches:
        return pd.concat(all_matches, ignore_index=True)
    else:
        print("No data files found. Please check the file path and naming.")
        return pd.DataFrame()
def get_head_to_head(df, team1, team2):
    mask = ((df["HomeTeam"] == team1) & (df["AwayTeam"] == team2)) | \
           ((df["HomeTeam"] == team2) & (df["AwayTeam"] == team1))
    h2h_matches = df.loc[mask, ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()

    def calculate_result(row):
        if row["FTHG"] > row["FTAG"]:
            return "H"
        elif row["FTHG"] < row["FTAG"]:
            return "A"
        else:
            return "D"

    h2h_matches["Result"] = h2h_matches.apply(calculate_result, axis=1)
    h2h_matches["TotalGoals"] = h2h_matches["FTHG"] + h2h_matches["FTAG"]
    h2h_matches["Date"] = pd.to_datetime(h2h_matches["Date"], format="%d/%m/%Y", errors="coerce")
    h2h_matches = h2h_matches.dropna(subset=["Date"]).sort_values("Date")

    return h2h_matches
