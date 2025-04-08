import pandas as pd
from datetime import datetime

def get_upcoming_fixtures(csv_path="data/epl-2024-GMTStandardTime.csv"):
    df = pd.read_csv(csv_path)
    fixtures = []

    now = datetime.now()

    for _, row in df.iterrows():
        try:
            match_datetime = pd.to_datetime(row["Date"], dayfirst=True)
        except Exception:
            continue
        if match_datetime < now:
            continue

        date_str = match_datetime.strftime("%Y-%m-%d %H:%M")
        home = row["Home Team"]
        away = row["Away Team"]
        label = f"{date_str} - {home} vs {away}"

        fixtures.append({
            "label": label,
            "date": date_str.split()[0],
            "home": home,
            "away": away
        })

    return fixtures


