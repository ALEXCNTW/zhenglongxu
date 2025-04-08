import requests

API_KEY = "584b756f612a704a7137654e7b2a1d57"  # API

def get_real_odds(date: str, home_team: str, away_team: str):
    # ...
    try:
        response = requests.get(
            "https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
            params={
                "regions": "uk",
                "markets": "h2h",
                "oddsFormat": "decimal",
                "apiKey": API_KEY
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        for match in data:
            if home_team.lower() in match["home_team"].lower() and away_team.lower() in match["away_team"].lower():
                for bookmaker in match.get("bookmakers", []):
                    if bookmaker["key"] == "williamhill":
                        market = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
                        if market:
                            outcomes = market["outcomes"]
                            odds_map = {o["name"].lower(): o["price"] for o in outcomes}
                            odds = [
                                odds_map.get(home_team.lower(), 2.0),
                                odds_map.get("draw", 3.0),
                                odds_map.get(away_team.lower(), 3.0)
                            ]
                            return odds, True

        print("No matching matches or William Hill odds found.")
        return [2.0, 3.0, 3.0], False

    except Exception as e:
        print(f"Failed to fetch odds: {e}")
        return [2.0, 3.0, 3.0], False





