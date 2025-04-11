#Using streamlit as UI.
import os
os.environ["STREAMLIT_WATCH_FILE_SYSTEM"] = "false"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import nest_asyncio
nest_asyncio.apply()
from utils.history_analysis import load_all_seasons, get_head_to_head
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from utils.odds_scraper import get_real_odds
from models.predict import predict_result
from models.predict_overunder import predict_over_under
from utils.get_fixtures import get_upcoming_fixtures

# Title and logo
st.set_page_config(page_title="Premier League Prediction", page_icon="⚽")
st.title("Premier League Prediction")

#Match selection
st.subheader("Select upcoming match")
fixtures = get_upcoming_fixtures("data/epl-2024-GMTStandardTime.csv")
if not fixtures:
    st.error("Failed to load fixture data. Please check your network connection.")
    st.stop()
match_labels = ["Please select a match."] + [f["label"] for f in fixtures]
selected_label = st.selectbox("Please select a match for prediction.", match_labels)
if selected_label == "Please select a match.":
    st.warning("Please select a match from the dropdown list first.")
    st.stop()
match_info = next((f for f in fixtures if f["label"] == selected_label), None)
date = match_info["date"]
home_team = match_info["home"]
away_team = match_info["away"]
st.info(f"Current selection：{date} - {home_team} vs {away_team}")

# Odds input method
st.subheader("Odds input method")
input_method = st.radio("Please select the odds input method.", ["Automatic", "Manual"])
odds = None
odds_source = ""
#Automatic
if input_method == "Automatic":
    odds, success = get_real_odds(date, home_team, away_team)
    odds_source = f"🔗 Automatic（William Hill）"
    if success and odds and odds != [2.0, 3.0, 3.0]:
        st.markdown("#### Current odds（ William Hill）")
        st.write(f"- Home Win（{home_team}）：`{odds[0]}`")
        st.write(f"- Draw：`{odds[1]}`")
        st.write(f"- Away Win（{away_team}）：`{odds[2]}`")
    else:
        st.warning("Failed to retrieve valid odds. The current match does not support automatic fetching. Please try manual input.")
        odds = None
#Manual
elif input_method == "Manual":
    st.markdown("Manually input the current odds")
    col1, col2, col3 = st.columns(3)
    home_odds_input = col1.text_input("Home win odds")
    draw_odds_input = col2.text_input("Draw odds")
    away_odds_input = col3.text_input("Away win odds")
    if home_odds_input and draw_odds_input and away_odds_input:
        try:
            odds = [
                float(home_odds_input),
                float(draw_odds_input),
                float(away_odds_input)
            ]
            odds_source = "Manual"
        except ValueError:
            st.error("Please enter a valid odds number.")
    else:
        st.warning("Please complete all three odds fields.")

# Prediction
if st.button("Start"):
    if not odds or len(odds) != 3:
        st.error("Unable to retrieve valid odds. Please check your input.")
    else:
        label_result, prob_result = predict_result(odds)
        label_ou, prob_ou = predict_over_under(odds)

        st.success(f"Win-Draw-Loss prediction result:{label_result}（Confidence level:{prob_result:.2f}）")
        st.info(f"Over-Under prediction result(2.5):{label_ou}（Confidence level:{prob_ou:.2f}）")
        st.caption(f"{odds_source} | Current odds：{odds}")

# Head-to-head data
st.subheader("Head-to-head record (last four seasons)")
@st.cache_data
def load_data():
    return load_all_seasons("data")
all_data = load_data()
h2h = get_head_to_head(all_data, home_team, away_team)
if h2h.empty:
    st.warning("The two teams have no head-to-head record in the past two seasons.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Result Trend")
        result_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        result_line = h2h.copy()
        result_line["Result_Label"] = result_line["Result"].map(result_map)

        fig, ax = plt.subplots()
        sns.lineplot(x="Date", y="Result_Label", data=result_line, marker="o", ax=ax)
        ax.set_ylabel("Match Result")
        ax.set_xlabel("Match Date")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with col2:
        st.markdown("#### Goal Trend")
        fig2, ax2 = plt.subplots()
        sns.lineplot(x="Date", y="TotalGoals", data=h2h, marker="o", ax=ax2)
        ax2.set_ylabel("Total Goals")
        ax2.set_xlabel("Match Date")
        plt.xticks(rotation=45)
        st.pyplot(fig2)




