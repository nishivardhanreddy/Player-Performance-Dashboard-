import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Load data and model
df = pd.read_csv("cricket_data.csv")
model = joblib.load("models/form_model.pkl")

st.title("🏏 Cricket Player Performance Dashboard")

# Player & Season Selectors
players = sorted(df["Player_Name"].dropna().unique())
years = sorted(df["Year"].dropna().unique())

selected_player = st.selectbox("Select Player", players)
selected_season = st.selectbox("Select Season", years)

# Filter player data
player_data = df[(df["Player_Name"] == selected_player) & (df["Year"] == selected_season)]

if not player_data.empty:
    st.subheader(f"📊 Batting Stats: {selected_player} ({selected_season})")

    st.write(player_data[[
        "Matches_Batted", "Runs_Scored", "Batting_Average", "Batting_Strike_Rate",
        "Centuries", "Half_Centuries", "Fours", "Sixes"
    ]].T)

    # Batting Chart
    fig = go.Figure(go.Bar(
        x=["Runs", "Avg", "SR", "100s", "50s"],
        y=[
            player_data["Runs_Scored"].values[0],
            player_data["Batting_Average"].values[0],
            player_data["Batting_Strike_Rate"].values[0],
            player_data["Centuries"].values[0],
            player_data["Half_Centuries"].values[0],
        ],
        marker_color='mediumseagreen'
    ))
    fig.update_layout(title="Batting Stats Overview", yaxis_title="Value")
    st.plotly_chart(fig)

    # ML Prediction: In Form / Out of Form
    features = ["Batting_Average", "Batting_Strike_Rate", "Matches_Batted", "Centuries", "Half_Centuries"]
    input_features = player_data[features].values
    prediction = model.predict(input_features)[0]
    st.markdown("### 🧠 Form Prediction")
    if prediction == 1:
        st.success(f"{selected_player} is predicted to be ✅ **In Form**")
    else:
        st.warning(f"{selected_player} is predicted to be ❌ **Out of Form**")

    # Bowling Stats Section
    st.subheader(f"🎯 Bowling Stats: {selected_player} ({selected_season})")

    st.write(player_data[[
        "Matches_Bowled", "Wickets_Taken", "Bowling_Average", "Economy_Rate", "Bowling_Strike_Rate",
        "Four_Wicket_Hauls", "Five_Wicket_Hauls"
    ]].T)

    # Bowling Chart
    fig_bowl = go.Figure(go.Bar(
        x=["Wickets", "Avg", "Economy", "Strike Rate"],
        y=[
            player_data["Wickets_Taken"].values[0],
            player_data["Bowling_Average"].values[0],
            player_data["Economy_Rate"].values[0],
            player_data["Bowling_Strike_Rate"].values[0],
        ],
        marker_color='indianred'
    ))
    fig_bowl.update_layout(title="Bowling Stats Overview", yaxis_title="Value")
    st.plotly_chart(fig_bowl)

else:
    st.error("No data found for that player in the selected season.")

# ========================
# 🏆 SEASON LEADERBOARD
# ========================
st.markdown("---")
st.header("🏆 Season Leaderboard")

leaderboard_stat = st.selectbox("Select Stat to Rank Players", ["Runs Scored", "Wickets Taken"])

if leaderboard_stat == "Runs Scored":
    top_bat = df[df["Year"] == selected_season].copy()
    top_bat["Runs_Scored"] = pd.to_numeric(top_bat["Runs_Scored"], errors="coerce")
    top_bat = top_bat[["Player_Name", "Runs_Scored"]].dropna()
    top_bat = top_bat.groupby("Player_Name").sum().sort_values(by="Runs_Scored", ascending=False).head(5)

    st.subheader(f"Top 5 Run Scorers in {selected_season}")
    st.table(top_bat.reset_index())

    fig = px.bar(top_bat.reset_index(), x="Runs_Scored", y="Player_Name", orientation='h',
                 color="Runs_Scored", title=f"Top 5 Run Scorers in {selected_season}")
    st.plotly_chart(fig)

elif leaderboard_stat == "Wickets Taken":
    top_bowl = df[df["Year"] == selected_season].copy()
    top_bowl["Wickets_Taken"] = pd.to_numeric(top_bowl["Wickets_Taken"], errors="coerce")
    top_bowl = top_bowl[top_bowl["Wickets_Taken"] > 0]
    top_bowl = top_bowl[["Player_Name", "Wickets_Taken"]].dropna()
    top_bowl = top_bowl.groupby("Player_Name").sum().sort_values(by="Wickets_Taken", ascending=False).head(5)

    st.subheader(f"Top 5 Wicket Takers in {selected_season}")

    if top_bowl.empty:
        st.warning("No bowlers with wickets found for this season.")
    else:
        st.table(top_bowl.reset_index())

        fig = px.bar(top_bowl.reset_index(), x="Wickets_Taken", y="Player_Name", orientation='h',
                     color="Wickets_Taken", title=f"Top 5 Wicket Takers in {selected_season}")
        st.plotly_chart(fig)
