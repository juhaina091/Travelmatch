import streamlit as st
import pandas as pd
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the data
df = pd.read_csv("destinations.csv")

st.set_page_config(page_title="🌍 TravelMatch", layout="wide")
st.title("🌍 TravelMatch - Smart Travel Explorer")

# --- Sidebar Navigation ---
st.sidebar.title("🔎 Explore Options")
mode = st.sidebar.radio("Choose how you'd like to explore:", [
    "Smart Recommendation",
    "By Interest",
    "By Budget",
    "By Month",
    "Data Insights",
    "Ask TravelBot"
])

# --- Match Score Function ---
def match_score(tags, interests):
    return len(set([t.strip() for t in tags.split(",")]) & set(interests))

# --- Load Chatbot Model ---
@st.cache_resource
def load_chat_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_chat_model()
if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None

# --- Smart Recommendation ---
if mode == "Smart Recommendation":
    st.header("✈️ Get Personalized Destination Matches")
    budget = st.selectbox("Choose your budget level", ["Low", "Medium", "High"])
    month = st.selectbox("When do you plan to travel?", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"])
    interests = st.multiselect("What are your interests?", [
        "culture", "history", "beach", "adventure", "nature",
        "food", "romantic", "temples", "wellness", "art", "wildlife"])

    if st.button("🔍 Find Destinations"):
        results = df[(df["Budget"] == budget) & df["Best_Months"].str.contains(month, case=False)]
        if interests:
            results = results.copy()
            results["Match_Score"] = results["Tags"].apply(lambda tags: match_score(tags, interests))
            results = results.sort_values("Match_Score", ascending=False)
        
        if not results.empty:
            st.subheader("🏝 Top Travel Matches")
            for _, row in results.head(5).iterrows():
                st.markdown(f"### {row['Destination']}, {row['Country']}")
                st.write(f"**Budget**: {row['Budget']}")
                st.write(f"**Best Months**: {row['Best_Months']}")
                st.write(f"**Famous for**: {row['Tags']}")
                st.write("---")
        else:
            st.warning("No destinations match your filters. Try changing some options.")

# --- Explore by Interest ---
elif mode == "By Interest":
    st.header("🎯 Explore Destinations by Interest")
    interest = st.selectbox("Choose an interest:", sorted(set(tag for tags in df['Tags'] for tag in tags.split(","))))
    matches = df[df['Tags'].str.contains(interest.strip(), case=False)]
    st.write(f"### 🌟 Destinations known for {interest.strip()}:")
    st.dataframe(matches[["Destination", "Country", "Budget", "Best_Months", "Tags"]])

# --- Explore by Budget ---
elif mode == "By Budget":
    st.header("💸 Explore Destinations by Budget")
    budget = st.selectbox("Select your budget level:", ["Low", "Medium", "High"])
    matches = df[df['Budget'] == budget]
    st.write(f"### 🏖️ Best places for {budget} budget:")
    st.dataframe(matches[["Destination", "Country", "Best_Months", "Tags"]])

# --- Explore by Month ---
elif mode == "By Month":
    st.header("🗓 Explore Destinations by Travel Month")
    month = st.selectbox("Select month:", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"])
    matches = df[df['Best_Months'].str.contains(month, case=False)]
    st.write(f"### ✨ Top destinations to visit in {month}:")
    st.dataframe(matches[["Destination", "Country", "Budget", "Tags"]])

# --- Data Insights ---
elif mode == "Data Insights":
    st.header("📊 Travel Trends and Insights")
    top_tags = pd.Series(
        tag.strip() for tags in df['Tags'] for tag in tags.split(',')).value_counts().head(10)
    st.subheader("🌟 Top Travel Themes")
    st.bar_chart(top_tags)

    st.subheader("📍 Most Popular Budget Categories")
    st.bar_chart(df['Budget'].value_counts())

# --- AI Chatbot (DialoGPT) ---
elif mode == "Ask TravelBot":
    st.header("🤖 Ask TravelBot Anything")
    user_input = st.text_input("Type your question:")
    if user_input:
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        if st.session_state.chat_history_ids is not None:
            bot_input_ids = torch.cat([st.session_state.chat_history_ids, input_ids], dim=-1)
        else:
            bot_input_ids = input_ids

        st.session_state.chat_history_ids = model.generate(
            bot_input_ids,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(
            st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        st.success(response)
