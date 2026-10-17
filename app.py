
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Social Media Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# Project heading
st.title("📊 Social Media Performance Predictor")

st.write(
    "Predict Instagram content performance "
    "using machine learning."
)

st.divider()

st.subheader("Enter Your Instagram Post Details")

# Input fields
caption_length = st.number_input(
    "Caption Length",
    min_value=0,
    value=100
)

hashtag_count = st.number_input(
    "Number of Hashtags",
    min_value=0,
    value=5
)

posting_hour = st.selectbox(
    "Posting Hour",
    list(range(24))
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday",
     "Thursday", "Friday", "Saturday", "Sunday"]
)

media_type = st.selectbox(
    "Media Type",
    ["Image", "Video", "Carousel", "Reel"]
)

# Prediction button
if st.button("Predict Performance"):

    st.success("Your prediction section is ready!")

    st.write("Caption Length:", caption_length)
    st.write("Hashtags:", hashtag_count)
    st.write("Posting Hour:", posting_hour)
    st.write("Day:", day_of_week)
    st.write("Media Type:", media_type)
