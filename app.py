
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Social Media Performance Predictor",
    page_icon="🩵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cta_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(
        "Could not load the trained model. "
        "Please check that cta_model.pkl is in the project folder."
    )
    st.stop()


# ---------------- SKY BLUE + PINK CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(
        135deg,
        #eafaff 0%,
        #fdf0f7 48%,
        #eaf5ff 100%
    );
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero banner */
.hero {
    background: linear-gradient(
        120deg,
        #8edcf5 0%,
        #b8eaff 45%,
        #f5b9d7 100%
    );
    padding: 38px 35px;
    border-radius: 25px;
    color: #23445d;
    margin-bottom: 28px;
    box-shadow: 0 12px 30px rgba(126, 190, 220, 0.20);
    border: 1px solid rgba(255,255,255,0.8);
}

.hero h1 {
    color: #24445e;
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.25;
    margin: 0;
}

.hero p {
    color: #395c73;
    font-size: 1rem;
    margin-top: 12px;
}

.pill {
    display: inline-block;
    background: rgba(255,255,255,0.65);
    color: #bd5d91;
    padding: 7px 14px;
    border-radius: 22px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-bottom: 14px;
}

/* Section headings */
.section-title {
    color: #477f9d;
    font-size: 1.35rem;
    font-weight: 700;
    margin: 18px 0 5px 0;
}

.helper {
    color: #718397;
    font-size: 0.9rem;
    margin-bottom: 20px;
}

/* Input cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid #f3cfe2;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 5px 18px rgba(160, 190, 210, 0.10);
}

/* Labels and text */
label {
    color: #42647a !important;
    font-weight: 500 !important;
}

.stMarkdown p {
    color: #587084;
}

h2, h3 {
    color: #477f9d;
}

/* Input fields */
div[data-baseweb="select"] > div {
    border-radius: 11px;
    border-color: #c4e7f5;
    background-color: #ffffff;
}

div[data-baseweb="input"] > div {
    border-radius: 11px;
    border-color: #c4e7f5;
    background-color: #ffffff;
}

/* Prediction button */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    border: 1px solid #f3c3dd;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    color: #ffffff;
    font-weight: 700;
    background: linear-gradient(
        100deg,
        #78cfee,
        #e99ac5
    );
    transition: 0.2s;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    border: 1px solid #e99ac5;
    color: #ffffff;
    box-shadow: 0 7px 20px rgba(219, 151, 190, 0.25);
    transform: translateY(-1px);
}

/* Result message */
div[data-testid="stAlert"] {
    border-radius: 15px;
}

/* Footer */
.footer {
    text-align: center;
    color: #8194a5;
    font-size: 0.82rem;
    padding: 25px 0 5px;
}

hr {
    border-color: #f1d9e7;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="pill">🩵 AI • DATA • SOCIAL MEDIA</div>
    <h1>Social Media<br>Performance Predictor 🩷</h1>
    <p>
        Explore your Instagram post details and predict whether
        a call-to-action is likely to be present.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- INTRODUCTION ----------------
st.markdown(
    '<div class="section-title">✨ Let’s get started</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="helper">Enter the details below to generate a model prediction.</div>',
    unsafe_allow_html=True
)


# ---------------- INPUT FORM ----------------
with st.form("prediction_form"):

    st.markdown("### 🩵 Account & Audience")

    col1, col2 = st.columns(2)

    with col1:
        account_type = st.selectbox(
            "Account type",
            ["Creator", "Business", "Personal"]
        )

        follower_count = st.number_input(
            "Follower count",
            min_value=0,
            max_value=100000000,
            value=5000,
            step=100
        )

    with col2:
        traffic_source = st.selectbox(
            "Traffic source",
            ["Feed", "Explore", "Hashtags", "Profile", "Other"]
        )

        media_type = st.selectbox(
            "Media type",
            ["Image", "Video", "Carousel", "Reel", "Story"]
        )

    st.markdown("---")
    st.markdown("### 🩷 Post Details")

    col3, col4 = st.columns(2)

    with col3:
        content_category = st.selectbox(
            "Content category",
            [
                "Education",
                "Entertainment",
                "Lifestyle",
                "Fashion",
                "Food",
                "Travel",
                "Technology",
                "Other"
            ]
        )

        posting_hour = st.slider(
            "Posting hour",
            min_value=0,
            max_value=23,
            value=18,
            help="0 = midnight, 12 = noon, 18 = 6 PM"
        )

    with col4:
        posting_day = st.selectbox(
            "Day of the week",
            options=list(range(7)),
            format_func=lambda day: [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ][day]
        )

    st.markdown("")
    submitted = st.form_submit_button(
        "🩵 Predict Post Details 🩷",
        use_container_width=True
    )


# ---------------- PREDICTION ----------------
if submitted:

    input_data = pd.DataFrame([{
        "account_type": account_type,
        "follower_count": follower_count,
        "media_type": media_type,
        "content_category": content_category,
        "traffic_source": traffic_source,
        "posting_hour": posting_hour,
        "posting_day": posting_day
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.markdown(
            '<div class="section-title">💗 Your Prediction</div>',
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.success(
                "The model predicts that this post is likely to have "
                "a call-to-action (CTA)."
            )
        else:
            st.info(
                "The model predicts that this post is likely not to have "
                "a call-to-action (CTA)."
            )

        st.caption(
            "This prediction concerns CTA presence only. "
            "It does not predict likes, reach, views, or engagement."
        )

        with st.expander("View the details you entered"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("""
<div class="footer">
    Made with 🩵 & 🩷 using Python and Streamlit<br>
    Social Media Performance Predictor • Student Project
</div>
""", unsafe_allow_html=True)


# ---------------- ANALYTICS DASHBOARD ----------------

import matplotlib.pyplot as plt

st.markdown("---")
st.markdown(
    "<h1 style='text-align:center; color:#65A9D8;'>"
    "📊 Instagram Analytics Dashboard</h1>",
    unsafe_allow_html=True
)

DATA_PATH = BASE_DIR / "data" / "Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv"

try:
    analytics_df = pd.read_csv(DATA_PATH)

    st.markdown(
        "<p style='text-align:center; color:#D98BAA;'>"
        "Explore your Instagram dataset</p>",
        unsafe_allow_html=True
    )

    # Filters
    col1, col2 = st.columns(2)

    with col1:
        account_options = ["All"] + sorted(
            analytics_df["account_type"].dropna().astype(str).unique().tolist()
        )
        selected_account = st.selectbox(
            "Filter by Account Type",
            account_options
        )

    with col2:
        media_options = ["All"] + sorted(
            analytics_df["media_type"].dropna().astype(str).unique().tolist()
        )
        selected_media = st.selectbox(
            "Filter by Media Type",
            media_options
        )

    # Apply filters
    filtered_df = analytics_df.copy()

    if selected_account != "All":
        filtered_df = filtered_df[
            filtered_df["account_type"].astype(str) == selected_account
        ]

    if selected_media != "All":
        filtered_df = filtered_df[
            filtered_df["media_type"].astype(str) == selected_media
        ]

    # Metric cards
    st.markdown("### ✨ Dataset Overview")

    m1, m2, m3 = st.columns(3)

    m1.metric("Total Posts", len(filtered_df))

    if not filtered_df.empty:
        m2.metric(
            "Average Followers",
            f"{filtered_df['follower_count'].mean():,.0f}"
        )

        if "has_call_to_action" in filtered_df.columns:
            cta_count = (
                filtered_df["has_call_to_action"]
                .astype(str).str.lower()
                .isin(["1", "true", "yes"])
                .sum()
            )
            m3.metric("Posts with CTA", int(cta_count))
        else:
            m3.metric("Account Types", filtered_df["account_type"].nunique())

    st.markdown("---")
    st.markdown("### 📈 Visual Insights")

    if filtered_df.empty:
        st.warning("No posts match these filters. Please change your selections.")
    else:
        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("**Posts by Media Type**")
            media_counts = filtered_df["media_type"].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            media_counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Media Type")
            ax.set_ylabel("Number of Posts")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with chart2:
            st.markdown("**Posts by Content Category**")
            category_counts = filtered_df["content_category"].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            category_counts.plot(kind="bar", ax=ax)
            ax.set_xlabel("Content Category")
            ax.set_ylabel("Number of Posts")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        chart3, chart4 = st.columns(2)

        with chart3:
            st.markdown("**Posts by Account Type**")
            account_counts = filtered_df["account_type"].value_counts()

            fig, ax = plt.subplots(figsize=(6, 4))
            account_counts.plot(
                kind="pie",
                autopct="%1.1f%%",
                ax=ax
            )
            ax.set_ylabel("")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with chart4:
            st.markdown("**Average Followers by Media Type**")
            follower_avg = filtered_df.groupby("media_type")[
                "follower_count"
            ].mean().sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(6, 4))
            follower_avg.plot(kind="bar", ax=ax)
            ax.set_xlabel("Media Type")
            ax.set_ylabel("Average Followers")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

except FileNotFoundError:
    st.error("Dataset not found. Check the CSV file inside your data folder.")

except Exception as e:
    st.error(f"Dashboard error: {e}")

