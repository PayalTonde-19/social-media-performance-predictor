
import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Social Media Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #EAF5FF;
}
h1, h2, h3 {
    color: #24445E;
}
div[data-testid="stMetric"] {
    background-color: #FDF0F7;
    border: 1px solid #E99AC5;
    padding: 14px;
    border-radius: 12px;
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #E99AC5;
    color: #24445E;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("cta_model.pkl")


@st.cache_data
def load_processed_data():
    return pd.read_csv("data/processed_instagram_data.csv")


@st.cache_data
def load_trend_data():
    paths = [
        "data/Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv",
        "Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv",
        "data/processed_instagram_data.csv",
    ]

    for path in paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    raise FileNotFoundError("Instagram dataset was not found.")


try:
    model = load_model()
    processed_df = load_processed_data()
    df = load_trend_data()
except Exception as e:
    st.error(f"Could not load project files: {e}")
    st.stop()

# --------------------------------------------------
# VALID MODEL INPUT OPTIONS
# These must match the values used to train the model.
# --------------------------------------------------

# --------------------------------------------------
# DROPDOWN OPTIONS — RESTORED
# --------------------------------------------------
account_options = [
    "Creator", "Business", "Personal", "Influencer", "Brand"
]

media_options = [
    "Image", "Video", "Carousel", "Reel", "Story"
]

category_options = [
    "Beauty", "Fashion", "Lifestyle", "Food",
    "Travel", "Fitness", "Music", "Education",
    "Technology", "Entertainment", "Health",
    "Photography", "Business", "Art", "Gaming",
    "Sports", "Other"
]

traffic_options = [
    "Feed","Hashtags", "Explore", "Home",
    "Profile", "Search", "Other"
]

day_options = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

# Verify that the expected model input columns exist.
MODEL_COLUMNS = [
    "account_type",
    "follower_count",
    "media_type",
    "content_category",
    "traffic_source",
    "posting_day",
]

if not all(col in processed_df.columns for col in MODEL_COLUMNS):
    st.error(
        "The processed dataset is missing one or more model input columns. "
        "Please check data/processed_instagram_data.csv."
    )
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 Social Media Performance Predictor")

st.caption(
    "An interactive dashboard for Instagram data, "
    "CTA predictions, trends, and model evaluation."
)

st.write(
    "Explore the dataset and test how different post "
    "characteristics are handled by the trained model."
)

st.divider()

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["✨ Predict", "📊 Trends", "🧠 Model Performance"]
)


# ==================================================
# TAB 1: PREDICTION
# ==================================================

def clear_prediction_inputs():
    st.session_state["account_type"] = account_options[0]
    st.session_state["follower_count"] = 5000
    st.session_state["media_type"] = media_options[0]
    st.session_state["content_category"] = category_options[0]
    st.session_state["traffic_source"] = traffic_options[0]
    st.session_state["posting_day"] = "Monday"
    st.session_state["posting_hour"] = 18


with tab1:
    st.subheader("✨ Predict CTA Presence")
    st.write("Enter your post details and click Predict CTA.")

    account_options = sorted(
        df["account_type"].dropna().astype(str).unique().tolist()
    )
    media_options = sorted(
        df["media_type"].dropna().astype(str).unique().tolist()
    )
    category_options = sorted(
        df["content_category"].dropna().astype(str).unique().tolist()
    )
    traffic_options = sorted(
        df["traffic_source"].dropna().astype(str).unique().tolist()
    )
    day_options = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]

    defaults = {
        "account_type": account_options[0],
        "follower_count": 5000,
        "media_type": media_options[0],
        "content_category": category_options[0],
        "traffic_source": traffic_options[0],
        "posting_day": "Monday",
        "posting_hour": 18,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Account Type",
            account_options,
            key="account_type"
        )

        st.number_input(
            "Follower Count",
            min_value=0,
            max_value=10000000,
            step=100,
            key="follower_count"
        )

        st.selectbox(
            "Media Type",
            media_options,
            key="media_type"
        )

        st.slider(
            "⏰ Posting Time",
            min_value=0,
            max_value=23,
            key="posting_hour",
            format="%d:00"
        )

    with col2:
        st.selectbox(
            "Content Category",
            category_options,
            key="content_category"
        )

        st.selectbox(
            "Traffic Source",
            traffic_options,
            key="traffic_source"
        )

        st.selectbox(
            "Posting Day",
            day_options,
            key="posting_day"
        )

    predict_col, clear_col = st.columns(2)

    with predict_col:
        predict_clicked = st.button(
            "🔮 Predict CTA",
            use_container_width=True
        )

    with clear_col:
        st.button(
            "🧹 Clear All",
            on_click=clear_prediction_inputs,
            use_container_width=True
        )

    if predict_clicked:
        input_data = pd.DataFrame([{
            "account_type": st.session_state["account_type"],
            "follower_count": st.session_state["follower_count"],
            "media_type": st.session_state["media_type"],
            "content_category": st.session_state["content_category"],
            "traffic_source": st.session_state["traffic_source"],
            "posting_day": st.session_state["posting_day"],
        }])

        st.markdown("### 🔮 Prediction Result")

        try:
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.success(
                    "✅ The model predicts that this post includes a CTA."
                )
            else:
                st.info(
                    "ℹ️ The model predicts that this post does not include a CTA."
                )

            st.caption(
                "This is a model estimate, not a guarantee of real-world performance. "
                "Posting time is displayed as an input but is not used by the "
                "current model."
            )

        except Exception as e:
            st.error("Prediction failed. Please check the model inputs.")
            st.caption(f"Technical details: {e}")
# ==================================================
# TAB 2: TRENDS
# ==================================================
with tab2:
    st.subheader("📊 Instagram Trends")
    st.write("Explore the available dataset.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Columns", len(df.columns))

    if "follower_count" in df.columns:
        col3.metric(
            "Average Followers",
            f"{df['follower_count'].mean():,.0f}"
        )

    with st.expander("View Dataset"):
        st.dataframe(df, use_container_width=True)

    if "has_call_to_action" in df.columns:
        st.markdown("### CTA Distribution")

        cta_counts = (
            df["has_call_to_action"]
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        fig, ax = plt.subplots()
        cta_counts.plot(kind="bar", ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No CTA", "CTA Present"], rotation=0)
        ax.set_title("Posts With and Without CTA")
        ax.set_xlabel("CTA Status")
        ax.set_ylabel("Number of Posts")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "follower_count" in df.columns:
        st.markdown("### Follower Count Distribution")

        fig, ax = plt.subplots()
        df["follower_count"].dropna().plot(
            kind="hist", bins=15, ax=ax
        )
        ax.set_title("Distribution of Follower Counts")
        ax.set_xlabel("Follower Count")
        ax.set_ylabel("Number of Posts")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "content_category" in df.columns:
        st.markdown("### Content Categories")

        fig, ax = plt.subplots()
        df["content_category"].value_counts().plot(
            kind="bar", ax=ax
        )
        ax.set_title("Posts by Content Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Number of Posts")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    if "media_type" in df.columns:
        st.markdown("### Media Type Distribution")

        fig, ax = plt.subplots()
        df["media_type"].value_counts().plot(
            kind="bar", ax=ax
        )
        ax.set_title("Media Type Distribution")
        ax.set_xlabel("Media Type")
        ax.set_ylabel("Number of Posts")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ==================================================
# TAB 3: MODEL PERFORMANCE
# ==================================================
with tab3:
    st.subheader("🧠 Model Performance")

    st.write(
        "Review the evaluation results supplied by the team."
    )

    comparison_path = "model_comparison.csv"

    if os.path.exists(comparison_path):
        try:
            comparison_df = pd.read_csv(comparison_path)
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )
        except Exception as e:
            st.error(f"Could not read comparison file: {e}")
    else:
        st.warning("model_comparison.csv is not available yet.")

    st.markdown("### Current Prediction Task")

    st.write(
        "The current saved model predicts CTA presence. "
        "It does not directly predict likes, comments, reach, "
        "or engagement rate."
    )

    st.markdown("### Input Features")

    st.write(
        "Account type, follower count, media type, "
        "content category, traffic source, and posting day."
    )

    st.caption(
        "The posting-hour slider is for reference only because "
        "posting hour is not an input to the current model."
    )

    importance_path = "feature_importance.png"

    if os.path.exists(importance_path):
        st.markdown("### Feature Importance")

        st.image(
            importance_path,
            caption=(
                "Feature importance from a separate Decision Tree model; "
                "not the Logistic Regression model's coefficients."
            ),
            use_container_width=True
        )
    else:
        st.info("feature_importance.png is not available yet.")

    st.warning(
        "The dataset is small and synthetic. Results may not "
        "generalise to real Instagram posts. The baseline model "
        "also achieved higher raw accuracy in the current evaluation."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Social Media Performance Predictor | Machine Learning Project"
)
