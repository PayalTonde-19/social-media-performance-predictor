
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
with tab1:
    st.subheader("✨ Predict Call-to-Action Presence")

    st.write(
        "Enter the details below and click Predict to generate "
        "a new prediction."
    )

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            account_type = st.selectbox(
                "Account Type",
                account_options,
                format_func=lambda x: x.title()
            )

            follower_count = st.number_input(
                "Follower Count",
                min_value=0,
                max_value=10000000,
                value=8000,
                step=100
            )

            media_type = st.selectbox(
                "Media Type",
                media_options,
                format_func=lambda x: x.title()
            )

            content_category = st.selectbox(
                "Content Category",
                category_options
            )

        with col2:
            traffic_source = st.selectbox(
                "Traffic Source",
                traffic_options
            )

            posting_hour = st.slider(
                "Posting Hour (reference only)",
                min_value=0,
                max_value=23,
                value=18
            )

            posting_day = st.selectbox(
                "Posting Day",
                day_options
            )

        submitted = st.form_submit_button(
            "🔮 Predict CTA",
            use_container_width=True
        )

    # A new submission always replaces the previous saved result.
    if submitted:
        input_data = pd.DataFrame([{
            "account_type": account_type,
            "follower_count": int(follower_count),
            "media_type": media_type,
            "content_category": content_category,
            "traffic_source": traffic_source,
            "posting_day": posting_day,
        }], columns=MODEL_COLUMNS)

        try:
            prediction = model.predict(input_data)[0]

            probabilities = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0].tolist()

            st.session_state["latest_prediction"] = {
                "inputs": input_data.to_dict(orient="records")[0],
                "prediction": int(prediction),
                "probabilities": probabilities,
                "posting_hour": int(posting_hour),
            }
            st.session_state["prediction_error"] = None

        except Exception as e:
            st.session_state["prediction_error"] = str(e)
            st.session_state.pop("latest_prediction", None)

    # Display any current prediction, including after Streamlit reruns.
    if st.session_state.get("prediction_error"):
        st.error("Prediction failed.")
        st.code(st.session_state["prediction_error"])

    result = st.session_state.get("latest_prediction")

    if result is not None:
        st.markdown("---")
        st.markdown("### Prediction Result")

        if result["prediction"] == 1:
            st.success(
                "The model predicts that this post includes a "
                "Call to Action (CTA)."
            )
        else:
            st.info(
                "The model predicts that this post does not include "
                "a Call to Action (CTA)."
            )

        st.caption(
            "This is a machine learning prediction, not a guarantee "
            "of actual post performance."
        )

        st.markdown("#### Latest submitted inputs")
        st.dataframe(
            pd.DataFrame([result["inputs"]]),
            hide_index=True,
            use_container_width=True
        )

        if result["probabilities"] is not None:
            classes = list(model.classes_)
            probability_data = pd.DataFrame({
                "Predicted class": classes,
                "Probability": result["probabilities"],
            })
            probability_data["Probability"] = (
                probability_data["Probability"].map(
                    lambda p: f"{p:.1%}"
                )
            )

            st.markdown("#### Model class probabilities")
            st.dataframe(
                probability_data,
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "Posting hour is displayed for reference only; "
            "it is not used by the current trained model."
        )

    else:
        st.info("Enter your choices and click Predict CTA to begin.")

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
