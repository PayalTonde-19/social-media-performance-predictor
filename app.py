
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Social Media Performance Predictor",
    page_icon="📱",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed_instagram_data.csv"
MODEL_PATH = BASE_DIR / "cta_model.pkl"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"


# --------------------------------------------------
# 2. CUSTOM DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F2F8FF;
    }

    h1, h2, h3 {
        color: #344C70;
    }

    [data-testid="stSidebar"] {
        background-color: #FCE8F1;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E7D9EA;
        padding: 16px;
        border-radius: 12px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background-color: #E9A9CA;
        color: #352B43;
        border: none;
        border-radius: 10px;
        font-weight: bold;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background-color: #D991B9;
        color: white;
        border: none;
    }

    .subtitle {
        color: #65758B;
        font-size: 16px;
    }

    .welcome-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border-left: 5px solid #E9A9CA;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not DATA_PATH.exists():
    st.error(
        "Processed dataset not found. "
        "Please run: python data_prep.py"
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. "
        "Please run: python model.py"
    )
    st.stop()

try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Could not load the data or model: {error}")
    st.stop()


required_columns = [
    "account_type",
    "follower_count",
    "media_type",
    "content_category",
    "traffic_source",
    "posting_day",
    "has_call_to_action"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Required columns are missing: {missing_columns}")
    st.stop()


# --------------------------------------------------
# 4. HEADER
# --------------------------------------------------

st.title("📱 Social Media Performance Predictor")

st.markdown(
    '<p class="subtitle">'
    'An interactive dashboard to explore Instagram post data '
    'and predict whether a post includes a call to action.'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="welcome-card">
        <strong>Welcome to your dashboard! 💗</strong><br>
        Explore the dataset, compare models, and test a post
        using the CTA prediction model.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Predict CTA",
        "Model Comparison",
        "About the Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Social Media Performance Predictor")
st.sidebar.caption("Machine Learning Project")


# --------------------------------------------------
# 6. DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("📊 Dashboard Overview")

    total_posts = len(df)
    unique_accounts = (
        df["account_type"].nunique()
    )

    cta_values = pd.to_numeric(
        df["has_call_to_action"],
        errors="coerce"
    )

    cta_count = int((cta_values == 1).sum())
    no_cta_count = int((cta_values == 0).sum())

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Posts", f"{total_posts:,}")
    c2.metric("Account Types", unique_accounts)
    c3.metric("Posts With CTA", f"{cta_count:,}")
    c4.metric("Posts Without CTA", f"{no_cta_count:,}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("CTA Distribution")

        cta_chart = pd.DataFrame({
            "CTA Status": [
                "CTA Present",
                "CTA Absent"
            ],
            "Number of Posts": [
                cta_count,
                no_cta_count
            ]
        })

        st.bar_chart(
            cta_chart.set_index("CTA Status")
        )

    with right:
        st.subheader("Posts by Media Type")

        media_counts = (
            df["media_type"]
            .value_counts()
            .rename_axis("Media Type")
            .to_frame("Number of Posts")
        )

        st.bar_chart(media_counts)

    st.markdown("---")

    left2, right2 = st.columns(2)

    with left2:
        st.subheader("Average Followers by Account Type")

        followers_by_account = (
            df.groupby("account_type")["follower_count"]
            .mean()
            .sort_values(ascending=False)
        )

        st.bar_chart(followers_by_account)

    with right2:
        st.subheader("Posts by Day of the Week")

        day_order = [
            "Monday", "Tuesday", "Wednesday",
            "Thursday", "Friday", "Saturday", "Sunday"
        ]

        day_counts = (
            df["posting_day"]
            .value_counts()
            .reindex(day_order, fill_value=0)
        )

        st.bar_chart(day_counts)

    st.markdown("---")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "The dataset contains synthetic sample data. "
        "These charts describe the supplied dataset, not all Instagram posts."
    )


# --------------------------------------------------
# 7. CTA PREDICTION
# --------------------------------------------------

elif page == "Predict CTA":

    st.header("💗 Predict CTA Presence")

    st.write(
        "Enter the post details below. The trained model will "
        "predict whether the post includes a call to action."
    )

    # Use categories actually present in the training dataset.
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

    day_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    available_days = (
        df["posting_day"].dropna().astype(str).unique().tolist()
    )

    day_options = [
        day for day in day_order if day in available_days
    ]

    with st.form("prediction_form"):

        st.subheader("Account and Audience")

        col1, col2 = st.columns(2)

        with col1:
            account_type = st.selectbox(
                "Account type",
                account_options
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
                traffic_options
            )

            media_type = st.selectbox(
                "Media type",
                media_options
            )

        st.subheader("Post Details")

        col3, col4 = st.columns(2)

        with col3:
            content_category = st.selectbox(
                "Content category",
                category_options
            )

        with col4:
            posting_day = st.selectbox(
                "Day of the week",
                day_options
            )

        submitted = st.form_submit_button(
            "💗 Predict CTA Presence",
            use_container_width=True
        )

    if submitted:

        # These columns must match the model's training inputs.
        input_data = pd.DataFrame([{
            "account_type": account_type,
            "follower_count": follower_count,
            "media_type": media_type,
            "content_category": content_category,
            "traffic_source": traffic_source,
            "posting_day": posting_day
        }])

        try:
            with st.spinner("Generating your prediction..."):
                prediction = int(model.predict(input_data)[0])

            st.markdown("---")
            st.subheader("💗 Prediction Result")

            if prediction == 1:
                st.success(
                    "The model predicts that this post "
                    "will have a call to action (CTA)."
                )
            else:
                st.info(
                    "The model predicts that this post "
                    "will not have a call to action (CTA)."
                )

            # Show model probability when available.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                classes = list(model.classes_)

                if prediction in classes:
                    probability = float(
                        probabilities[classes.index(prediction)]
                    )

                    st.metric(
                        "Model confidence for this prediction",
                        f"{probability:.1%}"
                    )

            with st.expander("View the input provided to the model"):
                st.dataframe(input_data, use_container_width=True)

            st.caption(
                "This is a model prediction, not a guarantee "
                "of actual Instagram post performance."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")
            st.info(
                "Check that model.py was run successfully and "
                "that the model uses the same input columns."
            )


# --------------------------------------------------
# 8. MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.header("📈 Model Comparison")

    st.write(
        "Compare the evaluated models using the results "
        "generated by model.py."
    )

    if COMPARISON_PATH.exists():

        try:
            comparison_df = pd.read_csv(COMPARISON_PATH)

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

            if (
                "Model" in comparison_df.columns
                and "Macro F1" in comparison_df.columns
            ):
                chart_data = comparison_df.set_index("Model")[
                    ["Macro F1"]
                ]

                st.subheader("Macro F1-score Comparison")
                st.bar_chart(chart_data)

            if (
                "Model" in comparison_df.columns
                and "Accuracy" in comparison_df.columns
            ):
                accuracy_data = comparison_df.set_index("Model")[
                    ["Accuracy"]
                ]

                st.subheader("Accuracy Comparison")
                st.bar_chart(accuracy_data)

        except Exception as error:
            st.error(f"Could not read the model comparison: {error}")

    else:
        st.warning(
            "Model comparison results are not available yet. "
            "Run `python model.py` in your project terminal first."
        )

    results_path = BASE_DIR / "model_results.txt"

    if results_path.exists():
        with st.expander("View detailed evaluation report"):
            st.text(results_path.read_text(encoding="utf-8"))


# --------------------------------------------------
# 9. ABOUT THE PROJECT
# --------------------------------------------------

elif page == "About the Project":

    st.header("ℹ️ About the Project")

    st.markdown(
        """
        ### Project Objective

        The Social Media Performance Predictor demonstrates how
        machine learning can analyse Instagram post characteristics
        and predict whether a post contains a call to action.

        ### Dataset

        - **Records:** 200 synthetic Instagram posts
        - **Target variable:** `has_call_to_action`
        - **Target value 1:** CTA present
        - **Target value 0:** CTA absent

        ### Input Features

        - Account type
        - Follower count
        - Media type
        - Content category
        - Traffic source
        - Day of the week

        ### Machine Learning Models

        - Decision Tree Classifier
        - Logistic Regression
        - Majority-class baseline for comparison

        ### Tools and Technologies

        - Python
        - Pandas
        - Scikit-learn
        - Joblib
        - Streamlit

        ### Important Limitation

        The dataset is synthetic and contains only 200 records.
        The model demonstrates a machine learning workflow but
        has not been established as a reliable predictor of
        real-world Instagram engagement or success.
        """
    )

st.markdown("---")
st.caption("Social Media Performance Predictor | Academic Project")
