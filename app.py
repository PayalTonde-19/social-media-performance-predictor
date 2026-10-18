
import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Social Media Performance Predictor",
    page_icon="🩵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cta_model.pkl"
DATA_PATH = BASE_DIR / "data" / "Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv"
COMPARISON_PATH = BASE_DIR / "model_comparison.csv"
FEATURE_CHART_PATH = BASE_DIR / "feature_importance.png"


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load cta_model.pkl: {e}")
    st.stop()


# ---------------- STYLING ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #eafaff, #fdf0f7, #eaf5ff);
}

.block-container {
    max-width: 1150px;
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(120deg, #8edcf5, #b8eaff, #f5b9d7);
    padding: 30px;
    border-radius: 22px;
    color: #24445e;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.8);
}

.hero h1 {
    color: #24445e;
    font-size: clamp(1.6rem, 4vw, 2.35rem);
    font-weight: 800;
    line-height: 1.25;
}

.hero p {
    color: #395c73;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 12px;
    background: linear-gradient(100deg, #78cfee, #e99ac5);
    color: white;
    border: 1px solid #f3c3dd;
    font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 16px;
}

.footer {
    text-align: center;
    color: #8194a5;
    font-size: 0.82rem;
    padding: 20px 0 5px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HERO HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Social Media Performance Predictor 🩵🩷</h1>
    <p>Explore Instagram post data, generate a CTA prediction,
    and review analytics and model performance.</p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "A student project demonstrating machine learning and Instagram analytics."
)


# ---------------- THREE TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "✨ Predict",
    "📊 Trends",
    "🧠 Model Performance"
])


# ==================================================
# TAB 1: PREDICT
# ==================================================
with tab1:
    st.subheader("💗 Predict CTA Presence")
    st.write(
        "Enter the post details to predict whether the post "
        "is likely to contain a call-to-action (CTA)."
    )

    with st.form("prediction_form"):
        st.markdown("#### 🩵 Account & Audience")

        col1, col2 = st.columns(2)

        with col1:
            account_type = st.selectbox(
                "Account type",
                ["Creator", "Brand", "Personal", "Business"]
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

        st.markdown("#### 🩷 Post Details")

        col3, col4 = st.columns(2)

        with col3:
            content_category = st.selectbox(
                "Content category",
                [
                    "Education", "Entertainment", "Lifestyle",
                    "Fashion", "Food", "Travel", "Technology",
                    "Music", "Comedy", "Fitness", "Beauty",
                    "Photography", "Other"
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
            day_names = [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ]

            posting_day = st.selectbox(
                "Day of the week",
                options=list(range(7)),
                format_func=lambda day: day_names[day]
            )

        submitted = st.form_submit_button(
            "🩵 Predict CTA Presence",
            use_container_width=True
        )

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
            with st.spinner("Generating your prediction..."):
                prediction = model.predict(input_data)[0]

            st.markdown("### 💗 Prediction Result")

            if prediction == 1:
                st.success(
                    "The model predicts that this post is likely "
                    "to have a call-to-action (CTA)."
                )
            else:
                st.info(
                    "The model predicts that this post is likely "
                    "not to have a call-to-action (CTA)."
                )

            st.caption(
                "This predicts CTA presence only, not likes, reach, "
                "views, or engagement."
            )

            with st.expander("View entered details"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ==================================================
# TAB 2: TRENDS
# ==================================================
with tab2:
    st.subheader("📊 Instagram Analytics Dashboard")
    st.write("Explore the available synthetic Instagram dataset.")

    try:
        analytics_df = pd.read_csv(DATA_PATH)

        col1, col2 = st.columns(2)

        with col1:
            account_options = ["All"] + sorted(
                analytics_df["account_type"]
                .dropna().astype(str).unique().tolist()
            )
            selected_account = st.selectbox(
                "Filter by Account Type",
                account_options
            )

        with col2:
            media_options = ["All"] + sorted(
                analytics_df["media_type"]
                .dropna().astype(str).unique().tolist()
            )
            selected_media = st.selectbox(
                "Filter by Media Type",
                media_options
            )

        filtered_df = analytics_df.copy()

        if selected_account != "All":
            filtered_df = filtered_df[
                filtered_df["account_type"].astype(str) == selected_account
            ]

        if selected_media != "All":
            filtered_df = filtered_df[
                filtered_df["media_type"].astype(str) == selected_media
            ]

        st.markdown("### Dataset Overview")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Posts", len(filtered_df))

        if not filtered_df.empty:
            m2.metric(
                "Average Followers",
                f"{filtered_df['follower_count'].mean():,.0f}"
            )

            if "has_call_to_action" in filtered_df.columns:
                cta_values = (
                    filtered_df["has_call_to_action"]
                    .astype(str).str.strip().str.lower()
                )
                cta_count = cta_values.isin(["1", "true", "yes"]).sum()
                m3.metric("Posts with CTA", int(cta_count))
            else:
                m3.metric(
                    "Account Types",
                    filtered_df["account_type"].nunique()
                )

        if filtered_df.empty:
            st.warning("No posts match these filters.")
        else:
            st.markdown("### Visual Insights")

            chart1, chart2 = st.columns(2)

            with chart1:
                st.markdown("**Posts by Media Type**")
                fig, ax = plt.subplots(figsize=(6, 4))
                filtered_df["media_type"].value_counts().plot(
                    kind="bar", ax=ax
                )
                ax.set_xlabel("Media Type")
                ax.set_ylabel("Number of Posts")
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with chart2:
                st.markdown("**Posts by Content Category**")
                fig, ax = plt.subplots(figsize=(6, 4))
                filtered_df["content_category"].value_counts().plot(
                    kind="bar", ax=ax
                )
                ax.set_xlabel("Content Category")
                ax.set_ylabel("Number of Posts")
                ax.tick_params(axis="x", rotation=40)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            chart3, chart4 = st.columns(2)

            with chart3:
                st.markdown("**Posts by Account Type**")
                fig, ax = plt.subplots(figsize=(6, 4))
                filtered_df["account_type"].value_counts().plot(
                    kind="pie", autopct="%1.1f%%", ax=ax
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
        st.error("Dataset not found. Check the CSV inside your data folder.")
    except Exception as e:
        st.error(f"Dashboard error: {e}")


# ==================================================
# TAB 3: MODEL PERFORMANCE
# ==================================================

       
with tab3:
    st.subheader("🧠 Model Performance")

    st.info(
        "This section presents model evaluation results and the "
        "features used by the prediction model. Some team deliverables "
        "are still pending."
    )

    st.markdown("### Current prediction model")
    st.write(
        "The current app predicts whether an Instagram post has a "
        "call to action (CTA). It does not directly predict likes, "
        "comments, shares, reach, or engagement rate."
    )

    st.markdown("### Model comparison")
    comparison_path = "model_comparison.csv"

    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        st.dataframe(comparison_df, use_container_width=True)
    else:
        st.warning(
            "Model comparison is pending. This table will appear "
            "when Person 1 provides the verified model_comparison.csv."
        )

    st.markdown("### Feature importance")
    importance_path = "feature_importance.png"

    if os.path.exists(importance_path):
        st.image(
            importance_path,
            caption="Feature-importance chart supplied by the team",
            use_container_width=True
        )
    else:
        st.warning(
            "The feature-importance chart is pending. No results "
            "have been added or estimated."
        )

    st.caption(
        "Note: Model metrics must be interpreted according to their "
        "respective prediction targets and evaluation methods."
    )


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="footer">
    Made with 🩵 & 🩷 using Python and Streamlit<br>
    Social Media Performance Predictor • Student Project
</div>
""", unsafe_allow_html=True)
