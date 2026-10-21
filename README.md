# 📊 Social Media Performance Predictor

An interactive machine learning project that explores Instagram post data and predicts whether a post includes a Call to Action (CTA).

Developed as part of **E.D.G.E. 2026**.

## 🚀 Project Overview

The Social Media Performance Predictor allows users to enter post details and receive a machine learning prediction of CTA presence. It also provides visualisations of the dataset and model evaluation results through an interactive Streamlit dashboard.

## ✨ Features

- **CTA Prediction:** Predicts whether a post includes a call to action.
- **Interactive Dashboard:** User-friendly interface with a pastel-blue and pink theme.
- **Data Visualisation:** Explore CTA distribution, follower counts, content categories, and media types.
- **Model Comparison:** Review evaluation metrics for Logistic Regression, Decision Tree, and a baseline classifier.
- **Feature Importance:** View feature importance from a separate Decision Tree model.
- **Input Validation:** Test different account types, media types, content categories, traffic sources, follower counts, and posting days.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib

## 📂 Project Structure

```text
Social_Media_Performance_Predictor/
├── app.py
├── cta_model.pkl
├── data_prep.py
├── model.py
├── cta_model.pkl
├── model_comparison.csv
├── feature_importance.png
├── requirements.txt
├── README.md
├── data/
│   ├── Instagram_Analytics_200_CLEANED_2026_SYNTHETIC.csv
│   └── processed_instagram_data.csv
└── outputs/
```

*Note: Some files or folders may differ depending on the current project version.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/PayalTonde-19/social-media-performance-predictor.git
```

### 2. Open the project folder

```bash
cd social-media-performance-predictor
```

### 3. Create and activate a virtual environment (optional)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The dashboard will open in your browser, usually at `http://localhost:8501`.

## 🤖 Machine Learning

The current application uses a saved classification pipeline to predict CTA presence.

The model input features are:

- Account type
- Follower count
- Media type
- Content category
- Traffic source
- Posting day

Posting hour is displayed as a reference input in the interface but is not used by the current model.

## 📈 Evaluation

The project compares Logistic Regression and Decision Tree classifiers with a baseline classifier using accuracy, balanced accuracy, and Macro F1.

In the current evaluation, Logistic Regression achieved 48% accuracy, while the baseline achieved 64% accuracy. These results indicate that the model requires further improvement.

## ⚠️ Limitations

- The dataset contains 200 synthetic records.
- Evaluation results may not generalise to real Instagram posts.
- The model predicts CTA presence, not likes, comments, reach, or engagement rate.
- Posting-hour effects cannot be evaluated with the current date-only timestamp data.
- Feature importance from the separate Decision Tree should not be interpreted as causal evidence or as Logistic Regression coefficients.

## 🔮 Future Improvements

- Use a larger dataset based on real-world observations.
- Improve model performance through further feature engineering and evaluation.
- Evaluate engagement-related targets if suitable labelled data becomes available.
- Improve model validation and test generalisation on unseen data.

## 👩‍💻 Project Context

Created as a student machine learning project for **E.D.G.E. 2026**.

## 📄 License

This repository is an educational project. Add a formal license if you intend to distribute or reuse the code publicly.
