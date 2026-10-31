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

## Model Evaluation

We evaluated multiple machine learning models to predict
Instagram social media performance.

### Model Comparison

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Random Forest | 54.5% | 53.4% | 52.8% |
| Logistic Regression | 48.5% | 48.5% | 47.3% |
| Decision Tree | 44.5% | 44.6% | 41.5% |
| Baseline | 63.5% | 50.0% | 38.8% |

### Best Model

Random Forest achieved the highest Macro F1 score of 52.8%
among the evaluated machine learning models.

Although the baseline achieved higher accuracy (63.5%),
Random Forest provided a better balance across classes,
with a balanced accuracy of 53.4%.

Therefore, Random Forest performed better than the other
trained models in terms of Macro F1 score.

### Conclusion

The model comparison highlights the importance of
evaluating classification models using multiple metrics
rather than accuracy alone.

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
