> **Note:** This report documents an earlier experimental
> run using a 150/50 train-test split. Random Forest was
> not included in this experiment. The results presented
> here belong to the earlier experiment and should not be
> considered the final model comparison. Refer to the
> README.md and model_comparison.csv for the updated
> evaluation results.
# Social Media Performance Predictor
## Day 5 and Day 6 — Model Evaluation Report

### 1. Dataset
- Total records: 200
- Training records: 150
- Testing records: 50
- Input features: 7
- Target: has_call_to_action

### 2. Model Comparison

| Model | Accuracy |
|---|---:|
| Decision Tree | 44% |
| Logistic Regression | 48% |
| Baseline | 64% |

### 3. Detailed Evaluation
- Decision Tree balanced accuracy: 49%
- Decision Tree macro F1-score: 0.439
- Logistic Regression balanced accuracy: 46%
- Logistic Regression macro F1-score: 0.458
- Baseline balanced accuracy: 50%
- Baseline macro F1-score: 0.390

### 4. Feature Importance
The Decision Tree assigned its highest importance to:
1. Follower count
2. Music content category
3. Travel content category
4. Posting day

Feature importance describes how the fitted tree uses its input features. It does not establish causation.

### 5. Limitations
- The dataset contains only 200 synthetic records.
- Both trained models performed below the baseline in accuracy.
- The target represents CTA presence, not actual engagement.
- The dataset does not contain likes, comments, shares, or reach.

### 6. Conclusion
The initial models have not demonstrated reliable predictive performance.
Further testing with more representative data and relevant features is needed.
The results are preliminary and should not be interpreted as real-world Instagram performance predictions.
