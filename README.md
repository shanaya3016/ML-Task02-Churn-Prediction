# ML-Task02-Churn-Prediction
# Task 02: Build and Compare Machine Learning Models

**Devixo Solutions: Machine Learning Task 2**

## Objective
Develop multiple machine learning models for a real-world prediction problem, evaluate them using appropriate metrics, and compare them to select the best one. The task covers preprocessing, feature engineering, model comparison and model selection.

## Dataset
**Telco Customer Churn** (Kaggle, by BlastChar)
- 7,043 customers and 21 columns
- Target: `Churn` (Yes/No). About 26.5% of customers churn, so the data is imbalanced.

## Technologies
Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn (run on Google Colab)

## Project Steps
1. **Data Preprocessing**
   - Loaded the dataset
   - Converted `TotalCharges` to numeric (11 blank values became missing and were filled with the median)
   - Removed duplicates (0 found)
   - Encoded categorical features (One-Hot Encoding)
   - Scaled numerical features (StandardScaler)
   - Split the data 80/20 (stratified)
2. **Feature Engineering:** created `AvgMonthlySpend` and `TenureGroup`.
3. **Model Development**
   - Classification: Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost
   - Regression (predicting `TotalCharges`): Linear Regression, Decision Tree, Random Forest, KNN, SVR, XGBoost
4. **Model Evaluation**
   - Classification: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
   - Regression: MAE, MSE, RMSE, R2 Score
5. **Bonus:** Hyperparameter tuning with GridSearchCV and RandomizedSearchCV.

## Results: Classification (default settings)
| Model | Accuracy | Precision | Recall | F1 | Training Time (s) |
|---|---|---|---|---|---|
| Logistic Regression | 0.7991 | 0.6532 | 0.5187 | 0.5782 | 0.14 |
| Random Forest | 0.7842 | 0.6144 | 0.5027 | 0.5529 | 1.86 |
| XGBoost | 0.7821 | 0.6091 | 0.5000 | 0.5492 | 1.20 |
| KNN | 0.7573 | 0.5421 | 0.5508 | 0.5464 | 0.15 |
| SVM | 0.7885 | 0.6429 | 0.4572 | 0.5344 | 4.85 |
| Decision Tree | 0.7339 | 0.4987 | 0.5134 | 0.5059 | 0.29 |

## Results: After Hyperparameter Tuning
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (GridSearchCV) | 0.7452 | 0.5131 | 0.7861 | 0.6209 |
| Random Forest (GridSearchCV) | 0.8006 | 0.6587 | 0.5160 | 0.5787 |
| XGBoost (RandomizedSearchCV) | 0.7970 | 0.6457 | 0.5214 | 0.5769 |

## Results: Regression (predicting TotalCharges)
| Model | MAE | MSE | RMSE | R2 |
|---|---|---|---|---|
| Random Forest | 54.30 | 6748.40 | 82.15 | 0.9987 |
| XGBoost | 61.58 | 8102.10 | 90.01 | 0.9984 |
| Decision Tree | 86.30 | 16051.28 | 126.69 | 0.9969 |
| SVM (SVR) | 109.42 | 18321.01 | 135.36 | 0.9965 |
| KNN | 268.99 | 171684.48 | 414.34 | 0.9668 |
| Linear Regression | 523.94 | 433085.75 | 658.09 | 0.9162 |

## Key Observations
- Churn is imbalanced (26.5%), so accuracy alone is misleading. Models were compared using Precision, Recall and F1 Score.
- `AvgMonthlySpend` and `MonthlyCharges` have a correlation of 1.00, so the engineered feature is largely redundant.
- Tuned Logistic Regression raised Recall from 0.5187 to 0.7861, but accuracy dropped from 0.7991 to 0.7452. This is a trade-off between catching more churners and raising more false alarms.
- Top churn drivers (Random Forest feature importance): tenure, total charges, monthly charges, month-to-month contract, no online security, no tech support, electronic check payment and fiber optic internet.

## Conclusion
- With default settings, Logistic Regression performed best and trained fastest.
- After tuning, Random Forest reached the highest accuracy (0.8006), while the **tuned Logistic Regression achieved the highest F1 (0.6209) and Recall (0.7861)**, catching about 79% of churning customers.
- **Recommended model: tuned Logistic Regression.** In churn prediction the main goal is to find customers who are about to leave, and this model is fast and interpretable. Its trade-off is lower accuracy and precision, meaning some loyal customers are flagged by mistake. If fewer false alarms are preferred, the tuned Random Forest is a good alternative.
- For regression, Random Forest was best (R2 = 0.9987). Linear Regression was the weakest because `TotalCharges` is roughly `tenure x MonthlyCharges`, a non-linear relationship that trees capture better.

## Files in this Repository
| File | Description |
|---|---|
| `Task02_ML_Models(1).ipynb` | Full notebook with code and outputs |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset |
| `model_comparison.csv` | Default classification models comparison |
| `tuned_models.csv` | Tuned models results |
| `regression_comparison.csv` | Regression models comparison |
| `final_classification_comparison.csv` | Combined classification comparison |
| `01_churn_distribution.png` | Churn distribution chart |
| `02_correlation_heatmap.png` | Correlation heatmap |
| `03_confusion_matrices.png` | Confusion matrices of all models |
| `04_model_comparison.png` | Model comparison bar chart |
| `05_feature_importance.png` | Top 10 feature importances |

## How to Run
1. Open the notebook in Google Colab.
2. Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the Colab files panel.
3. Run `!pip install xgboost -q`, then **Runtime > Run all**.
