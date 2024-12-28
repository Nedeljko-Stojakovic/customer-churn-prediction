# Telecom Customer Churn Prediction Project Documentation

## 1. Project Overview

This project aims to analyze customer data from a telecommunication provider to predict customer churn and understand the factors influencing it. I use machine learning techniques to segment customers and build a predictive model for churn.

## 2. Data Description

The dataset contains customer information including:
- Usage statistics (calls, data)
- Customer tenure
- Billing information
- Customer segment

Key variables:
- SUBSCRIBER_ID: Unique identifier for each customer
- CHURN: Target variable (1 for churn, 0 for non-churn)
- Segment: Customer segment category
- Various usage and billing metrics (e.g., BNUM_OUT, REV_OUT, USAGE_OUT_ONNET_DUR)

## 3. Methodology

### 3.1 Data Loading and Initial Exploration
- Implemented in `data_loader.py`
- Loads data from Excel files
- Performs initial data type checks and unique value counts

### 3.2 Exploratory Data Analysis (EDA)
- Implemented in `eda.py`
- Generates summary statistics
- Creates visualizations:
  - Correlation heatmap
  - Distribution plots for numerical features
  - Segment distribution
  - Churn rate by segment

### 3.3 Feature Engineering
- Implemented in `feature_engineering.py`
- Creates 'USAGE_RATIO' feature
- Performs one-hot encoding on 'Segment' column
- Handles missing values by filling with mean

### 3.4 Model Development
- Implemented in `model.py`
- Uses XGBoost classifier
- Implements Optuna for hyperparameter optimization
- Splits data into train, validation, and test sets
- Evaluates model using AUC-ROC and classification report

### 3.5 Result Visualization
- Implemented in `visualization.py`
- Creates feature importance plot using XGBoost's built-in method
- Generates SHAP summary plot for model interpretability

## 4. Results

After running the model, I obtained the following results:

### 4.1 Data Insights
- The dataset contains 5000 customer records with 14 original features.
- Customer segments are divided into 4 categories: Segment 1, 2, 3, and 5.
- Several features had missing values, which were addressed during feature engineering.

### 4.2 Feature Engineering
- Created a new feature 'USAGE_RATIO' to capture the relationship between on-net and off-net usage.
- Performed one-hot encoding on the 'Segment' column, resulting in 4 new boolean features.
- Filled missing values, ensuring all features have 5000 non-null values in the final dataset.

### 4.3 Model Performance
- The XGBoost model achieved an exceptional Test AUC of 0.9998, indicating near-perfect discrimination between churners and non-churners.
- Classification Report:
  - Precision: 1.00 for non-churn, 1.00 for churn
  - Recall: 1.00 for non-churn, 0.99 for churn
  - F1-score: 1.00 for both classes
  - Overall accuracy: 1.00

### 4.4 Optimal Hyperparameters
The best hyperparameters found by Optuna:
- max_depth: 5
- learning_rate: 0.0768
- min_child_weight: 1
- subsample: 0.9792
- colsample_bytree: 0.8055

These results indicate an extremely high-performing model, which may suggest the need for careful interpretation and validation to ensure it's not overfitting.

### 4.5 Feature Importance

Based on the feature importance plot (MDI) and SHAP summary plot:

1. Customer Segmentation is the most crucial factor in predicting churn:
   - Segment 1 is the most important predictor, followed by Segments 3, 5, and 2.
   - This suggests that churn behavior varies significantly across different customer segments.

2. Usage-related features are the next most important:
   - BNUM_OUT (number of outgoing calls to different numbers) is a key predictor.
   - USAGE_RATIO (the ratio of on-net to off-net usage) also plays a significant role.

3. Customer tenure (LNE_TENURE) is moderately important, indicating that the length of the customer relationship influences churn probability.

4. Revenue-related features (REV_OUT) have some importance, but less than usage patterns and segmentation.

5. Interestingly, some features like REV_BUN_MAC (revenue from bundles) and USAGE_OUT_INT_DUR (international call duration) have minimal impact on the model's predictions.

### 4.6 Correlation Analysis

The correlation heatmap reveals:

1. Strong positive correlation (0.88) between REV_OUT (total revenue) and TOPUP_AMT (top-up amount), suggesting that customers who spend more also tend to top up more frequently.

2. High correlation (0.73) between BNUM_OUT and BNUM_IN, indicating that customers who make more outgoing calls also receive more incoming calls.

3. Moderate negative correlation (-0.74) between Segment and CHURN, implying that certain segments are more prone to churn than others.

4. Usage metrics (USAGE_OUT_OFFNET_DUR, USAGE_OUT_ONNET_DUR) show moderate correlation with revenue metrics (REV_OUT, TOPUP_AMT), as expected.

### 4.7 Churn Rate by Segment

The churn rate by segment plot shows:

1. Segment 1 has an extremely high churn rate, close to 100%.
2. Other segments (2, 3, and 5) have very low or zero churn rates.
3. This extreme difference in churn rates between segments explains why the segmentation features are the most important predictors in the model.

## 5. Conclusions

1. Customer Segmentation is the primary driver of churn in this dataset. Segment 1 customers are at the highest risk of churning, while other segments show very low or zero churn rates.

2. Usage patterns, particularly the number of unique numbers called (BNUM_OUT) and the ratio of on-net to off-net usage (USAGE_RATIO), are strong indicators of churn risk.

3. Customer tenure has a moderate impact on churn probability, suggesting that both new and long-term customers may be at risk of churning.

4. Revenue-related features, while important, are not the primary drivers of churn in this model. This suggests that customer behavior and segmentation are more predictive of churn than spending patterns alone.

5. The extremely high model performance (AUC of 0.9998) coupled with the stark difference in churn rates between segments suggests that the segmentation itself might be partly based on churn likelihood or closely related factors.

6. The high correlation between certain usage and revenue metrics provides opportunities for creating more complex features that might capture customer behavior more comprehensively.

7. The model demonstrates exceptional predictive power for customer churn, with near-perfect accuracy across all metrics.

8. The shallow tree depth (5) in the optimal model suggests that the churn patterns can be captured with relatively simple decision rules.

9. Given the extremely high accuracy, it's crucial to verify that this performance translates to new, unseen data and isn't a result of overfitting.

## 6. Improvements

While the current implementation provides a solid foundation for churn prediction, several improvements could enhance the model's performance and the depth of insights:

1. Advanced Feature Engineering:
   - Create interaction terms between relevant features
   - Develop time-based features if historical data is available
   - Implement domain-specific features based on telecom industry knowledge

2. Model Comparison:
   - Implement other algorithms such as LightGBM, Random Forest, or Logistic Regression
   - Create an ensemble model combining predictions from multiple algorithms

3. Cross-Validation:
   - Implement k-fold cross-validation for more robust model evaluation
   - Consider time-based cross-validation if the data has a strong temporal component

4. Handling Class Imbalance:
   - If churn is a rare event, explore techniques like SMOTE or class weighting
   - Evaluate model performance using metrics suitable for imbalanced data (e.g., F1 score, precision-recall AUC)

5. Feature Selection:
   - Implement a feature selection step to identify the most predictive variables
   - Consider using techniques like Recursive Feature Elimination or Lasso regularization

6. Hyperparameter Tuning:
   - Expand the hyperparameter search space in Optuna
   - Experiment with different optimization metrics (e.g., AUC-ROC instead of log loss)

7. Model Interpretability:
   - Implement LIME (Local Interpretable Model-agnostic Explanations) for individual prediction explanations
   - Create partial dependence plots to visualize the relationship between features and churn probability


These improvements would enhance the model's performance, provide deeper insights, and more closely align the analysis with business objectives. They represent potential discussion points for expanding on the current implementation.

## 7. Recommendations

1. Conduct a deep-dive analysis into the characteristics of Segment 1 to understand why these customers are so prone to churning. Develop targeted retention strategies for this high-risk segment.

2. Investigate the factors that make Segments 2, 3, and 5 more stable. Consider strategies to move customers from Segment 1 into these more stable segments.

3. Monitor changes in usage patterns, especially BNUM_OUT and USAGE_RATIO, as early warning signs of potential churn.

4. Develop personalized retention strategies based on usage patterns and customer tenure, rather than focusing solely on revenue-related metrics.

5. Consider refining the customer segmentation model, as the current segmentation appears to be highly predictive of churn on its own.

6. Investigate potential overfitting in the model, given its extremely high performance. Consider collecting more diverse data or using cross-validation techniques to ensure the model generalizes well to new data.

7. Explore the creation of more complex features that combine usage and revenue metrics to capture more nuanced customer behaviors.

## 8. How to Run the Project

1. Ensure you have Python 3.11 installed
2. Install required packages: `pip install -r requirements.txt`
3. Run the main script: `python main.py`

## 9. Project Structure

```
project_root/ 
│ ├── data/ 
│ ├── dataset.xlsx 
│ └── features_description.xlsx 
│ ├── src/ 
│ ├── data_loader.py 
│ ├── eda.py 
│ ├── feature_engineering.py 
│ ├── model.py 
│ └── visualization.py 
│ ├── outputs/ 
│ └── (Generated plots and results) 
│ ├── main.py 
├── requirements.txt 
└── README.md
```
