
# Project Overview

This project focuses on predicting whether a bank client will subscribe to a term deposit based on their demographics, financial situation, and past campaign interactions.

The dataset contains 40,000 records with 14 features (5 numeric, 9 categorical) and a target variable y (yes/no). The dataset is imbalanced, with only ~7–8% of clients subscribing (y = yes).

Objectives:

Explore and understand the dataset (EDA).

Preprocess and encode categorical variables.

Build a predictive model.

Optimize model performance using hyperparameter tuning (Randomized Search).

Analyze feature importance.

Extract insights through customer segmentation.

# Step 1: Exploratory Data Analysis (EDA)

## Key observations from EDA:

Numeric features:
Age: Mean ~40, range 19–95 

Balance: Skewed, many negative values; max 102,127

Campaign: Avg ~2.88 calls per client

Duration: Ranges 0–4918 sec, median 175 sec

Categorical features:
job, marital, education, housing, loan, contact, month

Target variable imbalance:
Only ~7–8% of clients subscribed (y = yes)


Visualizations & Segmentation Insights (EDA):

Job: Students and retired clients have higher subscription rates.

Education: Tertiary-educated clients respond better.

Month: Campaigns in October, March, and April are most successful.

Balance: Clients with higher balances tend to subscribe more.

Housing/Loan: Clients without housing loans are slightly more responsive.

# Step 2: Feature Selection & Preprocessing

Duration is highly correlated with target (y) because the longer the call, the higher the chance of subscription.
In a real-world predictive scenario, duration is unknown before the call starts, so using it would be data leakage. Dropped duration to make predictions realistic and deployable.
Contact is also dropped because of the unknown values.

# Step 3: Baseline Model Training

Used XGBoost/Gradient Boosting as the baseline model.

Baseline Cross-Validation Results:
Accuracy: 75.8%
Recall: 0.49
F1-score: 0.23
ROC-AUC: 0.69

Test Accuracy: 77%
Confirms model predicts majority class well but struggles with positive class (y = yes).

# Step 4: Hyperparameter Tuning with Randomized Search
CV Accuracy improved to ~81.6%

# Step 5: Test Set Evaluation (Final Model)
Test Accuracy: ~81% ✅
Improved recall for positive class
ROC-AUC ~0.73–0.74
Randomized Search successfully improved both training CV and test accuracy.

# Step 6: Feature Importance

Top Features:
Campaign timing (month) and contact method are the most critical.
Marital status, housing, and numeric features (balance) also influence subscriptions.

# Step 7: Customer Segmentation Insights

Job: Students (15.6%) and retired clients (10.5%) respond more.

Education: Tertiary-educated clients have higher subscription rates.

Month: October, March, April are best months for campaigns.

Balance: Higher balance → higher subscription probability.

Housing/Loan: Clients without housing loans are slightly more likely to subscribe.

These insights can guide targeted campaigns for better ROI.

# Step 8: Key Takeaways
Randomized Search improved model accuracy from 76% → 81% and made the model more reliable on unseen data.

Dropping duration prevented data leakage and made predictions realistic.

Feature importance shows that month of campaign, contact type, and marital status are key drivers.

Segmentation analysis helps target high-probability clients, e.g., students, retired, high-balance clients in October and March.

To maximize campaign effectiveness, focus on targeting high-probability clients based on these features, particularly Month, Contact, Job, and Balance, while tailoring messaging for the audience.
