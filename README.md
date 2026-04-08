## Banking Marketing Response Prediction

**Machine Learning Model for Targeted Term Deposit Campaigns**

**Python | Pandas | Scikit-learn | XGBoost | Jupyter Notebook**

## Table of Contents
Executive Summary
Project Background
Problem Statement
Dataset Overview
Project Workflow
Technologies & Tools
Model Performance
Feature Importance
Customer Segmentation Insights
Business Impact
How to Reproduce
Conclusion
For Hiring Managers & Recruiters

## Executive Summary

This project develops an **end-to-end machine learning mode**l to predict whether a customer will subscribe to a **term deposit** during a banking marketing campaign.

The dataset presents a highly **imbalanced classification problem**, with only **~7–8%** of customers subscribing. The model pipeline includes **Exploratory Data Analysis (EDA), preprocessing, feature selection, model training, hyperparameter tuning using Randomized Search, evaluation, and business interpretation.**

To ensure real-world usability, features such as **duration** and **contact** were removed to prevent **data leakage** and reduce noise caused by missing values.

The final optimized model achieved:

- **~81%** Test Accuracy
- Improved predictive reliability
- Actionable insights for targeted marketing campaigns

## Project Background

Banks frequently conduct telemarketing campaigns to promote term deposits. However, most customers do not subscribe, resulting in:

- High operational costs
- Low conversion rates
- Inefficient resource allocation

This project demonstrates how **machine learning-driven targeting**can improve campaign efficiency and customer engagement.

## Problem Statement

**"Which customers are most likely to subscribe to a term deposit?"**

Key challenges addressed:

- Severe **class imbalance**
- Risk of **data leakage**
- Need for **interpretable features**
- Requirement for **business-relevant insights**
## Dataset Overview
- **Total Records:** 40,000
- **Total Features:** 14
- **Target Variable:**
   - y = yes → Subscribed
   - y = no → Not subscribed
## Feature Categories
**Demographic Features**
- age
- job
- marital
- education

**Financial Features**
- default
- balance
- housing
- loan

**Campaign Features**
- day
- month
- campaign

**Dropped Features**
- duration: Dropped due to data leakage risk.
- contact: Dropped due to a large proportion of "unknown" values
  
**Target Distribution**
- **No Subscription:** ~92%
- **Subscription:** ~8%

This confirms a **class imbalance problem.**

## Project Workflow

The project follows a structured **machine learning lifecycle.**

## Step 1: Exploratory Data Analysis (EDA)

Performed:

- Summary statistics
- Distribution analysis
- Target relationship analysis
- Segmentation-based exploration

**Key EDA Findings**
**Numeric Features**

| Feature  | Insight                    |
| -------- | -------------------------- |
| Age      | Mean ≈ 40 years            |
| Balance  | Highly skewed distribution |
| Campaign | Avg ≈ 2.88 contacts        |

**Target Behavior Insights**
- **Students and retired clients** showed higher response rates.
- **Tertiary education** correlated with higher subscriptions.
- **October, March, April** had higher campaign success.
- **Higher balance customers** responded more.
- Clients **without housing loans** showed slightly better response.

These findings guided later modeling decisions.

## Step 2: Feature Selection & Preprocessing

Performed:

- Dropped duration and contact
- One-Hot Encoding for categorical variables
- Train-Test Split (80/20)
- Stratified sampling
- Class imbalance handling
## Why Dropping duration Was Critical

This prevented:

# Data Leakage

Data leakage occurs when:

The model uses information not available during prediction.

Using duration would produce:

- Artificially high accuracy
- Poor real-world performance

Dropping it ensures:

- Realistic predictions
- Model reliability
- Production readiness

## Step 3: Baseline Model Training

Initial model performance evaluated using:

- Stratified 5-Fold Cross Validation
Baseline Model Results

| Metric   | Value |
| -------- | ----- |
| Accuracy | 75.8% |
| Recall   | 0.49  |
| F1 Score | 0.23  |
| ROC-AUC  | 0.69  |

## Test Performance (Baseline)
- **Test Accuracy:** ~77%

Observation:

Model predicted majority class well but needed improvement for minority class detection.

## Step 4: Hyperparameter Optimization (Randomized Search)

Used:

**RandomizedSearchCV**

Reason:

Faster than Grid Search
Efficient for large parameter spaces
Improves model performance

**Hyperparameters Tuned**
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

**Result After Optimization**

- **Cross-Validation Accuracy:** ~81.6%

Improved model generalization.

## Step 5: Final Model Evaluation

Evaluated on unseen test dataset.

**Final Performance**
| Metric          | Value      |
| --------------- | ---------- |
| Test Accuracy   | **~81%**   |
| ROC-AUC         | ~0.73–0.74 |
| Recall          | Improved   |
| Model Stability | High       |


This confirms:

- Successful model optimization
- Reliable predictive performance

## Feature Importance Analysis

Feature importance was extracted from the final trained model.

**Top Predictive Features**
- Month (Campaign timing)
- Marital Status
- Housing Loan Status
- Account Balance
- Campaign Day

**Interpretation**

Customer behavior is influenced by:

- **Seasonality**
- **Financial stability**
- **Household characteristics**

These features provide meaningful business signals.

## Customer Segmentation Insights

Customer segmentation was performed using:

**Probability-based grouping**

Method:

df.groupby('feature')['y'].mean()

This calculates:

**Subscription Probability per Group**

## Key Segmentation Results
**High-Response Customer Groups**
- Students → **15.6% response**
- Retired → **10.5% response**
  
**Education-Based Insights**

Tertiary education showed highest subscription rates.

**Time-Based Insights**

Best months:

- October
- March
- April

Indicates:

## Seasonal Customer Behavior

**Balance-Based Segmentation**

Used:

**Quantile Binning (qcut)**

Result:

Higher balance → Higher subscription probability.

## Business Impact

This model supports **targeted marketing strategy.**

**Expected Benefits**
**Improved Targeting**

Focus campaigns on:

- Students
- Retired customers
- High-balance clients

**Cost Reduction**

Reduce unnecessary outreach to:

Low-probability customers.

**Better Campaign ROI**

Prioritize:

- High-conversion months
- Financially stable customers
## Conclusion

This project demonstrates a **complete machine learning pipeline,** from raw data to actionable business insights.

Key strengths include:

- Proper handling of imbalanced data
- Prevention of data leakage
- Model optimization using Randomized Search
- Business-driven interpretation of results

The final model successfully predicts customer subscription behavior and provides actionable insights for improving marketing efficiency.

## For Hiring Managers & Recruiters

This project demonstrates:

- **End-to-End Machine Learning Workflow**

From data exploration to final evaluation.

- **Strong Feature Engineering Decisions**

Including realistic feature removal (duration, contact).

- **Hyperparameter Optimization Skills**

Using Randomized Search.

- **Business-Oriented Thinking**

Model insights translated into actionable strategy.

- **Real-World Modeling Practices**

Avoided leakage and ensured deployment-ready logic.
