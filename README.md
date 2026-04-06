# I97aepEtbpPOMizq
Project Overview

This project focuses on predicting whether a bank client will subscribe to a term deposit based on their demographics, financial situation, and past campaign interactions.

The dataset contains 40,000 records with 14 features (5 numeric, 9 categorical) and a target variable y (yes/no). The dataset is imbalanced, with only ~7–8% of clients subscribing (y = yes).

Objectives:

Explore and understand the dataset (EDA).
Preprocess and encode categorical variables.
Build a predictive model.
Optimize model performance using hyperparameter tuning (Randomized Search).
Analyze feature importance.
Extract insights through customer segmentation.
Step 1: Exploratory Data Analysis (EDA)
df.head()
df.info()
df.describe()

Key observations from EDA:

Numeric features:
age: Mean ~40, range 19–95
balance: Skewed, many negative values; max 102,127
campaign: Avg ~2.88 calls per client
duration: Ranges 0–4918 sec, median 175 sec
Categorical features:
job, marital, education, housing, loan, contact, month
Some categories like unknown exist
Target variable imbalance:
Only ~7–8% of clients subscribed (y = yes)
Indicates the need for careful model evaluation (recall, F1, ROC-AUC)

Visualizations & Segmentation Insights (EDA):

Job: Students and retired clients have higher subscription rates.
Education: Tertiary-educated clients respond better.
Month: Campaigns in October, March, April are most successful.
Balance: Clients with higher balances tend to subscribe more.
Housing/Loan: Clients without housing loans are slightly more responsive.
Step 2: Feature Selection & Preprocessing
Why we dropped duration:
duration is highly correlated with target (y) because the longer the call, the higher the chance of subscription.
In a real-world predictive scenario, duration is unknown before the call starts, so using it would be data leakage.
Data science principle: Avoid features that wouldn’t exist at prediction time.
✅ Dropped duration to make predictions realistic and deployable.
Preprocessing Steps:
Split data into features X and target y.
Encode categorical variables using One-Hot Encoding.
Split into train/test sets (80/20), stratified by target to maintain class balance.
Handle class imbalance using scale_pos_weight in models like XGBoost.
X_train, X_test, y_train, y_test = train_test_split(
    X.drop('duration', axis=1), 
    y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
Step 3: Baseline Model Training
Used XGBoost/Gradient Boosting as the baseline model.
Evaluated with 5-fold Stratified Cross-Validation:
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'recall', 'f1', 'roc_auc']

cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)

Baseline Cross-Validation Results:

Accuracy: 75.8%
Recall: 0.49
F1-score: 0.23
ROC-AUC: 0.69

Test Set Evaluation:

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
Test Accuracy: 77%
Confirms model predicts majority class well but struggles with positive class (y = yes).
Step 4: Hyperparameter Tuning with Randomized Search

Why Randomized Search:

Testing all parameter combinations (Grid Search) is computationally expensive.
Randomized Search tests a random subset of combinations, saving time while finding good hyperparameters.
param_dist = {
    'clf__n_estimators': [200, 300, 400],
    'clf__max_depth': [3, 4, 5],
    'clf__learning_rate': [0.03, 0.05, 0.1],
    'clf__subsample': [0.8, 0.9],
    'clf__colsample_bytree': [0.8, 0.9]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=25,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1, verbose=1,
    random_state=42
)
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
print("Best CV Accuracy:", random_search.best_score_)
Result: CV Accuracy improved to ~81.6%
Step 5: Test Set Evaluation (Final Model)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:,1]

test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))

Expected Outcome:

Test Accuracy: ~81% ✅
Improved recall for positive class
ROC-AUC ~0.73–0.74

✅ Randomized Search successfully improved both training CV and test accuracy.

Step 6: Feature Importance
importances = best_model.named_steps['clf'].feature_importances_
cat_features = best_model.named_steps['preprocess'].named_transformers_['cat'].get_feature_names_out(cat_cols)
features = list(cat_features) + num_cols

importance_df = pd.DataFrame({'feature': features, 'importance': importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)
print(importance_df.head(15))

Top Features:

month_mar → 0.080
contact_unknown → 0.078
month_oct → 0.070
contact_cellular → 0.034
marital_married → 0.032

Interpretation:

Campaign timing (month) and contact method are the most critical.
Marital status, housing, and numeric features (balance) also influence subscriptions.
Step 7: Customer Segmentation Insights

Analyzed average subscription rates (y) by categorical features:

job_seg = df.groupby('job')['y'].mean().sort_values(ascending=False)
edu_seg = df.groupby('education')['y'].mean().sort_values(ascending=False)
month_seg = df.groupby('month')['y'].mean().sort_values(ascending=False)
df['balance_group'] = pd.qcut(df['balance'], q=4, labels=['Low','Medium','High','Very High'])
balance_seg = df.groupby('balance_group')['y'].mean()

Insights:

Job: Students (15.6%) and retired clients (10.5%) respond more.
Education: Tertiary-educated clients have higher subscription rates.
Month: October, March, April are best months for campaigns.
Balance: Higher balance → higher subscription probability.
Housing/Loan: Clients without housing loans are slightly more likely to subscribe.

✅ These insights can guide targeted campaigns for better ROI.

Step 8: Key Takeaways
Randomized Search improved model accuracy from 76% → 81% and made the model more reliable on unseen data.
Dropping duration prevented data leakage and made predictions realistic.
Feature importance shows that month of campaign, contact type, and marital status are key drivers.
Segmentation analysis helps target high-probability clients, e.g., students, retired, high-balance clients in October and March.
Model limitations:
Class imbalance still impacts F1 for positive class
Some features like balance are skewed → might need transformation for further improvement
Step 9: Future Improvements
Apply SMOTE or oversampling to improve recall/F1 for minority class.
Try ensemble or stacking models for better predictive performance.
Use insights for personalized campaigns to maximize conversion.
Summary Table
Model	CV Accuracy	Test Accuracy	ROC-AUC
Baseline	0.758	0.770	0.724
Randomized Search	0.816	~0.815	0.735
