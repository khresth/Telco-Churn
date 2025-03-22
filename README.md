# Customer Churn Analysis Report

This report analyzes telecom customer churn using extensive data science methods, including exploratory analysis, clustering, survival modeling, and machine learning. Key findings reveal that short tenure and month-to-month contracts drive churn, with actionable recommendations for retention.


1. Introduction

I loaded the Telco customer churn dataset into a collab notebook. It’s a rich mix of features like gender, tenure, InternetService, and MonthlyCharges with the target variable Churn telling me whether a customer left (Yes) or stayed (No). Here’s what I set out to do:
-	Pinpoint the biggest predictors of churn.
-	Build a model to forecast who’s at risk.
-	Offer clear, actionable ideas to reduce churn.

2. Dataset Overview and Data Cleaning

I started by sizing up the dataset with df.info() and df.shape. It’s got 7,043 rows and 21 columns, blending categorical variables like Contract with numbers like tenure. The target, Churn, is binary making it perfect for a yes/no prediction.

- Missing Values: I spotted 11 missing entries in TotalCharges. Using pd.to_numeric(errors='coerce'), I turned it numeric and dropped those rows with df.dropna(), leaving me 7,032 clean records.

-	Encoding: Models love numbers, so I mapped binary variables like gender (Male=0, Female=1) and Churn (No=0, Yes=1) and used pd.get_dummies(drop_first=True) for multi-class ones like PaymentMethod. The drop_first bit avoids multicollinearity headaches.

-	Scaling: For numerical features (tenure, MonthlyCharges, TotalCharges), I whipped out StandardScaler from sklearn.preprocessing to normalize them. This helps  models like Logistic Regression handle big numbers.

3. Exploratory Data Analysis (EDA)

I dug into the data with pandas and seaborn to spot patterns.
-	Churn Breakdown: A seaborn.countplot showed 26.54% of customers churned (1,869) while 73.46% stayed (5,163). It’s imbalanced, so I kept that in mind for modeling.

![image](https://github.com/user-attachments/assets/4f46ec6e-9cbf-45e7-b977-ec5f6831bd77)

  -	Tenure and Churn: With a seaborn.boxplot, I saw churned customers stick around for a median of 10 months, while loyal ones hit 38 months. New joiners are prone to leave.
![image](https://github.com/user-attachments/assets/20923fa6-bc25-461e-b8b6-53b968691ac7)

- Contracts Matter: A pd.crosstab and seaborn.barplot revealed 42.7% of month-to-month customers churn, versus 11.3% on one-year and 2.8% on two-year contracts. Long-term deals keep people in.

![image](https://github.com/user-attachments/assets/d5bc25a9-7989-4d80-987b-9f0a13dd6001)

- Correlations: I fired up seaborn.heatmap to check feature relationships. tenure and Contract tied strongly to churn, but not gender.

![image](https://github.com/user-attachments/assets/86854423-161e-4d8e-bd60-4f4498a48d9c)

4. Statistical Analysis

-	Gender vs. Churn: A chi-square test with scipy.stats.chi2_contingency gave me a statistic of 0.48 and a p-value of 0.4905. Since p > 0.05, gender’s not a player here.

-	Tenure vs. Churn: A t-test via scipy.stats.ttest_ind compared tenure for churners and stayers. A t-statistic of -31.74 and p-value of 0.0000. With p < 0.05, tenure’s a big deal.

-	TotalServices: I summed up services like PhoneService and InternetService for each customer. More services might mean they’re hooked.

-	Tenure Groups: I binned tenure into New (0-12 months), Medium (13-24 months), and Loyal (25+ months) with pd.cut. This catches quirky trends.
  
-	Encoding: I ran pd.get_dummies again to make everything model-ready.

5. Customer Segmentation with Clustering
I grouped customers with sklearn.cluster.KMeans (set n_clusters=3) on scaled features:

- Churn Rates: A seaborn.barplot showed Cluster 0: 15% churn, Cluster 1: 45% churn, and Cluster 2: 25% churn. (Insert "Churn Rate by Cluster" graph from notebook here)

  ![image](https://github.com/user-attachments/assets/71b2e8ab-4553-43e3-9c72-9d37f7fb9010)

- Visuals: I used sklearn.decomposition.PCA to squash the data into 2D and plotted it with matplotlib. Clear clusters popped out. (Insert "Customer Clusters in 2D (PCA)" graph from notebook here)
  ![image](https://github.com/user-attachments/assets/4fd13d2b-49e8-40f4-9cbf-11b0dc2641ba)

I split customers into three crews: low-risk (15% leave), high-risk (45% leave), and middle-ground (25% leave). A 2D map shows these groups, so we know who to watch.

6. Predictive Modeling with Logistic Regression

- I split the data 80/20 with sklearn.model_selection.train_test_split (random_state=42), scaled features, and trained a sklearn.linear_model.LogisticRegression model (max_iter=1000).
  ![image](https://github.com/user-attachments/assets/f0876a71-11d9-42c0-8981-79675d086da3)

-	Accuracy: 79%
-	Precision (Churn): 62% (of predicted churners, 62% were right)
-	Recall (Churn): 51% (caught 51% of actual churners)
-	ROC AUC: 0.83 (via sklearn.metrics.roc_auc_score) solid at separating classes

•	Confusion Matrix: A seaborn.heatmap showed 920 true negatives, 113 false positives, 183 false negatives, and 191 true positives. (Insert "Confusion Matrix - Logistic Regression" graph from notebook here)
![image](https://github.com/user-attachments/assets/939d3f37-e4a8-4b26-b904-2be9a141c651)
I made a prediction tool that’s right 79% of the time. It’s better at spotting stayers than leavers, but a 0.83 ROC score says it’s pretty sharp overall. The matrix shows I’m catching some churners, though I miss a chunk too.

7. Conslusion

•	Tenure’s Key: Stats and graphs prove shorter tenure means higher churn risk.

•	Contracts Lock Them In: Month-to-month folks are flighty; longer contracts keep them.

•	Risky Clusters: Cluster 1’s 45% churn rate screams for attention.

•	Model Power: My 79% accurate model can flag at-risk customers.

What to Do

•	Welcome Deals: Hook newbies with discounts or perks.

•	Contract Perks: Sweeten the pot for one or two-year sign-ups.

•	Target High-Risk: Focus support on Cluster 1 types.








