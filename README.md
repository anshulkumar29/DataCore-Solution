# Customer Churn Prediction: Project Notes (Phase by Phase)

This README records what we did in each phase, why we did it, and what to say about it to a panel. It is also meant to be used as a knowledge source for the project chatbot, so each phase has clear headings, plain-language explanations and Q&A.

**Status:** Phases 1 to 4 are done and written up here. Phases 5 and 6 are pending.

---

## Project overview

- **Problem:** find which customers are likely to churn (leave) and understand why, so a retention team can act early.
- **Data file:** `customer_churn_dataset-testing-master.csv`
- **Approach:** understand data, clean, analyse (EDA and statistics), model, then present results in a Streamlit dashboard with a chatbot.

## Project files

| File | Purpose |
|---|---|
| `customer_churn_dataset-testing-master.csv` | The dataset (64,374 customers) |
| `data_check.ipynb` | Phase 1: understanding the data |
| `data_cleaning.ipynb` | Phase 2: data cleaning and preparation |
| `churn_clean.csv` | Output of Phase 2: readable clean data (no ID), used for EDA and dashboard |
| `churn_model_ready.csv` | Output of Phase 2: encoded numeric data, used for modeling |
| `data_eda.ipynb` | Phase 3A: graphs and exploration |
| `data_statistics.ipynb` | Phase 3B: statistical tests and hypothesis verdicts |
| `data_modeling.ipynb` | Phase 4: models, metrics, final model |
| `churn_model.pkl` | Output of Phase 4: saved final Random Forest model |
| `model_features.json` | Output of Phase 4: column order the model expects |
| `model_metrics.json` | Output of Phase 4: scores and top features |
| `churn_risk_scores.csv` | Output of Phase 4: churn probability and risk level for every customer |
| `Churn.ipynb`, `Churn_backup.ipynb` | Earlier full analysis and modeling |
| `Statistical_Testing.ipynb` | Earlier statistical tests |
| `streamlit_dashboard.py`, `streamlit_dashboard_rag.py` | Dashboard versions |
| `chatbot.py`, `chatbot_rag.py`, `knowledge.md` | Chatbot and retrieval knowledge |
| `README.md` | This file |

---

## Phase 1: Understanding the data (DONE)

**Notebook:** `data_check.ipynb`

### What the dataset is
- 64,374 rows and 12 columns.
- One row equals one customer, because `CustomerID` is unique for every row.
- One table only, so no joins are needed.
- Target column: `Churn` (1 = customer left, 0 = customer stayed).

### Data dictionary

| Column | Type | Meaning | Range or values |
|---|---|---|---|
| CustomerID | ID | Unique customer identifier | 1 to 64374 |
| Age | Numeric | Customer age | 18 to 65 |
| Gender | Categorical | Customer gender | Male, Female |
| Tenure | Numeric | Time with the company (assumed months) | 1 to 60 |
| Usage Frequency | Numeric | How often the service is used | 1 to 30 |
| Support Calls | Numeric | Number of calls to customer support | 0 to 10 |
| Payment Delay | Numeric | Delay in paying (assumed days) | 0 to 30 |
| Subscription Type | Categorical | Plan type | Basic, Standard, Premium |
| Contract Length | Categorical | Contract duration | Monthly, Quarterly, Annual |
| Total Spend | Numeric | Total amount spent | 100 to 1000 |
| Last Interaction | Numeric | Time since last contact (assumed days) | 1 to 30 |
| Churn | Target | Whether the customer left | 0 or 1 |

Units for Tenure, Payment Delay and Last Interaction are assumed from the column names and should be confirmed with the problem statement or panel.

### Key findings
1. No missing values and no duplicate rows.
2. No impossible values (age 18 to 65, support calls 0 to 10, payment delay 0 to 30, total spend 100 to 1000).
3. Churn rate is about 47.4% (52.6% stayed), so the classes are almost balanced. SMOTE and class weights are not needed. Accuracy is fair, but precision, recall and F1 are still reported.
4. Numeric columns have mean close to median and evenly spaced quartiles, which suggests simulated data. This is to be confirmed with histograms in Phase 3.
5. `CustomerID` is only an identifier and is dropped before modeling.
6. The file name says "testing", so it may be one half of a train/test split.

### Column groups
- **ID:** CustomerID (drop before modeling)
- **Target:** Churn
- **Categorical:** Gender, Subscription Type, Contract Length
- **Numeric:** Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend, Last Interaction

### Hypotheses to test in Phase 3
1. More support calls means more churn.
2. Longer payment delay means more churn.
3. Monthly contracts churn more than annual contracts.
4. Longer tenure means less churn.
5. Lower usage frequency means more churn.

### How to approach a new dataset in a hackathon
1. Read the problem statement: goal, target column, definition of churn.
2. Load the data and check `shape`, `head()`, `info()`.
3. Work out what one row represents (the grain).
4. Check missing values, duplicates and impossible values. Only list problems at this stage.
5. Check target balance.
6. Classify columns as numeric, categorical, ID or target.
7. Write hypotheses before analysis.
8. Prepare a short spoken summary.

### 30-second spoken summary
"The dataset has about 64,000 customers, one row per customer, with 11 features covering demographics, subscription, usage, support, payment and spend. The target is Churn, and about 47% of customers churned, so the classes are nearly balanced. There are no missing values, duplicates or impossible values. I will drop CustomerID since it is only an identifier. My hypotheses are that support calls, payment delay and contract type drive churn, and I will test them statistically next."

### Likely panel questions (Phase 1)
- **What does one row represent?** One customer, because CustomerID is unique.
- **Why drop CustomerID?** It is unique per row and carries no pattern, so a model could fit noise.
- **Is the data imbalanced?** No, about 47% vs 53%, so accuracy is meaningful, but we also report precision, recall and F1.
- **Which columns are categorical and which are numeric?** Gender, Subscription Type and Contract Length are categorical. Age, Tenure, Usage Frequency, Support Calls, Payment Delay, Total Spend and Last Interaction are numeric.
- **How did you check data quality?** `isnull().sum()`, `duplicated().sum()`, and min and max from `describe()`, plus explicit checks for impossible values.

---

## Phase 2: Data cleaning and preparation (DONE)

**Notebook:** `data_cleaning.ipynb`
**Outputs:** `churn_clean.csv` (64,374 rows, 11 columns) and `churn_model_ready.csv` (64,374 rows, 13 columns)

### Summary
The data was already clean, so this phase was mostly verification and preparation. Each check was run rather than assumed.

### Steps and findings
1. **Working copy:** the raw data is kept untouched in `raw`, and all changes are made on `df`.
2. **Missing values and duplicates:** none found, so nothing was removed.
3. **Text consistency:** Gender, Subscription Type and Contract Length have only their expected values and no spacing problems.
4. **Outliers (IQR rule, boxplots):** zero outliers in all 7 numeric columns, and skewness is close to 0 (the largest magnitude is Payment Delay at about -0.35). All rows are kept.
5. **Logical checks:** no tenure longer than the customer's age, no negative values, and Churn contains only 0 and 1.
6. **Dropped CustomerID:** it is only an identifier and carries no pattern.
7. **One-hot encoding with `drop_first=True`:** the 3 text columns became 5 numeric columns: Gender_Male, Subscription Type_Premium, Subscription Type_Standard, Contract Length_Monthly, Contract Length_Quarterly. Baseline categories are Female, Basic and Annual.
8. **Saved two datasets:** the readable one for EDA and the dashboard, and the encoded one for modeling. The churn rate is preserved at 47.37%.

### What we would do if the data were dirty

| Problem | Option |
|---|---|
| Missing numeric value | Fill with the median (not affected by extreme values) |
| Missing category | Fill with the most frequent value or an "Unknown" label |
| Column mostly missing (above about 40%) | Drop the column |
| Duplicate rows | `drop_duplicates()` |
| Data-error outlier (age 200) | Remove or correct |
| Real extreme value (very high spender) | Keep or cap at the IQR limit, do not delete |

### Decisions worth explaining
- **Why one-hot and not 0, 1, 2:** numeric labels imply an order that does not exist. One-hot treats each category separately.
- **Why `drop_first=True`:** one column per category group is redundant, and keeping it causes multicollinearity.
- **Why scaling is not done here:** scaling must be fitted on the training data only, after the train/test split, otherwise test information leaks into training. It is done in Phase 4.
- **Why two datasets:** the readable one is for analysis and the dashboard, the encoded one is for models.

### 30-second spoken summary
"The data was already clean: no missing values, no duplicates, no outliers by the IQR rule, and no logical inconsistencies. I verified each of these rather than assuming it. I dropped CustomerID because it is only an identifier, and one-hot encoded the three categorical columns so models can use them. I saved a readable version for analysis and an encoded version for modeling. Scaling is done in the modeling phase after the train/test split to avoid data leakage."

### Likely panel questions (Phase 2)
- **Why did you not remove any outliers?** The IQR rule found none. If there were real extreme values, I would keep or cap them, because they may be the most valuable customers.
- **How would you handle missing values?** Median for numeric columns, most frequent value for categories, and drop a column if most of it is missing.
- **Why one-hot encoding?** Numeric labels create a fake order. One-hot avoids that.
- **Why not scale now?** Scaling before the split leaks test information into training.
- **Why drop CustomerID?** It is unique per row, so the model could memorise it and learn noise.

---

## Phase 3: EDA and statistical analysis (DONE)

**Notebooks:** `data_eda.ipynb` (graphs) and `data_statistics.ipynb` (tests). Both read `churn_clean.csv`. The graphs suggest patterns, and the tests check whether the patterns are real or just chance.

**One-line answer:** "A graph can mislead me, a test gives a number. EDA gives me hypotheses and statistics confirms them."

### Part A: EDA with graphs (`data_eda.ipynb`)

| Step | Graph | Question it answers |
|---|---|---|
| 1 | Bar chart | How many customers churn? |
| 2 | Histograms | What shape does each numeric column have? |
| 3 | Boxplots split by churn | Do churners differ from stayers on numbers? |
| 4 | Churn rate bars by category | Does churn change by gender, plan, contract? |
| 5 | Churn rate by bands | At what level does the risk jump? |
| 6 | Correlation heatmap | Which columns move together or with churn? |
| 7 | Segment heatmap and risk flags | Which customer groups are riskiest? |
| 8 | Insights table | What do we tell the panel? |

**How to read the main graphs**
- **Histogram:** bars are value ranges and height is the number of customers. Bell shape means roughly normal, a long tail means skewed, several peaks mean mixed groups.
- **Boxplot:** the middle line is the median and the box holds the middle 50% of customers. Boxes that barely overlap mean a strong difference.
- **Churn rate (not count):** groups have different sizes, so a rate compares them fairly.
- **Bands:** cutting a number into ranges shows where risk jumps, which the business can act on.
- **Heatmap:** near +1 means move together, near -1 means opposite, near 0 means no linear link. Correlation shows linkage, not cause.

### What EDA found
- **Balance:** 52.6% stayed and 47.4% churned, so the classes are balanced and accuracy is a fair metric.
- **Payment Delay is the strongest signal.** Median delay is 23 days for churners versus 12 for stayers. Churn is about 10% in the lowest band, about 63% once the delay passes 16 days, and about 77% above 21 days.
- **Support Calls:** median 7 for churners versus 4 for stayers. Churn rises from 23% (0 to 2 calls) to about 61% (6 or more calls) and then flattens.
- **Tenure goes the opposite way to what most people expect.** Churners have longer tenure (median 37 versus 27 months). Do not claim that new customers churn more, because the data says the opposite.
- **Gender matters more than plan or contract:** Female 55.0% versus Male 38.6%. Subscription type is almost flat (46.5% to 48.3%). Contract is a small effect (Monthly 51.6%, Quarterly 44.0%).
- **The contract by subscription map is nearly uniform**, so plan and contract alone do not identify risky segments.
- **Risk flags:** customers with none of the 3 top risk flags churn 12%, with 1 flag 35%, with 2 flags 64%, and with all 3 flags 100%. A perfect 100% looks unusually clean, which is typical of a synthetic or heavily curated dataset, and we say so openly.
- **Last Interaction** has no relation to churn (correlation about 0).

### Part B: Statistical tests (`data_statistics.ipynb`)

| Step | Test | Used for | Why this test |
|---|---|---|---|
| 1 | Shapiro-Wilk and skewness | Is the data normal? | Decides which test is fair |
| 2 | Mann-Whitney U with effect size | Numeric column versus churn | Two groups, numeric outcome, data not normal |
| 3 | Chi-square with Cramer's V | Categorical column versus churn | Both variables are categories, compares observed and expected counts |
| 4 | Spearman correlation | Strength and direction of links | Works on ranks, so skew does not distort it |
| 5 | VIF | Do features duplicate each other? | Overlapping features make linear models unstable |
| 6 | Verdict table | Each hypothesis: supported or not | Combines p-value, direction and effect size |

**Two ideas to remember**
- **p-value:** the chance of seeing a difference this big if there were really no effect. Below 0.05 means "significant".
- **Effect size:** how big the difference is. With 64,000 rows almost everything is significant, so effect size tells us whether it matters. Rough guide: below 0.1 negligible, 0.1 to 0.3 small, 0.3 to 0.5 medium, above 0.5 large.

### What the tests found
- **Not normal:** every numeric column fails Shapiro-Wilk, so Mann-Whitney U is the right test. With 64,000 rows Shapiro flags even tiny departures, and skewness is small (between -0.35 and +0.05). A Welch t-test cross-check gives the same conclusions.
- **Numeric versus churn:** Payment Delay has a large effect (r = 0.64), Support Calls medium (0.34), Tenure small (0.22), Usage Frequency small (-0.14). Total Spend and Age are significant but negligible. **Last Interaction is not significant (p = 0.47).**
- **Categorical versus churn:** all three are statistically significant, but only Gender has a real effect (Cramer's V = 0.16, small). Contract (0.06) and Subscription (0.01) are significant only because the sample is large, and their effect is negligible. All expected counts were at least 5, so the chi-square assumption holds.
- **Spearman** agrees with the EDA heatmap: Payment Delay 0.55, Support Calls 0.30, Tenure 0.19.
- **VIF:** the maximum is 1.34, so there is no multicollinearity and linear models are safe to use.

### Hypothesis verdicts

| Hypothesis (as tested in `data_statistics.ipynb`) | Test | Verdict | Numbers |
|---|---|---|---|
| H1: More support calls, higher churn | Mann-Whitney U | Supported | effect r = 0.34 (medium) |
| H2: Longer payment delay, higher churn | Mann-Whitney U | Supported | effect r = 0.64 (large) |
| H3: Contract length affects churn | Chi-square | Supported, weak effect | Cramer's V = 0.06 (negligible) |
| H4: Shorter tenure, higher churn | Mann-Whitney U | **Not supported** | direction is reversed, effect r = 0.22 (small) |
| H5: Subscription type affects churn | Chi-square | Supported, weak effect | Cramer's V = 0.01 (negligible) |

**Note on wording:** the Phase 1 hypothesis list says "longer tenure means less churn" and "lower usage means more churn" for H4 and H5. The verdict table above uses the wording from the statistics notebook. Both describe the same finding: tenure is the reverse of the expected direction. Lower usage does go with more churn (r = -0.14), but the effect is small, so do not oversell it.

**Takeaway:** a "supported" result with a negligible effect size is statistically real but practically unimportant. The strong drivers are Payment Delay and Support Calls, then Tenure and Gender.

### 30-second spoken summary
"I started with graphs to see the data. The classes are balanced, and boxplots, churn-rate bars and bands showed that payment delay and support calls separate churners from stayers most clearly, with a sharp jump in risk after about 15 days of delay and after 4 support calls. Because the data is not normal, I used Mann-Whitney U for numeric columns and chi-square for categories, and reported effect sizes, not only p-values, because with this much data everything looks significant. Payment delay had a large effect and support calls a medium one. Tenure went the opposite way to my hypothesis, and contract and subscription were significant but negligible. VIF showed no multicollinearity."

### Likely panel questions (Phase 3)
- **Why a boxplot and not just the average?** The box shows spread, and the median is not distorted by extreme values.
- **Why churn rate and not count?** Groups have different sizes, and a rate compares them fairly.
- **Why bands?** They turn a number into a threshold the business can act on.
- **Does correlation mean cause?** No. It shows two things move together, not that one causes the other.
- **Why Mann-Whitney and not a t-test?** The data is not normal, and Mann-Whitney does not assume normality. A t-test cross-check gave the same conclusions.
- **Shapiro says everything is not normal, but skewness is tiny. Is that a problem?** With 64,000 rows Shapiro flags tiny departures. The skewness is small, so the choice does not change the conclusions.
- **What does p < 0.05 mean?** If there were no real effect, a result this extreme would happen less than 5% of the time.
- **Is a significant result always important?** No. Check the effect size too. Contract and subscription are significant but negligible.
- **Why chi-square for categories?** Both variables are categorical, and chi-square compares observed counts with the counts expected if they were independent.
- **Why Spearman?** It works on ranks, so skew and outliers do not distort it.
- **What is VIF for?** It catches features that duplicate each other, which makes linear models unstable.
- **Your hypothesis about tenure was wrong. What do you do?** Report it honestly. Longer-tenure customers churn more in this data, so I would not claim that new customers are the risk.
- **Why is the 3-flag segment exactly 100% churn?** It looks unusually clean, which suggests a synthetic or rule-based dataset. I say this openly, and it also explains the very high model scores in Phase 4.
- **What did you add beyond the basics?** The contract by subscription segment map and the risk-flag score.

---

## Phase 4: Modeling and metrics (DONE)

**Notebook:** `data_modeling.ipynb`

### Summary
We split the encoded data 80/20 (stratified, seed 42), trained five models from simple to complex, compared them on the hidden test set, checked stability with 5-fold cross-validation, chose Random Forest, and saved it for the dashboard.

### Results (test set, 12,875 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (majority) | 0.526 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.827 | 0.814 | 0.824 | 0.819 | 0.903 |
| Decision Tree (depth 5) | 0.954 | 0.931 | 0.975 | 0.952 | 0.994 |
| Random Forest (200 trees) | 0.998 | 0.999 | 0.997 | 0.998 | 1.000 |
| Gradient Boosting | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

5-fold cross-validation F1: Logistic Regression 0.819, Decision Tree 0.955, Random Forest 0.998 (std dev 0.0003), Gradient Boosting 1.000.

### The models in simple words
- **Dummy:** always guesses the majority class. The floor to beat.
- **Logistic Regression:** one weight per feature and a straight boundary. Needs scaling (done inside a pipeline, fitted on train only).
- **Decision Tree:** a flowchart of yes/no questions. Readable, but a deep tree memorises the data.
- **Random Forest:** 200 trees trained on random parts of the data that vote. More stable than one tree.
- **Gradient Boosting:** trees built one after another, each fixing the mistakes of the previous ones.

### Metrics in plain words
- **Accuracy:** share of predictions that were right.
- **Precision:** of customers flagged as churn, how many really left (low precision wastes retention offers).
- **Recall:** of customers who really left, how many we caught (low recall means lost customers we never saw).
- **F1:** balance of precision and recall.
- **ROC-AUC:** chance that a random churner gets a higher risk score than a random stayer (0.5 is a coin flip, 1.0 is perfect).
- **Confusion matrix:** counts of true/false positives and negatives. A false negative (missed churner) is usually the costlier mistake.

### Why the scores are so high (leakage check)
- No feature is suspiciously linked to the target (strongest correlation is Payment Delay at 0.56).
- Train and test scores are close, so the models are not memorising.
- Cross-validation is stable (tiny spread).
- The data has sharp threshold rules: churn is about 10% up to 15 days of payment delay and 59% to 77% beyond it, and about 25% up to 3 support calls versus about 60% from 5 calls. A tree with only 4 levels already reaches about 94%.
- Real data is noisier, so scores would be lower in production and should be re-validated on fresh customers.

### Final model: Random Forest
Chosen because its scores are essentially equal to boosting, it is easy to explain ("200 trees vote"), and it gives feature importance. It was retrained on all customers for deployment after validation.

**Top features (Random Forest importance):** Payment Delay (0.46), Support Calls (0.17), Tenure (0.10), Usage Frequency (0.08), Gender_Male (0.07). Contract and Subscription columns, and Last Interaction, matter very little.

### Hypotheses seen through the model
- H1 (more support calls, more churn): supported.
- H2 (longer payment delay, more churn): strongly supported.
- H3 (monthly contracts churn more): weakly supported (Monthly 51.6%, Annual 46.2%, Quarterly 44.0%).
- H4 (longer tenure, less churn): not supported. Churn is about 31% for tenure up to 24 months and about 56% above 24.
- H5 (lower usage, more churn): partly supported, mainly for very low usage (up to 5).
- Gender: women churn at 55%, men at 38.6%. Do not use gender to target retention offers (fairness).

### Risk bands (used by the dashboard)
Each customer gets an out-of-fold churn probability (scored by a model that did not train on them), then a band:

| Band | Probability | Customers | Actual churn rate |
|---|---|---|---|
| Low | below 0.30 | 33,557 | about 0.01% |
| Medium | 0.30 to 0.70 | 866 | about 62% |
| High | 0.70 and above | 29,951 | 100% |

The threshold can be moved: a lower threshold catches more churners but flags more customers who would have stayed.

### Files created
`churn_model.pkl`, `model_features.json`, `model_metrics.json`, `churn_risk_scores.csv`. The notebook also defines `predict_customer(...)`, which turns raw inputs (for example "Male", "Monthly") into the encoded columns and returns the churn probability and risk level. The dashboard reuses this logic.

### 30-second spoken summary
"I split the data 80/20 with stratification and compared five models, starting from a dummy baseline so I knew what to beat. Logistic Regression reached about 83% accuracy, tree-based models reached 95% to 100%, because churn here is driven by sharp thresholds like support calls above 4 and payment delay above 15 days. I chose Random Forest: it is as accurate as boosting, easy to explain, and gives feature importance. Because the scores were so high I checked for leakage and overfitting and found neither, and cross-validation was stable. The final model gives each customer a churn probability, grouped into Low, Medium and High risk, which feeds the dashboard."

### Likely panel questions (Phase 4)
- **Why is your accuracy so high? Is it leakage?** I checked. No feature leaks the target, train and test scores are close, and cross-validation is stable. The data has sharp threshold rules that trees learn easily. Real data would score lower, so I would re-validate on fresh customers.
- **Why Random Forest and not Gradient Boosting?** Scores are essentially equal. Random Forest is easier to explain and gives feature importance directly.
- **Why start with Logistic Regression?** It is a simple, explainable baseline that shows how much the non-linear models add.
- **Why stratify the split?** To keep the same churn share in train and test.
- **Why scale only for Logistic Regression?** It is sensitive to feature size. Trees only compare thresholds. Scaling is fitted on train data only, inside a pipeline, to avoid leakage.
- **Which metric matters most?** Recall for churn, because a missed churner is a lost customer, while watching precision so offers are not wasted. ROC-AUC shows how well risk is ranked.
- **Does feature importance prove causes?** No. It shows what the model relies on, not what causes churn.
- **Why retrain on all data at the end?** Validation is already done, and the final model benefits from every customer.
- **What would you do with more time?** Tune hyperparameters, test on a new time period, set the threshold from the real cost of a lost customer versus an offer, and explain single predictions with SHAP.

---

## Phase 5: Streamlit dashboard and chatbot (PENDING)

To be filled after the phase. Expected content: dashboard KPIs and filters, churn risk table, charts, how the chatbot uses this README and the data, and how the pieces connect.

## Phase 6: Panel practice (PENDING)

To be filled after the phase. Expected content: a 2-minute pitch and likely questions per phase with short answers.
