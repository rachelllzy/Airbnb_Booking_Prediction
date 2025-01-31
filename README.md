# <center>Airbnb Booking Prediction<center/>

## **Project Overview**

This project is a machine learning competition focused on predicting the number of days an Airbnb listing will be booked in the next 30 days. The dataset consists of real Airbnb listings from Los Angeles, with actual realized bookings available for only a small subset of listings.

To build an effective predictive model, I applied a combination of **advanced feature engineering, model selection, hyperparameter tuning, and interpretability techniques**. The main objective was to **maximize the model’s generalization performance**, ensuring it achieves high accuracy on unseen test data.

## **Dataset Description**

The dataset includes a variety of structured and semi-structured attributes about Airbnb listings. To extract meaningful patterns, the following feature categories were utilized:

​	•	**Categorical Features:** Property type, neighborhood, amenities, host verification status, etc.

​	•	**Numerical Features:** Price, number of reviews, review scores, latitude/longitude, etc.

​	•	**Boolean Features:** Superhost status, availability, instant bookability, etc.

​	•	**Time-based Features:** Host registration date, last review date, first review date.

​	•	**Text Features (Partially Explored):** Listing descriptions and host responses, where possible.

The target variable is **the number of days a listing will be booked in the next 30 days**.

## **Feature Engineering**

A crucial aspect of this project was transforming raw Airbnb data into **highly informative, predictive features**. Key feature engineering steps include:

​	1.	**Feature Selection and Reduction:**

​	•	Removed non-contributory and highly correlated features to prevent multicollinearity.

​	•	Used VarianceThreshold to drop features with near-zero variance.

​	•	Applied **Mutual Information Score (MI)** to rank features based on their dependency with the target variable.

​	2.	**Categorical Feature Encoding:**

​	•	Applied **Target Encoding** for high-cardinality categorical features.

​	•	Used **One-Hot Encoding** selectively for frequently occurring categories.

​	•	Grouped rare categories (e.g., property types with fewer than 10 listings) into an “Others” class.

​	3.	**Boolean Feature Transformation:**

​	•	Converted binary categorical features into **0/1 numerical format**.

​	•	Created **interaction features**, such as is_superhost & instant_bookable, to capture complex relationships.

​	4.	**Text Feature Engineering (Exploratory):**

​	•	Extracted **TF-IDF embeddings** from listing descriptions and host responses.

​	•	Utilized **Sentiment Analysis** (VADER) to determine positive/negative tone in host reviews.

​	5.	**Time-based Feature Engineering:**

​	•	Converted date columns into **elapsed days since listing creation** to capture listing age.

​	•	Created **cyclical features** for seasonal patterns (e.g., encoding month as sine/cosine values).

​	6.	**Geospatial Feature Engineering:**

​	•	Used **Haversine distance** to compute proximity to major landmarks (e.g., city center, airports).

​	•	Created **cluster-based neighborhood groups** using K-Means on latitude/longitude.

​	7.	**Scaling & Normalization:**

​	•	Applied **Min-Max Scaling** to numerical variables for models requiring scaled input (e.g., SVM, MLP).

​	•	Used **Power Transformation** (Box-Cox) to handle skewed distributions.

## **Model Selection & Training**

To ensure the best performance, multiple **diverse machine learning models** were tested. A combination of **ensemble learning, regularized regression, and deep learning** was explored:

​	•	**Linear Models:** Ridge Regression, Lasso Regression, ElasticNet

​	•	**Tree-Based Models:** Random Forest, XGBoost, LightGBM, CatBoost

​	•	**Kernel-Based Model:** Support Vector Machine (SVM)

​	•	**Neural Networks:** Multi-layer Perceptron (MLP) with batch normalization & dropout layers

​	•	**Stacking Ensemble:** Combining CatBoost, XGBoost, and Ridge Regression

## **Hyperparameter Tuning**

To optimize each model’s performance, extensive hyperparameter tuning was performed using:

​	•	**Grid Search & Random Search**

​	•	**Bayesian Optimization (Hyperopt)**

​	•	**Automated Hyperparameter Search (Optuna)**

The final **best-performing model** was **CatBoost**, achieving an **R² score of 0.27** on the validation set.

## **Feature Importance Analysis**

To interpret the trained models and understand **which features contribute most to predictions**, I used the following methods:

​	1.	**Model’s Built-in Feature Importance**:

​	•	CatBoost’s feature_importances_ method to rank predictive power.

​	2.	**Permutation Importance**:

​	•	Measured feature impact by shuffling each variable and observing performance drop.

​	3.	**SHAP (SHapley Additive Explanations)**:

​	•	Provided **local & global feature importance visualization**.

​	•	Identified non-linear interactions between variables.

​	4.	**PDP (Partial Dependence Plots) & ICE (Individual Conditional Expectation)**:

​	•	Analyzed how specific features affect model predictions.

​	•	Found that listing price and review score had a strong effect on booking probability.

## **Performance Evaluation**

Final model performances on the **validation set**:

| **Model**     | **R² Score** |
| ------------- | ------------ |
| **CatBoost**  | **0.27**     |
| Random Forest | 0.26         |
| Ridge         | 0.19         |
| Lasso         | 0.21         |
| ElasticNet    | 0.21         |
| SVM           | 0.23         |
| MLP (NN)      | 0.25         |

## **Cross-Validation Strategy**

To ensure **model generalization**, I implemented:

​	•	**5-Fold Cross-Validation** to prevent overfitting.

​	•	**Time-based Split Validation** to mimic real-world forecasting conditions.



