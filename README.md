# Sales Forecasting Model

## Overview

This repository contains the code and instructions for a machine learning model that forecasts sales based on various store and product features. The model is built using **Random Forest Regression**, which is a robust and efficient algorithm for regression tasks.

The project uses a dataset that includes features such as store size, product category, price, region, promotion details, and more. The goal is to predict the sales of products in various stores.

## Setup Instructions

### Prerequisites

Before running the code, ensure you have the following libraries installed:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them via `pip`:

- pip install numpy pandas matplotlib seaborn scikit-learn

# Parameters & Model Configuration

The model has several hyperparameters that can be adjusted in the train_model.py file:
a. n_estimators: The number of trees in the forest.
b. max_depth: The maximum depth of each tree.
c. min_samples_split: The minimum number of samples required to split a node.
d. min_samples_leaf: The minimum number of samples required to be at a leaf node.

# Description of the Model Components

The project is broken down into the following components:

1.  Data Preprocessing

- Loads and cleans the dataset.
- Handles missing values and converts categorical variables to numerical format.
- Extracts useful features like holiday weeks, promotions, etc.

2.  Model Training

- Splits the data into training and testing sets.
- Trains the Random Forest model using the processed data.
- Fine-tunes the model using GridSearchCV to find optimal hyperparameters.

3.  Prediction

- Takes input features and predicts the sales using the trained model.
  E

4.  valuation

- Uses evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared to assess model performance.
- Generates feature importance to understand which features are driving predictions.

# Report on Model Performance

1.  Summary of Model Performance
    After training the model using Random Forest Regression, we evaluated its performance using three key metrics:

- Mean Absolute Error (MAE): 6.21
  This value represents the average difference between the predicted and actual sales. The lower the MAE, the better the model's predictions.

- Mean Squared Error (MSE): 129.80
  The MSE gives us an indication of the average squared difference between predicted and actual values. Like MAE, a lower value is better.

- R-squared (R²) Score: -0.34
  The R² value indicates how well the model explains the variance in the sales data. Negative values suggest that the model is not able to explain the variance in the data, which might indicate the need for further improvements in feature selection or model choice.

# Feature Importance

The following features were determined to be the most important in predicting sales:

Feature Importance
a. Store Size (sqft) 0.2167
b. City 0.1623
c. Store ID 0.1152
d. Region 0.1028
e. Price (USD) 0.0868
f. Product ID 0.0776
g. Category 0.0715
h. Stock Level 0.0675
i. Promotion Type 0.0434
j. Promotion Value 0.0415
h. Holiday Affects Sales 0.0148
As seen from the table above, the most important feature for predicting sales is Store Size (sqft), followed by City and Store ID.

# Recommendations for Improving Model Accuracy

To improve the model accuracy in future iterations, consider the following:

1.  Feature Engineering:
    Add more time-based features such as Seasonality (e.g., monthly or quarterly trends).
    Include external factors that could influence sales (e.g., weather data, holidays, events).
    Create new interaction features (e.g., interaction between promotion and price).

2.  Hyperparameter Tuning:
    Experiment with different hyperparameters for the Random Forest model using RandomizedSearchCV or Bayesian Optimization.
    Increase the number of estimators (trees) and adjust max_depth and min_samples_split to fine-tune the model.

3.  Model Choice:
    Consider trying other machine learning models such as Gradient Boosting Machines or XGBoost, which may perform better with non-linear relationships in the data.
    Test using ensemble methods combining multiple models to improve robustness.

4.  Addressing Outliers:
    Handle outliers more effectively by using methods like RobustScaler or Isolation Forest to detect and remove them before training.

5.  Cross-validation:
    Perform cross-validation to ensure the model generalizes well to unseen data and is not overfitting.
    By implementing these recommendations, the accuracy of the sales forecasting model can be significantly improved.
