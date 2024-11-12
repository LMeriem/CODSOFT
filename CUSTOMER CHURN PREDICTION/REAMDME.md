# Customer Churn Prediction Using Random Forest and Logistic Regression

This project implements machine learning models to classify bank customers as "churn" or "non-churn" based on various customer attributes. We trained and evaluated two models—a Random Forest classifier and a Logistic Regression classifier—on the **Bank Customer Churn Prediction** dataset from Kaggle. This project includes data preprocessing, model training, evaluation, and a comparison of both models’ performance.

## Project Overview
The objective of this project is to predict customer churn, which can help the bank proactively manage customer retention. By analyzing customer demographics and account information, we use machine learning techniques to classify each customer as likely to churn or not. This project focuses on building and evaluating two models, Random Forest and Logistic Regression, to determine which performs better for this task.

The main steps in this project include:
1. **Data Loading and Cleaning**: Removing unnecessary columns and encoding categorical variables.
2. **Feature Scaling**: Standardizing features to improve model performance.
3. **Model Training**: Training the models with Random Forest and Logistic Regression.
4. **Model Evaluation**: Using accuracy, confusion matrix, precision, recall, and F1-score to evaluate model performance.

## Dataset
The dataset used in this project is the **Bank Customer Churn Prediction dataset** from Kaggle.

- **Download Link**: [Bank Customer Churn Prediction dataset on Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)
  
The dataset includes various features about each customer:
- **Customer Demographics**: Such as geography, age, and gender.
- **Account Information**: Such as balance, tenure, and number of products.
- **Exited**: The target variable indicating if a customer has churned (1) or not (0).

## Installation

### Prerequisites
- **Python 3.x**
- **Libraries**:
  - `pandas`
  - `numpy`
  - `scikit-learn`

### Install Required Libraries
Install the necessary libraries with the following command:
```bash
pip install pandas numpy scikit-learn
```
## Usage

1. **Download the Dataset**: Download the dataset from Kaggle and save it in the project directory as `customer_churn.csv`.

2. **Run the Code**: Execute the Python script in an IDE or terminal to train and evaluate both models.

### Code Explanation

#### Data Loading and Preprocessing
- **Data Loading**: The dataset is loaded from a CSV file.
- **Dropping Unnecessary Columns**: Columns that are not useful for prediction (e.g., `RowNumber`, `CustomerId`, `Surname`) are removed.
- **Encoding Categorical Variables**: `Geography` and `Gender` are transformed into numerical values using `LabelEncoder`.

#### Train-Test Split
The data is split into training and testing sets with an 80-20 split.

#### Feature Scaling
Standard scaling is applied to the features to improve model performance.

### Model Training and Evaluation
Two models are trained:
- **Random Forest Classifier**
- **Logistic Regression Classifier**

Each model's performance is evaluated using:
- **Accuracy Score**: Measures the overall success rate of the model.
- **Confusion Matrix**: Shows the number of correct and incorrect predictions for each class.
- **Classification Report**: Provides precision, recall, and F1-score for both churn and non-churn classes.


### Results
Based on the classification metrics, the **Random Forest** model performs better than **Logistic Regression** in predicting customer churn, particularly in identifying customers likely to churn. This model achieves higher recall and F1-score for the "churn" class, making it more suitable for this classification problem.

### Conclusion
The **Random Forest** model, with its higher recall for churned customers, is recommended for this use case. It provides better overall performance and can help the bank take proactive measures to reduce customer churn.

### Future Improvements
Potential improvements include:
- **Hyperparameter Tuning**: Using Grid Search or Random Search for optimal parameters.
- **Feature Engineering**: Creating new features that might increase model performance.
- **Ensemble Methods**: Combining predictions from multiple models to further enhance accuracy.
