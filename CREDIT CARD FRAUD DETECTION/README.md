# Credit Card Fraud Detection

## Problem Statement
In today's digital world, credit card fraud is a significant issue that affects consumers and financial institutions alike. The objective of this project is to build a machine learning model that accurately detects fraudulent credit card transactions from a dataset containing transaction information. By classifying transactions as fraudulent or legitimate, we aim to minimize financial losses and enhance security measures.

## Dataset
The dataset used for this project can be downloaded from Kaggle:
- [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection?resource=download&select=fraudTrain.csv)

The dataset contains various features related to credit card transactions, including:
- Transaction details
- Customer demographics
- Information on whether the transaction was fraudulent

## Solution Approach
1. **Data Preprocessing**: 
   - Load and clean the dataset.
   - Convert categorical variables into numeric format using one-hot encoding.
   - Handle missing values using imputation methods.
   - Extract relevant time features from the transaction timestamp.

2. **Balancing the Dataset**:
   - Since fraudulent transactions are rare, apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset, ensuring that the model learns effectively from both classes.

3. **Model Selection**:
   - Experiment with various classification algorithms, including:
     - Logistic Regression
     - Decision Tree Classifier
     - Random Forest Classifier

4. **Model Evaluation**:
   - Use metrics such as accuracy, precision, recall, and F1-score to evaluate model performance.
   - Implement Grid Search for hyperparameter tuning, particularly for the Random Forest model.

5. **Model Interpretation**:
   - Analyze model predictions and the importance of various features to understand factors contributing to fraudulent transactions.

## Steps and Explanation
1. **Import Libraries**: 
   - Import necessary libraries for data manipulation (Pandas, NumPy), model building (scikit-learn, XGBoost), and handling imbalanced datasets (imblearn).

2. **Load and Explore Data**: 
   - Load the dataset from a CSV file and perform exploratory data analysis (EDA) to understand the structure and distribution of the data.

3. **Preprocessing**:
   - Clean column names, convert date features, drop unnecessary columns, and apply one-hot encoding to categorical features.
   - Handle missing values and apply SMOTE to create a balanced dataset.

4. **Split the Data**: 
   - Divide the dataset into training and testing sets to evaluate model performance.

5. **Standardization**: 
   - Standardize the feature values for better model performance, especially for algorithms like Logistic Regression.

6. **Model Training and Prediction**:
   - Train different models on the training set and evaluate their performance on the test set using various classification metrics.

7. **Hyperparameter Tuning**:
   - Use Grid Search with cross-validation to optimize the hyperparameters of the Random Forest classifier.

8. **Evaluation**:
   - Generate classification reports to summarize model performance.

## Requirements
To run this project, you will need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `xgboost`

### Installation
You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost
```

##Conclusion
This project demonstrates the effectiveness of machine learning techniques in detecting fraudulent credit card transactions. The models developed can serve as a foundation for further enhancements, such as incorporating more complex algorithms, feature engineering, or using additional datasets to improve accuracy.
