# Spam Detection Using Logistic Regression

This project implements a machine learning model to classify SMS messages as "spam" or "ham" (not spam) using a Logistic Regression classifier. The model is trained and evaluated on the [SMS Spam Collection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle. This project includes data analysis, feature extraction, hyperparameter tuning, and model evaluation.

## Project Overview

The aim of this project is to develop a spam detection model that can automatically classify text messages as spam or ham. Using natural language processing (NLP) techniques and logistic regression, this model achieves accurate classification of SMS messages.

The main steps in the project are:
- Exploratory Data Analysis (EDA) to visualize the distribution of spam and ham messages.
- Preprocessing and feature extraction with TF-IDF.
- Hyperparameter tuning using grid search.
- Model evaluation with metrics like accuracy, precision, recall, and F1-score.

## Dataset

The dataset used in this project is the **SMS Spam Collection** dataset from Kaggle, which contains SMS messages labeled as either `spam` or `ham`. 

- **Download Link**: [SMS Spam Collection dataset on Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

The dataset includes:
- `v1`: The label for each message (`spam` or `ham`).
- `v2`: The text content of the message.

The dataset is preprocessed to remove symbols and convert text to lowercase, improving model performance.

## Installation

### Prerequisites
- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`
  - `wordcloud`

### Install Required Libraries

Install the necessary libraries with the following command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn wordcloud
```

## Usage

1. **Download the Dataset**: Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and save it as `spam.csv` in the project directory.
2. **Run the Code**: Execute the Python code in an IDE or terminal to train and evaluate the model.

## Code Explanation

### Data Loading and Error Handling
- The script reads `spam.csv`, retains only the relevant columns (`v1` and `v2`), and renames them to `label` and `message`. Basic error handling is included for file-related issues.

### Exploratory Data Analysis (EDA)
- **Count Plot**: Displays the distribution of spam vs. ham messages.
- **Word Cloud**: Visualizes the most common words in spam messages, highlighting spam keywords.

### Data Preprocessing
Messages are preprocessed by:
- Converting text to lowercase.
- Removing special characters.

### Train-Test Split
- The data is split into training and testing sets with an 80-20 split.

### TF-IDF Vectorization
- TF-IDF (Term Frequency-Inverse Document Frequency) is used to transform text into numerical features that represent word importance.

### Model Training with Logistic Regression
- A logistic regression model is trained using grid search to optimize the `C` parameter (regularization strength) and `class_weight` (to address imbalanced classes).

### Model Evaluation
The model's performance is evaluated with:
- **Accuracy Score**: Measures the overall success rate of the model.
- **Classification Report**: Provides precision, recall, and F1-score for spam and ham messages.

## Results

Example results:

```yaml
Accuracy: 0.98
Classification Report:
              precision    recall  f1-score   support

         ham       0.99      0.98      0.99      965
        spam       0.96      0.98      0.97      150

    accuracy                           0.98     1115
   macro avg       0.98      0.98      0.98     1115
weighted avg       0.98      0.98      0.98     1115
```

These results indicate that the model accurately distinguishes between spam and ham messages, with strong performance across all metrics.
