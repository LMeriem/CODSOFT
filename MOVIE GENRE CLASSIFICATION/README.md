# Movie Genre Classification

## Problem Statement
In the world of entertainment, accurately identifying the genre of a movie based on its plot is valuable for categorizing content, improving search algorithms, and enhancing user recommendations. The goal of this project is to create a machine learning model that classifies movie plots into their respective genres. This project leverages text preprocessing, TF-IDF vectorization, and multiple classification algorithms to predict movie genres with improved accuracy.

## Dataset
The dataset used for this project can be downloaded from Kaggle:

[Genre Classification Dataset (IMDb)](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb)

The dataset contains information about movies, including:
- **Genre**: The target variable we want to classify.
- **Plot**: A text description of the movie's storyline.

Each line in the dataset files contains fields separated by `" ::: "` in the format: `movie_id ::: title ::: genre ::: plot`.

## Solution Approach

### 1. Data Preprocessing
- **Data Loading**: Load the dataset from text files and split each line into respective columns.
- **Text Cleaning**: Apply preprocessing techniques such as converting text to lowercase, removing special characters, and eliminating extra spaces to ensure clean input data for the model.

### 2. Feature Extraction with TF-IDF
- **TF-IDF Vectorization**: Extract important features from the movie plots using Term Frequency-Inverse Document Frequency (TF-IDF). This approach transforms text data into numerical values, allowing machine learning models to interpret plot descriptions effectively.

### 3. Model Selection and Training
We trained and evaluated the following classification models:
- **Logistic Regression**: A linear model that can handle multi-class classification and offers interpretability.
- **Support Vector Machine (SVM)**: A powerful classifier that aims to find the best separation between classes.

### 4. Model Evaluation
- **Train/Test Split**: Split the training data into training and validation sets to monitor model performance and avoid overfitting.
- **Evaluation Metrics**: Use accuracy, precision, recall, and F1-score to evaluate the model's performance.
  
### 5. Prediction on Test Data
- The best-performing model is applied to predict genres for the test dataset. These predictions are saved to a CSV file for further analysis or submission.

## Steps and Explanation

### Import Libraries
Import essential libraries for data manipulation (`pandas`, `re`), feature extraction (`TfidfVectorizer`), and model building and evaluation (`LogisticRegression`, `SVC`, `accuracy_score`, `classification_report`).

### Load and Explore Data
- Load the training and test datasets from the specified files.
- Convert plot descriptions into lowercase and remove special characters to standardize input text.

### Preprocessing
- Clean the plot text using regular expressions.
- Apply TF-IDF vectorization on the preprocessed text to transform it into a numerical format suitable for modeling.

### Model Training and Prediction
- Train the Logistic Regression and SVM models on the training data.
- Evaluate each model on the validation set using metrics like accuracy and classification report to determine the best-performing model.

### Evaluation on Test Data
- Use the selected model to predict genres on the test data.
- Save predictions to `predicted_test_data.csv` for submission or further review.

## Requirements
The following libraries are required to run this project:
- pandas
- scikit-learn
- re (regular expressions)

### Installation
Install the required libraries using pip:
```bash
pip install pandas scikit-learn
```

##Conclusion
This project showcases the use of text classification techniques to predict movie genres based on plot descriptions. The models developed here can serve as a baseline for more advanced solutions, such as incorporating deep learning models or using additional text data to improve classification accuracy. This classification model could be used in various recommendation systems or movie databases to help users find content that matches their interests.
