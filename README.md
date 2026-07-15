## Gender Classification from Names
This project predicts gender (male or female) from a person's name using Machine Learning models built from scratch, without using scikit-learn.

## What this project does
- Takes a dataset of names with their gender label
- Cleans the data (removes duplicates, empty rows, invalid names)
- Builds features from each name (like first letter, last letter, name length)
- Trains 3 ML models from scratch:
   - Naive Bayes
   - Logistic Regression
   - K-nearest Neighbors (KNN)
- Compares accuracy of all 3 models
- Shows charts using Matplotlib
- Predicts gender for any new name

## Dataset
- File: 'Names_dataset.csv'
- Columns: 'name' and 'gender' (values: 'm' and 'f')
- Source: Kaggle (Indian and English Names dataset)
- Shape before cleaning: 125231
- Shape after cleaning: 98240 (male: 36153 and female: 62087)

## Models (Build from Scratch)
No scikit-learn is used. All models are written using only Numpy:
1. Naive Bayes: uses probability of letters for each gender.
2. Logistic Regression: uses gradient descent to learn weights.
3. KNN: finds the closest matching names in the training data.

## Results (accuracy):
1.Naive Bayes: 82.28%
2.Logistic Regression: 81.29%
3.KNN: 80.47%

## Train-test split (80%-20%):
- Training data: 78592
- Testing data: 19648

Note: The dataset is imbalanced (more female names than male names), so accuracy should always be compared against a baseline (guessing the majority class). 
