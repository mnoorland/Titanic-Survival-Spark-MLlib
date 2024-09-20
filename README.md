# Titanic Survival Prediction Using Spark MLlib

## Project Overview
This project involves building machine learning models using the Titanic dataset to predict passenger survival. The analysis included extensive data cleaning and exploratory analysis, followed by applying Spark MLlib pipelines for classification. The models were evaluated and compared using three different algorithms: Logistic Regression, Support Vector Machine, and Random Forest. The project demonstrates the power of Spark's distributed computing capabilities in optimizing model performance through hyperparameter tuning.

## Objectives
- **Predict passenger survival** based on Titanic dataset features.
- **Test multiple classification algorithms** including Logistic Regression, Support Vector Machine (SVM), and Random Forest.
- **Optimize model performance** using Spark MLlib for hyperparameter tuning and parallel processing.

## Key Features
- **Data Cleaning and Processing**: Performed extensive data cleaning to handle missing values, and feature engineering to improve model performance.
- **Spark MLlib Pipelines**: Used Spark pipelines to streamline the machine learning workflow.
- **Model Comparisons**: Compared the performance of Logistic Regression, SVM, and Random Forest, evaluating based on accuracy and other metrics.
- **Hyperparameter Tuning**: Applied hyperparameter tuning using Spark’s parallel processing to find the best model parameters.

## Tools & Technologies
- **Apache Spark (MLlib)**: For building, training, and tuning machine learning models in a distributed environment.
- **Jupyter Notebook**: For documenting the analysis and running the code.
- **Python & Pandas**: Used for data processing and feature engineering.
  
## Files Included
- **[Jupyter Notebook](./Cloud_Cognitive_Max_Diego_Jacopo.ipynb)**: Contains the full code for data cleaning, feature engineering, model building, and evaluation.
- **[Logistic Regression Predictions](./predictions_logistic_regression_spark.csv)**: CSV file with predictions from the Logistic Regression model.
- **[SVM Predictions](./predictions_svm_spark.csv)**: CSV file with predictions from the Support Vector Machine model.
- **[Random Forest Predictions](./predictions_spark_random_forest_classifier.csv)**: CSV file with predictions from the Random Forest model.

## Dataset
The dataset includes:
- **Passenger Details**: Such as age, gender, fare, cabin class, and family size.
- **Survival Outcome**: Whether the passenger survived or not (target variable).
