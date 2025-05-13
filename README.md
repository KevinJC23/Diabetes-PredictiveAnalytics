# Machine Learning Project Report - Kevin Juan Carlos

## Project Domain
Diabetes is the one global chronic disease that will have an impact on life quality and premature death. Based on research conducted by Takashima *et al*. (2024) in *Journal of Epidemiology*, diabetes significantly reduces healthy life expectancy in Japan—both men and women—and is categorized as influenced by risk factors such as blood pressure, body mass index (BMI), and smoking habits. According to the American Diabetes Association by Wild *et al*. (2004), diabetes prevalence sharply increases when people get older, making the older generation more vulnerable to this disease. Moreover, gender also plays a crucial role in the development of diabetes, as biological and hormonal differences contribute to variations in how men and women experience chronic illnesses such as diabetes (Regitz-Zagrosek, 2012). For example, men are more likely to develop diabetes at lower BMI values, while women with hormonal conditions like PCOS are at higher risk. This fact shows that diabetes cannot only cause physical complications but also shorten a person's disability lifespan. Therefore, early detection and intervention are even more essential to maintaining the quality of the community. Advancements in technology have enabled the utilization of Artificial Intelligence, particularly Machine Learning, which can be applied to predict the probability of people having diabetes based on their medical data. This project focuses on predictive analytics and aims to compare Machine Learning model classification to know what models can make the diagnosis process efficient and accurate.

## Business Understanding
### Problem Statements
- How to predict diabetes risks based on the factors which are blood pressure, body mass index (BMI), smoking habits, age and gender?
- How to use Machine Learning for increasing the accuracy to detect the diabetes risks?
- What the most effective algorithm for analyze medical data and predict the probability of people who have diabetes and how these models can improve efficiency for diagnosis process?

### Goals
- Indentify and understand the main risk factors (blood pressure, body mass index (BMI), smoking habits, age and gender) that can influence diabetes prediction and how they can be used on Machine Learning model for accurate diagnosis.
- Build prediction model using medical data and risks factors to give early diagnosis that more efficient and accurate.
- Compare each Machine Learning algorithm to knowing which one can give high accuracy to predict diabetes risk, and optimalize model to get the better result.

### Solution Statements
- Comparing four different Machine Learning algorithms which are Linear Support Vector Classifier (LinearSVC), Linear Regression, Random Forest, and AdaBoost.
- Data preprocessing is done to make sure the input model quality and model can work optimally.
- Doing hyperparameter tuning to find the best parameter combinations and optimize performance for each model using GridSearchCV and RandomizedSearchCV.  

## Data Understanding
### Source
The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) with the name "Diabetes Prediction Dataset". It can be downloaded using kagglehub library with the following code:
```
import kagglehub

path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
```

### Dataset Information
This dataset consists of 100000 entries before data cleaning and is designed to predict the likelihood of someone having diabetes based on the features and lifestyle factors. The dataset includes nine features, one of which is the target label Diabetes, with values defined as follows: 
- 1 = diabetes
- 0 = non-diabetes

### Data Condition
- There are no missing values
- Have 3854 data duplication
- Data imbalance

The dataset consists of eight input features and one target label, which are as follows:
- Age: The range starts from 0 and goes up to 80.
- Gender: Consist of three categories which are male, female, and other.
- Body Mass Index (BMI): The range of BMI in the dataset is from 10.16 to 71.55. (Underweight (BMI < 18.5),  Normal (18.5 ≤ BMI < 24.9), Overweight (25 ≤ BMI < 29.9), and Obese (BMI ≥ 30)).
- Hypertension: It has values of 0 or 1 where 0 indicates indicates they don’t have hypertension and 1 means they have hypertension.
- Heart Disease: It has values of 0 or 1 where 0 indicates they don’t have heart disease and 1 means they have heart disease.
- Smoking History: Consists of five categories which are not current, former, no info, current, never, and ever.
- HbA1c Level: A measure of a person's average blood sugar level over the past 2-3 months, higher levels indicate a greater risk of developing diabetes.
- Blood Glucose Level: Refers to the amount of glucose in the bloodstream at a given time, high levels are a key indicator of diabetes.
- Diabetes: The target variable being predicted, 1 indicates the presence of diabetes and 0 indicates the absence of diabetes.

### Datatypes Overview
This part explained datatypes from each column (features) on dataset. Datatype shows how data stored and processed, example is the data in number (integer, float), text (object), or categorical and also this is become basic foundation to create a better machine learning model. 
| Column               | Non-Null Count | Dtype   |
|----------------------|----------------|---------|
| gender               | 100,000        | object  |
| age                  | 100,000        | float64 |
| hypertension         | 100,000        | int64   |
| heart_disease        | 100,000        | int64   |
| smoking_history      | 100,000        | object  |
| bmi                  | 100,000        | float64 |
| HbA1c_level          | 100,000        | float64 |
| blood_glucose_level  | 100,000        | int64   |
| diabetes             | 100,000        | int64   |

### Datasets Statistic Descriptive
This part displays the statistical summary of each column on the dataset's numeric features.
| Statistic | age       | hypertension | heart_disease  | bmi       | HbA1c_level | blood_glucose_level  | diabetes |
|-----------|-----------|--------------|----------------|-----------|-------------|----------------------|----------|
| count     | 100000.00 | 100000.00    | 100000.00      | 100000.00 | 100000.00   | 100000.00            | 100000.00|
| mean      | 41.89     | 0.07         | 0.04           | 27.32     | 5.53        | 138.06               | 0.09     |
| std       | 22.52     | 0.26         | 0.19           | 6.64      | 1.07        | 40.71                | 0.28     |
| min       | 0.08      | 0.00         | 0.00           | 10.01     | 3.50        | 80.00                | 0.00     |
| 25%       | 24.00     | 0.00         | 0.00           | 23.63     | 4.80        | 100.00               | 0.00     |
| 50%       | 43.00     | 0.00         | 0.00           | 27.32     | 5.80        | 140.00               | 0.00     |
| 75%       | 60.00     | 0.00         | 0.00           | 29.58     | 6.20        | 159.00               | 0.00     |
| max       | 80.00     | 1.00         | 1.00           | 95.69     | 9.00        | 300.00               | 1.00     |

### Data Distribution Visualization
This part presenting the visualization using bar chart or pie chart from distribution in each categorical feature.

- #### Bar Chart for Gender Distribution
![Gender Distribution](src/gender.png)

This histogram shows that the number of females (approximately 60%) significantly exceeds the number of males (approximately 40%). Additionally, there is a small number of 'Other' labels, and it doesn't provide significant insight for modelling. For this reason, it might be removed from the dataset to simplify the analysis and help reduce noise and increase model interpretability. After removal, the data remains well-distributed between males and females, eliminating the need for additional balancing.

- #### Bar Chart for Smoking History Distribution
![Smoking History Distribution](src/smoking_history.png)

This histogram shows that the majority of data for 'No Info' labels (approximately 70%), 'Never' labels (approximately 70%), and the other label that represents a relatively small portion of data is distributed among the categories 'former', 'current', 'not current', and 'ever' (approximately 30%). Since the 'Never' labels are still essential for the analysis, it should be retained. However, the 'No Info' labels lack of interpretive value, it can removed to eliminate noise and enhance data quality. After doing this preprocessing, it could help to achieve a more balanced distribution, improved model performance, and interpretability.

#### Pie Chart for Hypertension Distribution
![Hypertension Distribution](src/hypertension.png)

#### Pie Chart for Heart Disease Distribution
![Heart Disease Distribution](src/heart_disease.png)

## Data Preparation
To make the model perform better and produce more reliable predictions, a thorough data preparation process was conducted. This step is crucial to ensure that the dataset is clean, consistent, and suitable for machine learning algorithms. The following techniques were applied in the notebook, in the order they were executed:
- Dropping Duplicate Records: Duplicate rows in the dataset can skew the model by overrepresenting certain patterns. To ensure that each instance in the training data is unique and to prevent bias and overfitting, duplicate entries were identified and removed.
  
- Removing Irrelevant Values: Some values in the dataset may be irrelevant or too rare to contribute meaningful statistical insight. These values were examined and removed to improve data quality and model learning efficiency.
  
- Label Encoding: Categorical features such as `gender` and `smoking_history` were encoded using **Label Encoding**, converting string labels into numeric form. This step is necessary because most machine learning algorithms require numerical input. Label encoding allows these categorical features to be used effectively in the modeling process while preserving ordinal relationships, if any.

- Outliers handling: Outliers in numerical features like `bmi`, `HbA1c_level`, and `blood_glucose_level` were detected and treated using the **Interquartile Range (IQR)** method. Outliers can introduce noise and distort the learning process, reducing model accuracy. By managing these outliers, the model’s generalization performance is improved.

- Normalization: Numerical features including `age`, `bmi`, `HbA1c_level`, and `blood_glucose_level` were scaled using **StandardScaler**. Normalization ensures that all features lie within the same scale range (typically 0 to 1), which is especially important for gradient-based optimization algorithms like Logistic Regression

- Spliting the Dataset: Before model training, the dataset was split into training and testing sets:
  - **Test Size**: 20% of the data was allocated for testing to ensure a reliable evaluation.
  - **Stratification**: The split was stratified based on the target variable (`diabetes`) to preserve its distribution in both sets, which is crucial for imbalanced datasets.
  - **Random State**: A fixed random state was used to make the split reproducible.
  
Each of these steps plays a vital role in transforming the raw data into a structured and clean form, enabling the machine learning model to learn patterns effectively and make accurate predictions.

## Modeling
To solve the classification problem of predicting diabetes, we experimented with four different machine learning models. Each model underwent hyperparameter tuning using GridSearchCV to achieve optimal performance. Below is a detailed explanation of the modeling process, evaluation, and reasoning behind the model selection.
- Models Used:
  ### Linear Support Vector Classifier (LinearSVC)
  - Hyperparameters Tuned:
    - C: Controls the regularization strength (smaller values → stronger regularization)
    - max_iter: Sets the maximum number of iterations before the algorithm stops
  - Pros:
    - Efficient for high-dimensional datasets
    - Fast training time on smaller datasets
  - Cons:
    - Assumes data is linearly separable
    - Sensitive to feature scaling and may not perform well with unscaled or noisy data
  ### Linear Regression
  - Hyperparameters Tuned:
    - C: Inverse of regularization strength (higher value = less regularization)
    - solver: Optimization algorithm used (liblinear for small datasets, lbfgs for larger ones)
  - Pros:
    - Simple and easy to interpret
    - Performs well as a baseline model
    - Less prone to overfitting with proper regularization
  - Cons:
    - Assumes a linear relationship between features and the target
    - May struggle with complex or non-linear data distributions
  ### Random Forest
  - Hyperparameters Tuned:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum depth each tree can grow
    - min_samples_split: Minimum number of samples required to split a node
  - Pros:
    - Captures non-linear relationships and feature interactions effectively
    - Robust to outliers, noise, and missing values
    - Works with both categorical and numerical features without scaling
  - Cons:
    - Training can be slower with a large number of trees
    - Harder to interpret compared to linear models
   ### AdaBoost Classifier (Boosting)
   - Hyperparameters Tuned:
     - n_estimators: Number of boosting stages
     - learning_rate: Weight reduction for each weak learner's contribution
   - Pros:
     - Increases accuracy by combining weak learners
     - Works well on imbalanced datasets
     - Often improves over simple models with minimal tuning
   - Cons:
     - Sensitive to outliers and noisy data
     - Can overfit if the number of estimators is too high or learning rate too low

- Model Performance Summary:

| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|----------------|
| LinearSVC          | 0.9481         | 0.9473         |
| LogisticRegression | 0.9476         | 0.9471         |
| RandomForest       | 0.9638         | 0.9631         |
| Boosting (AdaBoost)| 0.9627         | 0.9630         |

After evaluating all models based on test accuracy, the best performing model was selected as the final solution, Random Forest yielded the highest test accuracy with minimal overfitting and strong performance across both training and test data. It also handles both categorical and numerical features effectively without requiring feature scaling.

## Evaluation
### Classification Report
This report presents the evaluation metrics of each model's performance by summarizing the following: 
- **Precision**: Measures the accuracy of positive predictions made by the model. This metric indicates the proportion of correctly predicted positive instances out of all predicted as positive.
- **Recall**: Measures the ability of the model to identify positive instances. This metric indicates how many propositions from all of the positive examples that successfully detected by the model.
- **F1-score**: The harmonic mean of both precision and recall, balancing both metrics into a single value. It's useful when dealing with a dataset that has an imbalanced class distribution.
- **Support**: The number of actual instances of each class in the dataset gives a view of the class proportion that was evaluated. Therefore, we can understand the data distribution on that model.

Here are the results of the comparison analysis from each model:
- #### LinearSVC

```
              precision    recall  f1-score   support

           0       0.95      0.99      0.97     11243
           1       0.87      0.62      0.72      1407

    accuracy                           0.95     12650
   macro avg       0.91      0.80      0.85     12650
weighted avg       0.94      0.95      0.94     12650
```
Based on this evaluation result, Class 0 (Non-diabetes) demonstrates 95% Precision, indicating it correctly predicted the non-diabetes case. Additionally, the Recall for the same class is 99%, meaning that the model accurately identified 99% of all non-diabetes instances, and the F1-Score for this class is 97%, which indicates a strong performance in identifying non-diabetes cases. For Class 1 (Diabetes), demonstrate 87% for the Precision, signifying it predict cases were correct. However, the Recall was 62%, indicating that the model was able to identify only 62% of the actual diabetes cases. Class 1 F1-Score demonstrates 72%, which shows a balanced measure of Precision and Recall, though it is relatively lower compared to Class 0. Overall, the model achieved an accuracy of 95%, a macro average F1-Score of 0.85, and a weighted average F1-Score of 0.94, highlighting its effectiveness in identifying non-diabetes cases while showing some limitations in correctly detecting diabetes.

- #### Logistic Regression

```
              precision    recall  f1-score   support

           0       0.96      0.99      0.97     11243
           1       0.85      0.63      0.73      1407

    accuracy                           0.95     12650
   macro avg       0.90      0.81      0.85     12650
weighted avg       0.94      0.95      0.94     12650
```
Similar to LinearSVC, Logistic Regression also performs well in classifying Class 0 (Non-diabetes) cases, achieving 96% for Precision, indicating it correctly predicted the non-diabetes case. Additionally, the Recall was also notably high at 99%, meaning the model successfully identified all Class 0 (Non-diabetes) instances. The F1-Score for this class is 97%, reflecting a strong balance between Precision and Recall. Class 1 (Diabetes) shows a precision of 85% of the predicted diabetes cases being accurate. The Recall for class 1 (Diabetes) is 63%, which is slightly better than LinearSVC's Recall of 62%, indicating that the model was able to accurately detect 63% of the real diabetes cases. The F1-score for this class is 73%, indicating an improved balance between precision and recall compared to LinearSVC. Overall, the model achieved an accuracy of 95%, a macro average F1-score of 0.85, and a weighted average F1-score of 0.94.

#### Random Forest

```
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     11243
           1       1.00      0.67      0.80      1407

    accuracy                           0.96     12650
   macro avg       0.98      0.83      0.89     12650
weighted avg       0.96      0.96      0.96     12650
```

- **Precision for Class 0**: 96%, showing it makes accurate predictions for non-diabetes.
- **Recall for Class 0**: 100%, demonstrating that Random Forest correctly identifies all non-diabetes cases.
- **Precision for Class 1**: 100%, which is excellent, but this comes at the cost of lower recall (67%) for diabetes. This means Random Forest is more confident when predicting diabetes, but it may miss some cases.

#### AdaBoost (Boosting)


```
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     11243
           1       1.00      0.67      0.80      1407

    accuracy                           0.96     12650
   macro avg       0.98      0.83      0.89     12650
weighted avg       0.96      0.96      0.96     12650
```

- AdaBoost also shows very high precision for non-diabetes (96%) and perfect recall for non-diabetes (100%).
- Similar to Random Forest, it shows a high precision (100%) for diabetes, but lower recall (67%).

### Confusion Matrix

The **confusion matrix** helps visualize the performance of a classification model by displaying the true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN).

- **True Positive (TP)**: Correctly predicted positive outcomes (diabetes).
- **False Positive (FP)**: Incorrectly predicted positive outcomes (non-diabetes predicted as diabetes).
- **True Negative (TN)**: Correctly predicted negative outcomes (non-diabetes).
- **False Negative (FN)**: Incorrectly predicted negative outcomes (diabetes predicted as non-diabetes).

**Confusion matrices** provide insights into the type of errors each model makes and are crucial for understanding trade-offs between precision and recall.

- **LinearSVC Confusion Matrix**
  
  ![LinearSVC Confusion Matrix](src/LinearSVC.png)
  
- **Logistic Regression Confusion Matrix**
  
  ![Logistic Regression Confusion Matrix](src/LogisticRegression.png)
  
- **Random Forest Confusion Matrix**
  
  ![Random Forest Confusion Matrix](src/RandomForest.png)
  
- **AdaBoost Confusion Matrix**
  
  ![AdaBoosting Confusion Matrix](src/Boosting.png)

### ROC-AUC

The **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** is a performance measurement for classification problems. It shows the trade-off between **True Positive Rate (Recall)** and **False Positive Rate** across different thresholds.

- **ROC Curve**: A plot of the true positive rate (recall) against the false positive rate.
- **AUC (Area Under Curve)**: A measure of how well the model distinguishes between classes. The higher the AUC, the better the model’s ability to differentiate between the positive and negative classes.

![ROC-AUC](src/ROC-AUC.png)

## References
Wild, S., Roglic, G., Green, A., Sicree, R., & King, H. (2004). Global prevalence of diabetes. Diabetes Care, 27(5), 1047–1053. https://doi.org/10.2337/diacare.27.5.1047

Regitz-Zagrosek, V. (2012). Sex and gender differences in health. EMBO Reports, 13(7), 596–603. https://doi.org/10.1038/embor.2012.87

Tsukinoki, R., Murakami, Y., Hayakawa, T., Kadota, A., Harada, A., Kita, Y., Okayama, A., Miura, K., Okamura, T., & Ueshima, H. (2025). Comprehensive assessment of the impact of blood pressure, body mass index, smoking, and diabetes on healthy life expectancy in Japan: NIPPON DATA90. Journal of Epidemiology. https://doi.org/10.2188/jea.je20240298
