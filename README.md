# Machine Learning Project Report - Kevin Juan Carlos

## Project Domain
Diabetes is the one global chronic disease that will have an impact on life quality and premature death. Based on research conducted by Takashima *et al*. (2024) in *Journal of Epidemiology*, diabetes significantly reduces healthy life expectancy in Japan—both men and women—and is categorized as influenced by risk factors such as blood pressure, body mass index (BMI), and smoking habits. Furthermore, according to the American Diabetes Association by Wild *et al*. (2004), diabetes prevalence sharply increases when people get older, making the older generation more vulnerable to this disease. Additionally, gender also plays a crucial factor in diabetes development. According to Regitz-Zagrosek (2012), biological and hormonal differences contribute to variations in how men and women experience chronic illnesses such as diabetes. For example, men are more likely to develop diabetes at lower BMI values, while women with hormonal conditions like PCOS are at higher risk. This fact shows diabetes cannot only cause physical complications but also shorten a person's disability life span. Therefore, early detection and intervention are even more essential to maintaining the quality of the community.

Advancements in technology have enabled the utilization of Artificial Intelligence, particularly Machine Learning, which can be applied to predict the probability of people having diabetes based on their medical data. This project focuses on predictive analytics and aims to compare Machine Learning model classification to know what models can make the diagnosis process efficient and accurate.

## Business Understanding
### Problem Statements
- How to predict diabetes risks based on the factors which is blood pressure, body mass index (BMI), smoking habits, age and gender?
- How to use Machine Learning for increasing the accuracy to detect the diabetes risks?
- What the most effective algorithm for analyze medical data and predict the probability of people who have diabetes and how these models can improve efficiency for diagnosis process?

### Goals
- Indentify and understand the main risk factors (blood pressure, body mass index (BMI), smoking habits, age and gender) that can influence diabetes prediction and how they can be used on Machine Learning model for accurate diagnosis.
- Build prediction model using medical data and risks factors to give early diagnosis that more efficient and accurate.
- Compare each Machine Learning algorithm to knowing which one can give high accuracy to predict diabetes risk, and optimalize model to get the better result.

### Solution Statements
- Comparing four different Machine Learning algorithms which is Linear Support Vector Classifier (LinearSVC), Linear Regression, Random Forest, and AdaBoost 
- Doing Data Preprocessing to make sure input model quality and model can working optimally
- Doing hyperparameter tuning to find the best parameter combinatins and optimize performance for each model using Grid Search and Random Search CV   

## Data Understanding
Dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) using kagglehub:
<pre>
  import kagglehub
  
  path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
</pre>

Dataset Information
- Consist of 100000 lines (before data cleaning)
- Have 9 feature include 1 labels
- Target label: Diabetes (1 = diabetes, 0 = non-diabetes)

Data Condition
- There is no missing values
- have 3854 data duplication
- Data imbalance

The Dataset have 8 features and 1 label that consist of:
- Age: There are three categories in it male ,female and other.
- Gender: Ranges from 0-80
- Body Mass Index (BMI): The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese.
- Hypertension: It has values a 0 or 1 where 0 indicates they don’t have hypertension and for 1 it means they have hypertension.
- Heart Disease: It has values a 0 or 1 where 0 indicates they don’t have heart disease and for 1 it means they have heart disease.
- Smoking History: have 5 categories i.e not current,former,No Info,current,never and ever.
- HbA1c Level: Higher levels indicate a greater risk of developing diabetes.
- Blood Glucose Level: High blood glucose levels are a key indicator of diabetes.
- Diabetes: with values of 1 indicating the presence of diabetes and 0 indicating the absence of diabetes.

More detailed variables on Diabetes Dataset:
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

Show the summary of descriptive statistic
| Statistic | age       | hypertension | heart_disease | bmi       | HbA1c_level | blood_glucose_level | diabetes |
|-----------|-----------|--------------|----------------|-----------|-------------|----------------------|----------|
| count     | 100000.00 | 100000.00    | 100000.00      | 100000.00 | 100000.00   | 100000.00            | 100000.00|
| mean      | 41.89     | 0.07         | 0.04           | 27.32     | 5.53        | 138.06               | 0.09     |
| std       | 22.52     | 0.26         | 0.19           | 6.64      | 1.07        | 40.71                | 0.28     |
| min       | 0.08      | 0.00         | 0.00           | 10.01     | 3.50        | 80.00                | 0.00     |
| 25%       | 24.00     | 0.00         | 0.00           | 23.63     | 4.80        | 100.00               | 0.00     |
| 50%       | 43.00     | 0.00         | 0.00           | 27.32     | 5.80        | 140.00               | 0.00     |
| 75%       | 60.00     | 0.00         | 0.00           | 29.58     | 6.20        | 159.00               | 0.00     |
| max       | 80.00     | 1.00         | 1.00           | 95.69     | 9.00        | 300.00               | 1.00     |

Visualize Gender Distribution using bar chart

![Gender Distribution](src/gender.png)

Visualize Hypertension Distribution using pie chart

![Hypertension Distribution](src/hypertension.png)

Visualize Heart Disease Distribution using pie chart

![Heart Disease Distribution](src/heart_disease.png)

Visualize Smoking History Distribution using bar chart

![Smoking History Distribution](src/smoking_history.png)

## Data Preparation
To make the model perform better and produce more reliable predictions, a thorough data preparation process was conducted. This step is crucial to ensure that the dataset is clean, consistent, and suitable for machine learning algorithms. The following techniques were applied in the notebook, in the order they were executed:
- Dropping Duplicate Records
  - Duplicate rows in the dataset can skew the model by overrepresenting certain patterns. Therefore, we used:
    <pre></pre>
    Reason: Removing duplicates ensures that each instance in the training data is unique, avoiding bias in learning and helping to prevent overfitting.
- Label Encoding
  - Categorical features such as gender and smoking_history were encoded using label encoding to convert string labels into numeric form.
    <pre></pre>
    Reason: Most machine learning models only accept numerical input. Label encoding enables the use of categorical data in these models while maintaining the order of categories if needed.
- Outliers handling
  - Outliers were detected and treated using IQR (Interquartile Range) method on numerical features like bmi, HbA1c_level, and blood_glucose_level.
    <pre></pre>
    Reason: Outliers can introduce noise and distort the learning process, leading to lower model accuracy. By handling them, we improve the model's ability to generalize.
- Normalization
  - Numerical features such as age, bmi, HbA1c_level, and blood_glucose_level were scaled using MinMaxScaler:
    <pre></pre>
    Reason: Normalization ensures that features are on the same scale, preventing models from favoring features with larger magnitudes. This is especially important for distance-based models and gradient-based optimization.
- Spliting the Dataset
  Before training the model, we split the dataset into training and testing sets:
  <pre></pre>
  Reason:
  - Data splitting is essential to evaluate how well the machine learning model generalizes to unseen data.
  - Stratify is used to ensure the distribution of the target variable (diabetes) is preserved in both training and testing sets, which is particularly important when dealing with imbalanced datasets.
  - A 20% test size was selected to provide enough data for a reliable evaluation while maintaining sufficient data for training.
  - A random state was fixed to ensure the results are reproducible.

Each of these steps plays a vital role in transforming the raw data into a structured and clean form, enabling the machine learning model to learn patterns effectively and make accurate predictions.
## Modeling
To solve the classification problem of predicting diabetes, we experimented with four different machine learning models. Each model underwent hyperparameter tuning using GridSearchCV to achieve optimal performance. Below is a detailed explanation of the modeling process, evaluation, and reasoning behind the model selection.
- Models Used:
  - Linear Support Vector Classifier (LinearSVC)
  - Linear Regression
  - Random Forest
  - AdaBoost Classifier (Boosting)
- Modeling Process & Parameters
  Each model was tuned using GridSearchCV with 5-fold cross-validation and the accuracy metric as the scoring method. The following parameter grids were used:
  - LinearSVC
    - C: Regularization parameter (smaller → stronger regularization)
    - max_iter: Maximum number of iterations for convergence
  - LogisticRegression
    - C: Inverse of regularization strength
    - solver: Optimization algorithm (liblinear for small datasets, lbfgs for larger datasets)
  - RandomForestClassifier
    - n_estimators: Number of trees
    - max_depth: Maximum depth of the trees
    - min_samples_split: Minimum samples required to split an internal node
  - AdaBoostClassifier
    - n_estimators: Number of boosting stages
    - learning_rate: Shrinks contribution of each classifier

Each model was trained on the training set, and accuracy was evaluated on both the training and test sets.

- Model Performance Summary
  
| Model               | Train Accuracy | Test Accuracy |
|--------------------|----------------|----------------|
| LinearSVC          | 0.9481         | 0.9473         |
| LogisticRegression | 0.9476         | 0.9471         |
| RandomForest       | 0.9638         | 0.9631         |
| Boosting (AdaBoost)| 0.9627         | 0.9630         |

- Pros & Cons of Each Algorithm
  - LinearSVC
    - Pros:
      - Works well on high-dimensional data
      - Fast training on smaller datasets
    - Cons:
      - Not effective on non-linearly separable data
      - Sensitive to feature scaling
  - Logistic Regression
    - Pros:
      - Simple and interpretable
      - Good baseline model
    - Cons:
      - Assumes linear relationship
      - Limited power on complex datasets
  - Random Forest
    - Pros:
      - Handles non-linearity and interactions well
      - Robust to outliers and noise
    - Cons:
      - Can be slow with large number of trees
      - Less interpretable
  - AdaBoost (Boosting)
    - Pros:
      - Improves accuracy by combining weak learners
      - Works well on imbalanced datasets
    - Cons:
      - Sensitive to noisy data
      - Can overfit if not properly tuned

After evaluating all models based on test accuracy, the best performing model was selected as the final solution, Random Forest yielded the highest test accuracy with minimal overfitting and strong performance across both training and test data. It also handles both categorical and numerical features effectively without requiring feature scaling.

## Evaluation
### Classification Report for LinearSVC:

```
              precision    recall  f1-score   support

           0       0.95      0.99      0.97     11243
           1       0.87      0.62      0.72      1407

    accuracy                           0.95     12650
   macro avg       0.91      0.80      0.85     12650
weighted avg       0.94      0.95      0.94     12650
```

---

### Classification Report for LogisticRegression:

```
              precision    recall  f1-score   support

           0       0.96      0.99      0.97     11243
           1       0.85      0.63      0.73      1407

    accuracy                           0.95     12650
   macro avg       0.90      0.81      0.85     12650
weighted avg       0.94      0.95      0.94     12650
```

---

### Classification Report for RandomForest:

```
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     11243
           1       1.00      0.67      0.80      1407

    accuracy                           0.96     12650
   macro avg       0.98      0.83      0.89     12650
weighted avg       0.96      0.96      0.96     12650
```

---

### Classification Report for Boosting:

```
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     11243
           1       1.00      0.67      0.80      1407

    accuracy                           0.96     12650
   macro avg       0.98      0.83      0.89     12650
weighted avg       0.96      0.96      0.96     12650
```

![LinearSVC Confusion Matrix](src/LinearSVC.png)
![Logistic Regression Confusion Matrix](src/LogisticRegression.png)
![Random Forest Confusion Matrix](src/RandomForest.png)
![AdaBoosting Confusion Matrix](src/Boosting.png)

![ROC-AUC](src/ROC-AUC.png)

## References
Wild, S., Roglic, G., Green, A., Sicree, R., & King, H. (2004). Global prevalence of diabetes. Diabetes Care, 27(5), 1047–1053. https://doi.org/10.2337/diacare.27.5.1047

Regitz-Zagrosek, V. (2012). Sex and gender differences in health. EMBO Reports, 13(7), 596–603. https://doi.org/10.1038/embor.2012.87

Tsukinoki, R., Murakami, Y., Hayakawa, T., Kadota, A., Harada, A., Kita, Y., Okayama, A., Miura, K., Okamura, T., & Ueshima, H. (2025). Comprehensive assessment of the impact of blood pressure, body mass index, smoking, and diabetes on healthy life expectancy in Japan: NIPPON DATA90. Journal of Epidemiology. https://doi.org/10.2188/jea.je20240298
