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
- Comparing four different Machine Learning algorithms which are Linear Support Vector Classifier (LinearSVC), Linear Regression, Random Forest, and AdaBoost.
- Data preprocessing is done to make sure the input model quality and model can work optimally.
- Doing hyperparameter tuning to find the best parameter combinations and optimize performance for each model using GridSearchCV and RandomizedSearchCV.  

## Data Understanding
### Source
The dataset used in this project was obtained from [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) with the name "Diabetes Prediction Dataset". It can be downloaded using kagglehub library, this following code can be used to download the dataset:
<pre>
  import kagglehub
  
  path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
</pre>

### Dataset Information
This dataset consists of 100000 entries before data cleaning and is designed to predict the likelihood of someone having diabetes based on the features and lifestyle factors. The dataset includes nine features, one of which is the target label Diabetes, with values defined as follows: 
- 1 = diabetes
- 0 = non-diabetes

### Data Condition
- There are no missing values
- Have 3854 data duplication
- Data imbalance

The dataset consists of eight input features and one target label, which are as follows:
- Age: There are three categories in it male, female, and other.
- Gender: Ranges from 0-80
- Body Mass Index (BMI): The range of BMI in the dataset is from 10.16 to 71.55. BMI less than 18.5 is underweight, 18.5-24.9 is normal, 25-29.9 is overweight, and 30 or more is obese.
- Hypertension: It has values of 0 or 1 where 0 indicates they don’t have hypertension and 1 means they have hypertension.
- Heart Disease: It has values of 0 or 1 where 0 indicates they don’t have heart disease and 1 means they have heart disease.
- Smoking History: consists of 5 categories which is not current, former, no info, current, never, and ever.
- HbA1c Level: Higher levels indicate a greater risk of developing diabetes.
- Blood Glucose Level: High blood glucose levels are a key indicator of diabetes.
- Diabetes: values of 1 indicate the presence of diabetes and 0 indicates the absence of diabetes.

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
This part presenting the statistic summary from numeric feature on dataset.
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

### Data Ditribution Visualization
This part presenting the visualization (like bar chart and pie chart) from distribution in each categorical feature.
#### Bar Chart for Gender Distribution
![Gender Distribution](src/gender.png)

#### Bar Chart for Smoking History Distribution
![Smoking History Distribution](src/smoking_history.png)

#### Pie Chart for Hypertension Distribution
![Hypertension Distribution](src/hypertension.png)

#### Pie Chart for Heart Disease Distribution
![Heart Disease Distribution](src/heart_disease.png)

## Data Preparation
To make the model perform better and produce more reliable predictions, a thorough data preparation process was conducted. This step is crucial to ensure that the dataset is clean, consistent, and suitable for machine learning algorithms. The following techniques were applied in the notebook, in the order they were executed:
- Dropping Duplicate Records: duplicate rows in the dataset can skew the model by overrepresenting certain patterns. This process ensures that each instance in the training data is unique, avoiding bias in learning and helping to prevent overfitting. Therefore, we used:
    <pre>
     # Drop Duplicate Columns
     diabetes = diabetes.drop_duplicates()
    </pre>
    
- Removing Irrelevant Values: some values in the dataset may not contribute meaningful information or are too rare to be statistically significant. In this case:
    <pre>
       # Drop Unnecesary Value on Gender Column
       gender_label = diabetes[diabetes['gender'] == 'Other'].index
       diabetes.drop(gender_label, inplace=True)
  
       # Drop Unnecesary Value on Smoking History Column
       smoking_history_label = diabetes[diabetes['smoking_history'] == 'No Info'].index
       diabetes.drop(smoking_history_label, inplace=True)
    </pre>
    
- Label Encoding: categorical features such as gender and smoking_history were encoded using label encoding to convert string labels into numeric form. Most machine learning models only accept numerical input. Label encoding enables the use of categorical data in these models while maintaining the order of categories if needed.
    <pre>
      # Do Label Encoding for 'Gender' and 'Smoking History' Column
      labelEncoder = LabelEncoder()
      diabetes['gender'] = labelEncoder.fit_transform(diabetes['gender'])
      diabetes['smoking_history'] = labelEncoder.fit_transform(diabetes['smoking_history'])
    </pre>
    
- Outliers handling: outliers were detected and treated using IQR (Interquartile Range) method on numerical features like bmi, HbA1c_level, and blood_glucose_level. Outliers can introduce noise and distort the learning process, leading to lower model accuracy. By handling them, we improve the model's ability to generalize.
    <pre>
      # Outliers Handling
      def cap_outliers(diabetes, column):
          Q1 = diabetes[column].quantile(0.25)
          Q3 = diabetes[column].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR
          diabetes[column] = diabetes[column].clip(lower_bound, upper_bound)
          return diabetes
      
      for col in ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']:
          diabetes = cap_outliers(diabetes, col)
    </pre>
    
- Normalization: Numerical features such as age, bmi, HbA1c_level, and blood_glucose_level were scaled using MinMaxScaler. Normalization ensures that features are on the same scale, preventing models from favoring features with larger magnitudes. This is especially important for distance-based models and gradient-based optimization:
    <pre>
      # Numeric Feature Normalization
      features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
      scaler = StandardScaler()
      
      diabetes[features] = scaler.fit_transform(diabetes[features])
    </pre>
  
- Spliting the Dataset: Before training the model, we split the dataset into training and testing sets. Data splitting is essential to evaluate how well the machine learning model generalizes to unseen data, Stratify is used to ensure the distribution of the target variable (diabetes) is preserved in both training and testing sets, which is particularly important when dealing with imbalanced datasets, A 20% test size was selected to provide enough data for a reliable evaluation while maintaining sufficient data for training, A random state was fixed to ensure the results are reproducible.:
    <pre>
      X = diabetes.drop('diabetes', axis=1)
      y = diabetes['diabetes']
      
      X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    </pre>
  
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
