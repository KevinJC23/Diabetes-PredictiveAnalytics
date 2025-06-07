# Machine Learning Project Report - Kevin Juan Carlos

## Project Domain
Diabetes is the one global chronic disease that will have an impact on life quality and premature death. Based on research conducted by Takashima *et al*. (2024) in *Journal of Epidemiology*, diabetes significantly reduces healthy life expectancy in Japan—both men and women—and is categorized as influenced by risk factors such as blood pressure, body mass index (BMI), and smoking habits. According to the American Diabetes Association by Wild *et al*. (2004), diabetes prevalence sharply increases when people get older, making the older generation more vulnerable to this disease. 

Moreover, gender also plays a crucial role in the development of diabetes, as biological and hormonal differences contribute to variations in how men and women experience chronic illnesses such as diabetes (Regitz-Zagrosek, 2012). For example, men are more likely to develop diabetes at lower BMI values, while women with hormonal conditions like PCOS are at higher risk. This fact shows that diabetes cannot only cause physical complications but also shorten a person's disability lifespan. Therefore, early detection and intervention are even more essential to maintaining the quality of the community. Advancements in technology have enabled the utilization of Artificial Intelligence, particularly Machine Learning, which can be applied to predict the probability of people having diabetes based on their medical data. This project focuses on predictive analytics and aims to compare Machine Learning model classification to know what models can make the diagnosis process efficient and accurate.

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
- Contains 3,854 duplicate entries
- Data imbalance

The dataset consists of eight input features and one target label, which are as follows:
- Age: The range starts from 0 and goes up to 80.
- Gender: Consist of three categories which are male, female, and other.
- Body Mass Index (BMI): The range of BMI in the dataset is from 10.16 to 71.55. (Underweight (BMI < 18.5),  Normal (18.5 ≤ BMI < 24.9), Overweight (25 ≤ BMI < 29.9), and Obese (BMI ≥ 30)).
- Hypertension: It has values of 0 or 1 where 0 indicates indicates they don’t have hypertension and 1 means they have hypertension.
- Heart Disease: It has values of 0 or 1 where 0 indicates they don’t have heart disease and 1 means they have heart disease.
- Smoking History: Consists of six categories which are not current, former, no info, current, never, and ever.
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

### Datasets Descriptive Statistic
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
![gender](https://github.com/user-attachments/assets/7c3d5b92-ff49-4c3e-8325-1def51199e3e)

The distribution reveals the other category has a value that approximates to 0, in contrast to the female category, which approximates 58,000 instances, and the male category, which registers around 40,000. As the other label holds no utility for the intended modelling process, its deletion will be necessary.

- #### Bar Chart for Smoking History Distribution
![smoking_history](https://github.com/user-attachments/assets/c57084ce-15cb-4c44-bedf-72af41c89c62)

The resulting depiction highlights 'No Info' as the most prevalent category, while others are markedly less frequent. Given that 'No Info' label likely denote missing or unavailable data, its unaddressed presence has the potential to introduce bias into the model. Following the removal for duplicate rows become 96,146, the dataset comprises ~60,000 entries, underscoring the need to assess the impact of retaining the 'No Info' category on subsequent modeling.

- #### Pie Chart for Hypertension Distribution
![hypertension](https://github.com/user-attachments/assets/807804f4-3a8c-46f1-8fce-b320a35914fa)

The numerical encoding designates 0 as the absence of hypertension; In contrast, numerical encoding designates 1 as the indication of hypertension. These charts show that the majority are people who don't have hypertension. Considering that the 0 value accounts for 92.5% of the overall dataset composition, the implementation of undersampling is deemed inadvisable, as it would precipitate a substantial reduction in the total number of observations, potentially compromising the integrity of other columns.

- #### Pie Chart for Heart Disease Distribution
![heart_disease](https://github.com/user-attachments/assets/4a8fcd9c-1141-4f18-944b-b1a70b573706)

The numerical encoding designates 0 as the absence of heart disease; In contrast, numerical encoding designates 1 as the indication of heart disease. Regarding the distributional characteristics of hypertension, the application of undersampling is deemed imprudent due to the overwhelming prevalence of 0 values, comprising 96.1% of the overall dataset.

### Data Preparation Visualization
This part presenting the visualization after data preprocessing.

- #### Examine the boxplot of each feature and ensure outliers have been properly handled
![Boxplot](https://github.com/user-attachments/assets/c5c5ae61-0a04-4398-8b08-8eba851b1d17)

The boxplots reveal that most numerical features such as age, BMI, HbA1c level, and blood glucose level contain some outliers, particularly on the higher end, indicating possible extreme values that may need further investigation or treatment. Categorical or binary features like gender, hypertension, heart disease, and diabetes appear well-distributed without apparent outliers, which is expected given their limited value range. Notably, the smoking_history feature, although numeric, shows variability that suggests it may represent encoded categorical data, with a few outliers at the lower end. Overall, while categorical features are clean, numerical features might benefit from outlier handling to improve model performance.

- #### Explore relationships between each feature using pairplot
![Pairplot](https://github.com/user-attachments/assets/8ba363e9-9598-412a-b61c-e029605d1971)

The pairplot reveals a few notable patterns in the data. Individuals diagnosed with diabetes (diabetes = 1) tend to have higher HbA1c and blood glucose levels, which aligns with medical expectations, indicating these features are strong indicators of diabetes. There is no clear linear relationship between age or BMI with diabetes, although slight clustering suggests that older individuals and those with higher BMI may have a higher risk. The distribution of HbA1c and blood glucose levels is more skewed for diabetic individuals, while non-diabetic cases are more concentrated at lower values. Overall, HbA1c and blood glucose levels show the strongest visual separation between diabetic and non-diabetic groups.

- #### Analyze the correlation between features using heatmap
![Heatmap](https://github.com/user-attachments/assets/5aa38f4f-7821-453e-93b6-a60b62f1b021)

The correlation heatmap shows that HbA1c_level and blood_glucose_level have the strongest positive correlation with the target variable diabetes, with coefficients around 0.45 and 0.39 respectively. This aligns with known clinical indicators of diabetes. Other features such as age, bmi, and hypertension exhibit weak to very weak positive correlations with diabetes (less than 0.1), suggesting they might contribute less directly. Additionally, most features do not show strong correlations with each other, indicating minimal multicollinearity. Overall, the heatmap highlights HbA1c_level and blood_glucose_level as the most informative predictors for diabetes in this dataset.

## Data Preparation
### Dropping Duplicate Records
Entries with irrelevant, unclear, or overly rare values—such as the "Other" category in the `gender` column, "No Info" in the `smoking_history` column, and duplicate rows that could skew the model by overrepresenting certain patterns—were identified and removed to improve data quality, reduce noise and bias, enhance model learning efficiency, and prevent overfitting by maintaining only meaningful and unique instances in the training data.
  
### Label Encoding
Categorical features such as `gender` and `smoking_history` were encoded using Label Encoding with `LabelEncoder`, converting string labels into numeric form to allow these features to be used effectively in the modeling process, as most machine learning algorithms require numerical input and label encoding preserves any potential ordinal relationships.

### Outliers handling
Outliers in numerical features such as `age`, `bmi`, `HbA1c_level`, and `blood_glucose_level` were detected and treated using the Interquartile Range (IQR) method, which calculates the lower and upper bounds based on the `1.5 * IQR` rule and caps values outside this range to the nearest acceptable limit; this approach helps reduce noise and distortion during the learning process, thereby improving the model’s generalization performance.

### Split the Dataset
Before training the model, the dataset was split into training and testing subsets using an 80:20 ratio with `train_test_split`, where the features (X) were separated from the target variable (y), and stratification based on the target (diabetes) was applied to maintain the original class distribution in both sets—an essential step when handling imbalanced data. Additionally, a fixed random state was used to ensure reproducibility, allowing consistent results across multiple runs of the model.

### Standardization
Feature standardization was applied using StandardScaler, which transforms numerical features such as `age`, `bmi`, `HbA1c_level`, and `blood_glucose_level` to have a mean of 0 and a standard deviation of 1, rather than scaling the values to a fixed range. Importantly, the scaler was fitted only on the training data (X_train) to avoid data leakage, and then applied to both the training and test sets. This approach improves model performance and ensures fair evaluation by preserving the integrity of unseen test data.

## Modeling
To solve the classification problem of predicting diabetes, experiment conducted with four different machine learning models. Each model underwent hyperparameter tuning using GridSearchCV to achieve optimal performance. Below is a detailed explanation of the modeling process, evaluation, and reasoning behind the model selection.
### Models Used
- #### Linear Support Vector Classifier (LinearSVC)
Works by finding a hyperplane that best separates the data into two classes while maximizing the margin between them. This approach is particularly effective for high-dimensional datasets. We tuned two key hyperparameters: C, which controls the regularization strength (with lower values indicating stronger regularization), and max_iter, which sets the maximum number of iterations to ensure the algorithm converges. The tested values for C were [0.01, 0.1, 1, 10], and for max_iter were [1000, 2000]. After hyperparameter tuning using GridSearchCV, the best parameters found for LinearSVC were ```C=1``` and ```max_iter=1000```.The advantage of LinearSVC is its efficiency when handling high-dimensional data, and it trains relatively fast on smaller datasets. However, it assumes that the data is linearly separable and is sensitive to unscaled or noisy data.

- #### Logistic Regression
Predicts class membership probabilities using a logistic function, making it suitable for binary classification tasks. The hyperparameters tuned for Logistic Regression were C, which is the inverse of regularization strength, and solver, which determines the optimization algorithm used. The values tested for C were [0.01, 0.1, 1, 10], while the solver options included 'liblinear' and 'lbfgs'. After hyperparameter tuning using GridSearchCV, the best parameters found for Logistic Regression were ```C=0.1``` and ```solver='liblinear'```. Logistic Regression is simple, easy to interpret, and serves well as a baseline model, especially when proper regularization is applied. However, it assumes a linear relationship between features and the target, which can be limiting when dealing with non-linear data distributions.

- #### Random Forest
Works by constructing a large number of decision trees during training and outputting the most common class as the final prediction. This method captures complex relationships and feature interactions effectively. The key hyperparameters tuned were n_estimators (number of trees), max_depth (maximum depth of each tree), and min_samples_split (minimum samples required to split a node). We experimented with n_estimators values of [50, 100, 150], max_depth values of [10, 16, 20], and min_samples_split values of [2, 5]. After hyperparameter tuning using GridSearchCV, the best parameters found for Random Forest were ```max_depth=10```, ```min_samples_split=2```, and ```n_estimators=50```. The strength of Random Forest lies in its ability to handle non-linear relationships and feature interactions, making it robust to outliers and missing values. However, it can be slower to train when using a large number of trees and is generally harder to interpret compared to linear models.

- #### AdaBoost Classifier
Combines multiple weak classifiers (typically decision trees) to build a stronger predictive model. The model iteratively focuses on instances that are harder to classify correctly. The hyperparameters tuned were n_estimators, representing the number of boosting stages, and learning_rate, which reduces the weight of each weak learner’s contribution. The tested values for n_estimators were [50, 100, 200], and for learning_rate, they were [0.001, 0.01, 0.1]. After hyperparameter tuning using GridSearchCV, the best parameters found for AdaBoost were ```learning_rate=0.1``` and ```n_estimators=200```. AdaBoost is particularly useful when dealing with imbalanced datasets and often yields improved accuracy by combining weak learners. However, it can be sensitive to noisy data and outliers, and if not tuned properly, it may overfit.

## Evaluation
### Model Performance Summary
| Model              | Train Accuracy | Test Accuracy  |
|--------------------|----------------|----------------|
| LinearSVC          | 0.9481         | 0.9473         |
| LogisticRegression | 0.9476         | 0.9471         |
| RandomForest       | 0.9638         | 0.9631         |
| AdaBoost           | 0.9627         | 0.9630         |

The obtained model performance results demonstrate that all four models achieve elevated accuracy on both the training and testing datasets. Notably, LinearSVC and LogisticRegression show slightly diminished accuracy (94%) relative to RandomForest and AdaBoosting (96%). Consequently, these outcomes suggest a slightly more robust and reliable outcome for this particular prediction task. Among the evaluated models, **RandomForest** achieves the highest accuracy on both training and testing datasets, indicating it is the best-performing algorithm in this case.

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

- #### Logistic Regression

```
              precision    recall  f1-score   support

           0       0.96      0.99      0.97     11243
           1       0.85      0.63      0.73      1407

    accuracy                           0.95     12650
   macro avg       0.90      0.81      0.85     12650
weighted avg       0.94      0.95      0.94     12650
```

#### Random Forest

```
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     11243
           1       1.00      0.67      0.80      1407

    accuracy                           0.96     12650
   macro avg       0.98      0.83      0.89     12650
weighted avg       0.96      0.96      0.96     12650
```

#### AdaBoost (Boosting)


```
              precision    recall  f1-score   support

           0       0.96      1.00      0.98     11243
           1       1.00      0.67      0.80      1407

    accuracy                           0.96     12650
   macro avg       0.98      0.83      0.89     12650
weighted avg       0.96      0.96      0.96     12650
```

Despite achieving high accuracy (95-96%), indicating that they are able to correctly classify the majority of the data overall. However, differences in performance are more apparent in the recall and F1-score metrics, especially for the minority class.

### Confusion Matrix

The **confusion matrix** helps visualize the performance of a classification model by displaying the true positives (TP), false positives (FP), true negatives (TN), and false negatives (FN). It provides insight into the type of errors each model makes and are crucial for understanding trade-offs between precision and recall.

- **True Positive (TP)**: Correctly predicted positive outcomes (diabetes).
- **False Positive (FP)**: Incorrectly predicted positive outcomes (non-diabetes predicted as diabetes).
- **True Negative (TN)**: Correctly predicted negative outcomes (non-diabetes).
- **False Negative (FN)**: Incorrectly predicted negative outcomes (diabetes predicted as non-diabetes).

- #### **LinearSVC Confusion Matrix**
![LinearSVC](https://github.com/user-attachments/assets/38292905-efcd-4542-810c-27e9ae378f64)

- #### **Logistic Regression Confusion Matrix**
![LogisticRegression](https://github.com/user-attachments/assets/0ce47040-32d8-45e5-8a48-b36545e88354)
  
- #### **Random Forest Confusion Matrix**
![RandomForest](https://github.com/user-attachments/assets/0a7b1276-f630-4f49-9537-29e186be48f1)

- #### **AdaBoost Confusion Matrix**
![download (1)](https://github.com/user-attachments/assets/ae035796-7381-4f57-b33d-bc2afa92acb1)

The comparative analysis reveals the performance of **AdaBoost** for all aspects is assessed, notably eliminating false positives and achieving a higher degree of accuracy. However, **RandomForest** has a higher recall than **AdaBoost**. Both **LinearSVC** and **LogisticRegression** demonstrate a weakness in predicting positive cases, resulting in lower recall values.

### ROC-AUC

The **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)** is a performance measurement for classification problems. It shows the trade-off between **True Positive Rate (Recall)** and **False Positive Rate** across different thresholds.

- **ROC Curve**: A plot of the true positive rate (recall) against the false positive rate.
- **AUC (Area Under Curve)**: A measure of how well the model distinguishes between classes. The higher the AUC, the better the model’s ability to differentiate between the positive and negative classes.

![ROC-AUC](https://github.com/user-attachments/assets/7b89f53b-b94c-4989-8d4a-12132a2a2148)


Based on the observed results, both RandomForest and AdaBoost emerged as effective diabetes detection within this dataset. Conversely, LinearSVC and LogisticRegression, while offering the advantage of simplicity and speed, may still be considered as alternative approaches.

## Conclusion
Based on the evaluation result of the Machine Learning models created in this extend, all the laid out issue articulations have been successfully addressed utilizing well-aligned methodologies that reflect the overarching trade objective, which is improving the precision and effectiveness of early diabetes risk location.

To start with, the show effectively predicts diabetes hazard utilizing key factors such as blood weight, BMI, smoking propensities, age, and sex. This directly reacts to the primary issue articulation, and moreover contributes to the objective of distinguishing and understanding the essential risk factors affecting diabetes onset. The discoveries highlight that numerical highlights like BMI, HbA1c_level, and blood_glucose_level play a especially powerful part within the model's predictive capability.

Moreover, the usage of different Machine Learning calculations has clearly illustrated their potential to improve symptomatic accuracy. By assessing the execution of four unmistakable algorithms—LinearSVC, Logistic Regression, Random Forest, and AdaBoost—the extend gives profitable insights into the qualities and confinements of each strategy when connected to therapeutic information. This does not, as it were, fulfill the moment issue explanation, but also underpins the objective of creating a vigorous and dependable forecast framework.

In terms of show execution, Random Forest and AdaBoost rose as the foremost successful algorithms, conveying higher precision and way better generalization on inconspicuous test information compared to LinearSVC and Logistic Regression. This result addresses the third issue explanation and emphasizes which strategies are most reasonable for supporting the restorative determination handling. The use of hyperparameter tuning through GridSearchCV and RandomizedSearchCV assists in refining show results, approving the adequacy of the proposed arrangement methodologies.

In conclusion, the end-to-end process—from preprocessing and including change to demonstrate optimization—has altogether contributed to the realization of the project's commercial objective. The ultimate result may be a data-driven decision support device planned to help healthcare experts in recognizing diabetes dangers at an prior arrange, empowering quicker, more precise, and eventually more impactful therapeutic interventions.

## References
Wild, S., Roglic, G., Green, A., Sicree, R., & King, H. (2004). Global prevalence of diabetes. Diabetes Care, 27(5), 1047–1053. https://doi.org/10.2337/diacare.27.5.1047

Regitz-Zagrosek, V. (2012). Sex and gender differences in health. EMBO Reports, 13(7), 596–603. https://doi.org/10.1038/embor.2012.87

Tsukinoki, R., Murakami, Y., Hayakawa, T., Kadota, A., Harada, A., Kita, Y., Okayama, A., Miura, K., Okamura, T., & Ueshima, H. (2025). Comprehensive assessment of the impact of blood pressure, body mass index, smoking, and diabetes on healthy life expectancy in Japan: NIPPON DATA90. Journal of Epidemiology. https://doi.org/10.2188/jea.je20240298
