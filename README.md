# Machine Learning Project Report - Kevin Juan Carlos

## Project Domain
Diabetes is the one global chronic disease that will have an impact on life quality and premature death. Based on research conducted by Takashima *et al*. (2024) in *Journal of Epidemiology*, diabetes significantly reduces healthy life expectancy in Japan—both men and women—and is categorized as influenced by risk factors such as blood pressure, body mass index (BMI), and smoking habits.

Furthermore, according to the American Diabetes Association by Wild *et al*. (2004), diabetes prevalence sharply increases when people get older, making the older generation more vulnerable to this disease. Additionally, gender also plays a crucial factor in diabetes development. According to Regitz-Zagrosek (2012), biological and hormonal differences contribute to variations in how men and women experience chronic illnesses such as diabetes. For example, men are more likely to develop diabetes at lower BMI values, while women with hormonal conditions like PCOS are at higher risk. This fact shows diabetes cannot only cause physical complications but also shorten a person's disability life span. Therefore, early detection and intervention are even more essential to maintaining the quality of the community.

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

## Data Preparation


## Modeling


## Evaluation


## References
Wild, S., Roglic, G., Green, A., Sicree, R., & King, H. (2004). Global prevalence of diabetes. Diabetes Care, 27(5), 1047–1053. https://doi.org/10.2337/diacare.27.5.1047

Regitz-Zagrosek, V. (2012). Sex and gender differences in health. EMBO Reports, 13(7), 596–603. https://doi.org/10.1038/embor.2012.87

Tsukinoki, R., Murakami, Y., Hayakawa, T., Kadota, A., Harada, A., Kita, Y., Okayama, A., Miura, K., Okamura, T., & Ueshima, H. (2025). Comprehensive assessment of the impact of blood pressure, body mass index, smoking, and diabetes on healthy life expectancy in Japan: NIPPON DATA90. Journal of Epidemiology. https://doi.org/10.2188/jea.je20240298
