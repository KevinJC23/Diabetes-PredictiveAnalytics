# Import Library/Packages
import os
import shutil
import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score

# Download Latest Version of Dataset
path = kagglehub.dataset_download("iammustafatz/diabetes-prediction-dataset")
print("Path to dataset files:", path)

# Change the Location of Dataset Directory
target_path = '/content/diabetes-dataset'

if not os.path.exists(target_path):
    shutil.move(path, target_path)

# Read CSV Using Pandas
diabetes = pd.read_csv('/content/diabetes-dataset/diabetes_prediction_dataset.csv')
diabetes

# Show The Summary of Dataset Structure
diabetes.info()

# Show The Summary of Descriptive Statistic for Each Column
diabetes.describe()

# Count The Missing Value in Each Column
diabetes.isnull().sum()

# Check Data Duplicates
diabetes.duplicated().sum()

# Show 'gender' Column Distribution Using Bar Chart
gender = diabetes['gender'].value_counts()

gender.plot(kind='bar')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Total')

plt.show()

# Show 'hypertension' Column Distribution Using Pie Chart
hypertension = diabetes['hypertension'].value_counts()

hypertension.plot(kind='pie', autopct='%1.1f%%')
plt.title('Hypertension Distribution')
plt.ylabel('')
plt.show()

# Show 'heart_disease' Column Distribution Using Pie Chart
heart_disease = diabetes['heart_disease'].value_counts()

heart_disease.plot(kind='pie', autopct='%1.1f%%')
plt.title('Heart Disease Distribution')
plt.ylabel('')
plt.show()

# Show 'smoking_history' Column Distribution Using Bar Chart
smoking_history = diabetes['smoking_history'].value_counts()

smoking_history.plot(kind='bar')
plt.title('Smoking History Distribution')
plt.xlabel('History')
plt.ylabel('Total')
plt.show()

# Drop Duplicate Columns
diabetes = diabetes.drop_duplicates()

# Drop Unnecesary Value on Gender Column
gender_label = diabetes[diabetes['gender'] == 'Other'].index
diabetes.drop(gender_label, inplace=True)

# Drop Unnecesary Value on Smoking History Column
smoking_history_label = diabetes[diabetes['smoking_history'] == 'No Info'].index
diabetes.drop(smoking_history_label, inplace=True)

# Do Label Encoding for 'Gender' and 'Smoking History' Column
labelEncoder = LabelEncoder()
diabetes['gender'] = labelEncoder.fit_transform(diabetes['gender'])
diabetes['smoking_history'] = labelEncoder.fit_transform(diabetes['smoking_history'])

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

# Numeric Feature Normalization
features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
scaler = StandardScaler()

diabetes[features] = scaler.fit_transform(diabetes[features])

# Boxplot
plt.figure(figsize=(15, 10))
for i, col in enumerate(diabetes.columns):
    plt.subplot(3, 4, i+1)
    sns.boxplot(x=diabetes[col])
    plt.title(f'Boxplot {col} Column')

plt.tight_layout()
plt.show()

# Pairplot
features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
sns.pairplot(diabetes[features], diag_kind='kde')

# Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = diabetes.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidth=0.5, )
plt.title('Correlation Matrix', size=20)

X = diabetes.drop('diabetes', axis=1)
y = diabetes['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print(f'Total sample in whole dataset: {len(X)}')
print(f'Total sample in train dataset: {len(X_train)}')
print(f'Total sample in test dataset: {len(X_test)}')

model_dict = {
    'LinearSVC': LinearSVC(random_state=55),
    'LogisticRegression': LogisticRegression(random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1),
    'Boosting': AdaBoostClassifier(learning_rate=0.01, random_state=42)
}

accuracy = pd.DataFrame(columns=['train', 'test'], index=model_dict.keys())

for name, model in model_dict.items():
    model.fit(X_train, y_train)
    accuracy.loc[name, 'train'] = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
    accuracy.loc[name, 'test'] = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))

accuracy

# Classification Report
for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    print(f"Classification Report for {name}:\n")
    print(classification_report(y_test, y_pred))
    print("\n" + "="*50 + "\n")

# Confusion Matrix
for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.grid(False)
    plt.show()

# ROC-AUC
plt.figure(figsize=(10, 8))

for name, model in model_dict.items():
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            print(f'Model {name} tidak mendukung ROC-AUC (tidak ada predict_proba atau decision_function)')
            continue

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

    except Exception as e:
        print(f"Gagal menghitung ROC-AUC untuk model {name}: {e}")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve untuk Semua Model')
plt.legend(loc='lower right')
plt.grid()
plt.show()

pred = X_test.copy()
pred_dict = {'y_true': y_test[:20000].values}

for name, model in model_dict.items():
    pred_dict['pred_' + name] = model.predict(pred).round(1)

predictions_diabetes = pd.DataFrame(pred_dict)
predictions_diabetes
