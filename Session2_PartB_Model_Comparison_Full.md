# Session 2 – Part B  
## Model Comparison: From Data to Confusion Matrices  

This lab compares Logistic Regression, KNN, and Decision Tree using the same data and preprocessing.

---

## 1. Load & Prepare Data (Minimal, Safe)

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Customer_data/Customer_Churn.csv'
)

df.head()
```

---

## 2. Define Target (y) and Features (X)

```python
y = df['Churn']
X = df.drop(['Churn', 'customerID'], axis=1)
```

---

## 3. Fix Data Types (Important)

```python
# Convert target to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Fix TotalCharges (important)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Drop ID
df = df.drop('customerID', axis=1)
```

---

## 4. Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

```

---

## 5. Preprocessing (One-Hot + Scaling)

```python
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ]
)
```

---

## 6. Train Multiple Models

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
}
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

results = []

for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 3),
        "Recall (Churn)": round(recall, 3)
    })

results_df = pd.DataFrame(results)
results_df
```

---

## 7. Confusion Matrix (All Models)

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=['No', 'Yes'])

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} – Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
```

---

## Key Learning Points

- Same data, same preprocessing → fair comparison
- Different models make different types of mistakes
- Confusion matrix shows *how* models fail, not just how often

