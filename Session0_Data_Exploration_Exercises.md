# Session 0 – Data Exploration & Analysis Exercises (Python)

Dataset: **Telco Customer Churn**  
Objective: Understand the data before building any machine learning model.

---

## Setup and load dataset:

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

## Exercise 1: Understand Dataset Structure

**Task:**  
Check the size and structure of the dataset.

```python
df.shape
df.columns
```

**Question:**  
How many customers and how many features are there?

---

## Exercise 2: Identify Data Types

**Task:**  
Inspect data types of each column.

```python
df.info()
df.dtypes
```

**Question:**  
Which variables are numerical and which are categorical?

---

## Exercise 3: Detect Missing and Problematic Values

**Task:**  
Check missing values and problematic columns.

```python
df.isnull().sum()
```
**Question:**  
Which columns may require cleaning before modeling?

---

## Exercise 4: Churn Distribution

**Task:**  
Analyze the target variable.

```python
df['Churn'].value_counts()
```

Visualization:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()
```

**Question:**  
Is the dataset balanced?

---

## Exercise 5: Tenure Analysis

**Task:**  
Compare tenure between churned and non-churned customers.

```python
df.groupby('Churn')['tenure'].mean()
```
Example.
Churn
No     37.57
Yes    17.98

Customers who did NOT churn
Stayed on average ~37.6 months
Customers who DID churn
Stayed on average only ~18 months

Visualization:

```python
sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()
```

**Question:**  
Do long-term customers churn less?

---

## Exercise 6: Contract Type and Churn

**Task:**  
Explore churn by contract type.

```python
df.groupby('Contract')['Churn'].value_counts()
```

Visualization:

```python
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=20)
plt.show()
```

**Question:**  
Which contract type has the highest churn risk?

---

## Exercise 7: Monthly Charges Analysis

**Task:**  
Compare monthly charges for churned vs non-churned customers.

```python
df.groupby('Churn')['MonthlyCharges'].mean()
```

Visualization:

```python
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()
```

**Question:**  
Do higher charges relate to higher churn?

---

## Exercise 8: Service & Payment Method Analysis

**Task:**  
Explore churn across services and payment methods.

```python
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title("Internet Service vs Churn")
plt.show()
```

```python
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=30)
plt.show()
```

**Question:**  
Which services or payment methods show higher churn?

---

## Exercise 9: Correlation Analysis (Numerical Only)

**Task:**  
Analyze correlations among numerical variables.

```python
numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corr()
```

Visualization:

```python
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
```

**Question:**  
Which numerical features are strongly related?

---

## Exercise 10: From Observation to Research Question

**Task:**  
Write **3 observations** and convert them into research questions.

**Example:**  
- Observation: Short-tenure customers churn more  
- Research Question:  
  *Does tenure significantly influence customer churn probability?*

---

## Final Reminder

> If you do not understand your data, you cannot trust your model.

End of Session 0.
