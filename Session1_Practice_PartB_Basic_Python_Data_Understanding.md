# Session 1 – Practice Notes (Part B)
## Understanding the Dataset with Basic Python Commands

This note is for **PhD and Master students** to understand the dataset **before any machine learning is applied**.

> Goal of Part B:  
> Learn basic Python commands to answer the question:  
> **“What kind of data do I have?”**

---

## Prerequisite

Before starting Part B, make sure you have:
- Opened Google Colab
- Connected Google Drive
- Successfully loaded the dataset into a DataFrame named `df`

Example:

```python
import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Customer_data/Customer_Churn.csv'
)
```

---

## 1. View the First Few Rows

```python
df.head()
```

**Explanation:**
- Shows the first 5 rows of the dataset
- Helps you understand what one customer record looks like

> Research mindset: Always understand one observation before modeling many.

---

## 2. Check Dataset Size (Rows and Columns)

```python
df.shape
```

**Output format:**
```
(number_of_rows, number_of_columns)
```

**Explanation:**
- Rows = number of customers
- Columns = number of variables

> Dataset size helps determine whether machine learning is appropriate.

---

## 3. Count the Number of Rows Only

```python
len(df)
```

or

```python
df.shape[0]
```

**Explanation:**
- Total number of customers (samples)

---

## 4. Count the Number of Columns Only

```python
df.shape[1]
```

**Explanation:**
- Total number of variables describing each customer

> More variables do not always mean a better model.

---

## 5. List All Column Names

```python
df.columns
```

**Explanation:**
- Displays all available variables
- Helps identify potential features and the target variable

> This prepares you for defining X and y in the next step.

---

## 6. Check Data Types

```python
df.dtypes
```

**Explanation:**
- Shows whether each column is numeric or categorical
- Machine learning models require numerical input

> This explains why data preprocessing is often needed.

---

## 7. Get a Dataset Overview

```python
df.info()
```

**Explanation:**
- Shows:
  - Number of rows and columns
  - Data types
  - Missing values

> This is a quick “health check” of the dataset and a good research habit.

---

## 8. View Basic Statistics (Numerical Columns)

```python
df.describe()
```

**Explanation:**
- Shows summary statistics such as:
  - Mean
  - Minimum and maximum values
- Applies only to numerical columns

> Focus on understanding value ranges, not formulas.

---

## 9. Examine the Target Variable Distribution

```python
df['Churn'].value_counts()
```

**Explanation:**
- Shows how many customers:
  - Stayed (No)
  - Left (Yes)

> This helps identify class imbalance, which affects model evaluation.

---

## 10. Reflection Question (Think, Do Not Code)

Answer this question in your own words:

> “If most customers did not churn, what would happen if a model always predicted ‘No’?”

This question prepares you for evaluation metrics in later sessions.

---

## Key Takeaways from Part B

- Understand your data before modeling
- Know how many samples and variables you have
- Identify variable types and data quality
- Never rush into machine learning without data understanding

---

## Next Step

Once you are comfortable with these commands, we will move to:

### **Part C – Defining Features (X) and Target Variable (y)**

This is where machine learning formally begins.

---

**End of Practice Notes – Session 1 (Part B)**
