# Session 2 – Practice Notes (Part A)
## Data Preprocessing for Machine Learning

> **Purpose of Part A:**  
> Learn how to prepare data *properly* before training machine learning models, and understand **why preprocessing matters** in research.

---

## 1. Why Data Preprocessing Is Necessary

In Session 1, we trained a baseline model quickly.  
In Session 2, we improve **methodological correctness**.

Machine learning models **do not understand raw data** the way humans do.

Common problems without preprocessing:
- Categorical data cannot be used directly
- Features with large scales dominate others
- Models behave unfairly or unstably
- Results become hard to compare across models

---

## 2. Types of Variables in the Churn Dataset

Before preprocessing, we must understand the data.

### 2.1 Numerical Variables
Examples:
- tenure
- MonthlyCharges
- TotalCharges

These variables:
- Have numeric meaning
- Can be compared by magnitude
- Often need **scaling**

---

### 2.2 Categorical Variables
Examples:
- Contract
- InternetService
- PaymentMethod
- gender

These variables:
- Represent categories, not quantities
- Cannot be used directly by most ML algorithms
- Must be **encoded**

---

## 3. Encoding Categorical Variables

### 3.1 Why Encoding Is Required

Machine learning models work with numbers, not text.

Example:
```text
Contract = Month-to-month, One year, Two year
```

The model does NOT know:
- One year > Month-to-month (this is NOT numeric)

Therefore, we convert categories into numeric form.

---

### 3.2 One-Hot Encoding (Concept)

One-hot encoding creates:
- One column per category
- Values of 0 or 1

Example:
```text
Contract_Month-to-month
Contract_One year
Contract_Two year
```

---

### 3.3 Apply One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

categorical_features = X.select_dtypes(include='object').columns

encoder = OneHotEncoder(drop='first', sparse=False)

X_encoded = encoder.fit_transform(X[categorical_features])
```

> Note:  
> We will later combine encoded categorical features with numerical features.

---

## 4. Scaling Numerical Variables

### 4.1 Why Scaling Is Important

Some models are sensitive to feature scale.

Example:
- tenure ranges from 0 to 72
- MonthlyCharges ranges from 20 to 120

Without scaling:
- Larger numbers dominate distance calculations
- Some models behave unfairly

---

### 4.2 Common Scaling Method: Standardization

Standardization transforms data to:
- Mean = 0
- Standard deviation = 1

This makes features comparable.

---

### 4.3 Apply Standard Scaling

```python
from sklearn.preprocessing import StandardScaler

numerical_features = X.select_dtypes(exclude='object').columns

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X[numerical_features])
```

---

## 5. Important Rule: Fit on Training Data Only

### Why this matters

If preprocessing uses test data:
- The model indirectly sees future information
- Results become overly optimistic
- This is called **data leakage**

Correct approach:
1. Split data into train and test
2. Fit encoder and scaler on training data only
3. Apply the same transformation to test data

> This rule is critical in research and will be revisited in Session 3.

---

## 6. Conceptual Check (Very Important)

Ask yourself:
- Did we treat categorical and numerical data differently?
- Did we change data meaning or only representation?
- Did preprocessing happen **before** model training?

If the answers are yes, preprocessing is correct.

---

## 7. Research Takeaway

> **Good preprocessing often improves model performance more than changing the algorithm.**

This is why preprocessing is a core part of machine learning methodology.

---

## What Comes Next

In **Part B**, we will:
- Rebuild logistic regression using proper preprocessing
- Compare results with Session 1
- Observe how performance changes

---

**End of Session 2 – Part A**
