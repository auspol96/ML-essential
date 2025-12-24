# Session 2 – Practice Notes (Part B)
## Rebuilding Logistic Regression (Simple & Correct Way)

> **Goal of Part B:**  
> Train Logistic Regression again using **proper preprocessing**, and understand **why the results change**.

This part focuses on **doing things correctly**, not quickly.

---

## 1. Why We Rebuild the Model

In Session 1:
- We trained a model quickly
- We used simple encoding
- The model worked, but with warnings

In Session 2:
- We fix preprocessing
- We make the model more reliable
- We prepare for fair model comparison

> **Important idea:**  
> A better model often comes from better data preparation, not a new algorithm.

---

## 2. What “Proper Preprocessing” Means

We now treat data **based on its type**:

### Categorical data
- Example: gender, Contract, InternetService
- Solution: **One-Hot Encoding**
- Example
```sql
  Contract_Month-to-month   Contract_One year   Contract_Two year
1                         0                   0
0                         1                   0
0                         0                   1

```
  
### Numerical data
- Example: tenure, MonthlyCharges
- Solution: **Scaling**

This prevents:
- Fake numeric order
- Unfair influence of large numbers

---

## 3. Split Data First (Very Important)

We always split the data **before preprocessing**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

> We never learn from test data.  
> This avoids **data leakage**.

---

## 4. One-Hot Encode Categorical Features

```python
from sklearn.preprocessing import OneHotEncoder

categorical_cols = X.select_dtypes(include='object').columns

encoder = OneHotEncoder(drop='first', sparse=False)

X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])
```

Why:
- No fake ranking
- Each category becomes yes/no

---

## 5. Scale Numerical Features

```python
from sklearn.preprocessing import StandardScaler

numerical_cols = X.select_dtypes(exclude='object').columns

scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])
```

Why:
- Makes features comparable
- Helps model learn faster and more stably

---

## 6. Combine All Features

```python
import numpy as np

X_train_final = np.hstack([X_train_num, X_train_cat])
X_test_final = np.hstack([X_test_num, X_test_cat])
```

Now the data is:
- Numeric
- Scaled
- Ready for ML models

---

## 7. Train Logistic Regression Again

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train_final, y_train)
```

You should see:
- Fewer warnings
- More stable training

---

## 8. Make Predictions

```python
y_pred = model.predict(X_test_final)
```

---

## 9. Evaluate the Model

```python
from sklearn.metrics import accuracy_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

accuracy, cm
```

Compare:
- Accuracy
- Confusion matrix
- Recall (from Session 1)

---

## 10. What Students Should Observe

You may notice:
- Slightly better accuracy
- Better recall
- Fewer training warnings

> **This improvement comes from preprocessing, not from changing the model.**

---

## 11. Key Learning Message

> **Good preprocessing improves model performance and reliability.**

This is why preprocessing is a core part of machine learning research.

---

## What Comes Next

In **Part C**, we will:
- Introduce a different model (KNN)
- See why some models depend heavily on preprocessing

---

**End of Session 2 – Part B**
