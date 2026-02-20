# Session 1 – Practice Notes (Part D)
## Train–Test Split and First Machine Learning Model

This part introduces the **first true machine learning step**:
splitting data and training a simple model.

> Goal of Part D:  
> Learn how to **separate training and testing data** and train a **baseline machine learning model** correctly.

---

## Prerequisite

Before starting Part D, you must have completed:

- Part A – Colab & Google Drive connection  
- Part B – Understanding the dataset  
- Part C – Defining features (X) and target (y)

You should already have:

```python
X   # feature variables
y   # target variable
```

---

## 1. Why We Split Data (Very Important)

In machine learning research:

- The model must **learn** from some data
- The model must be **tested** on unseen data

> Testing a model on data it has already seen gives misleading results.

This is why we split data into:
- **Training set**
- **Testing set**

---

## 2. Train–Test Split Concept (Human Explanation)

You can think of this as:

- **Training data** → learning from past examples  
- **Testing data** → answering new exam questions  

> A model that performs well only on training data is not useful.

---

## 3. Import Required Function

We use `train_test_split` from scikit-learn:

```python
from sklearn.model_selection import train_test_split
```

---

## 4. Perform the Train–Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

**Explanation:**
- 80% data → training
- 20% data → testing
- `random_state` ensures reproducible results

---

## 5. Verify the Split

```python
X_train.shape
X_test.shape
```

```python
y_train.shape
y_test.shape
```

**Check:**
- Training and testing sets add up to original data
- X and y shapes are aligned

---

## 6. Why We Start with a Simple Model

For research:

- Simple models are easier to explain
- Simple models are strong baselines
- Complex models do not guarantee better research

We start with **Logistic Regression**.

---

## 7. Import Logistic Regression Model

```python
from sklearn.linear_model import LogisticRegression
```

---

## 8. Train the First Model

```python
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
```
**This will be error because:**
Machine learning models like Logistic Regression:
Work with numbers
Cannot calculate with strings
As a result we need to 
## 8.1. import Encoder and Encode categorical columns in X
```python
from sklearn.preprocessing import LabelEncoder

X_encoded = X.copy()

for col in X_encoded.columns:
    if X_encoded[col].dtype == 'object':
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X_encoded[col])
```
## 8.2. Re-split data
```python
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42
)

```
## 8.3. Train model again
```python
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

```

**What happens here:**
- The model learns patterns from training data
- No predictions yet

---

## 9. Make Predictions on Test Data

```python
y_pred = model.predict(X_test)
```เอาข้อมูลที่โมเดล ไม่เคยเห็นมาก่อน ให้มันลองทำนาย

**Important:**
- Predictions are made on **unseen data**
- This simulates real-world prediction

---

## 10. Evaluate Model Performance

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
accuracy
```

**Interpretation:**
- Accuracy shows how often the model predicts correctly
- Accuracy is not the only metric, but it is a starting point

---

## 11. What Accuracy Means (And What It Does Not)

Accuracy tells us:
- How often predictions are correct

Accuracy does NOT tell us:
- Why predictions are wrong
- Whether the model is fair
- Whether the model is perfect

> Moderate accuracy is normal and expected.

---

## 12. Reflection Question (Important)

Think about this question:

> “If most customers do not churn, could a model achieve high accuracy by always predicting ‘No’?”

This prepares us for better evaluation metrics in later sessions.

---

## 13. What We Have Achieved So Far

By the end of Part D:

- Data was split correctly
- A baseline ML model was trained
- Predictions were evaluated honestly
- No complex techniques were used

This is a **valid machine learning experiment**.

---

## 14. Key Research Takeaway

> A simple, well-evaluated model is better than a complex, poorly designed one.

---

## Next Step

In the next session, we will:
- Compare multiple models
- Improve evaluation methods
- Discuss overfitting and validation

---

**End of Practice Notes – Session 1 (Part D)**
