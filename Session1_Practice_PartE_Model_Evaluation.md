# Session 1 – Practice Notes (Part E)
## Model Evaluation Beyond Accuracy

This part focuses on **evaluating the machine learning model properly**, beyond a single accuracy number.

> Goal of Part E:  
> Understand **how the model makes mistakes** and whether it is **useful for decision-making**.

---

## Prerequisite

Before starting Part E, you should have completed:

- Part A – Colab & Google Drive  
- Part B – Understanding the dataset  
- Part C – Defining X and y  
- Part D – Train/Test split and first model  

You should already have:

```python
y_test   # true labels
y_pred   # model predictions
```

---

## 1. Why Accuracy Alone Is Not Enough

Accuracy tells us:
- How many predictions were correct overall

Accuracy does NOT tell us:
- Which type of errors the model makes
- Whether important cases were missed
- Whether the model is useful in practice

> A model can have high accuracy but still be useless.

---

## 2. Introducing the Confusion Matrix

A **confusion matrix** shows how predictions compare with actual outcomes.

It answers questions such as:
- How many churners were correctly identified?
- How many churners were missed?
- How many loyal customers were wrongly flagged?

---

## 3. Create the Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
cm
```

The output is a 2×2 matrix.

---

## 4. Understanding the Confusion Matrix (Human Explanation)

The confusion matrix contains four values:

| Actual \ Predicted | No Churn | Churn |
|---------------------|----------|-------|
| **No Churn**        | True Negative (TN) | False Positive (FP) |
| **Churn**           | False Negative (FN) | True Positive (TP) |

Explanation:
- **True Positive (TP)**: Correctly predicted churn
- **True Negative (TN)**: Correctly predicted non-churn
- **False Positive (FP)**: Predicted churn, but customer stayed
- **False Negative (FN)**: Missed a customer who churned

Example:
|                | Predicted No | Predicted Yes |
| -------------- | ------------ | ------------- |
| **Actual No**  | **938** (TN) | **98** (FP)   |
| **Actual Yes** | **160** (FN) | **213** (TP)  |

938 True Negatives
→ Customers who did not churn, and the model correctly said “No churn”

98 False Positives
→ Customers who did not churn, but the model wrongly said “Churn”

160 False Negatives ⚠️
→ Customers who did churn, but the model missed them

213 True Positives
→ Customers who did churn, and the model correctly identified them

Note: 938 + 98 + 160 + 213 = 1,409 (total test sample)
---

## 5. Why False Negatives Matter in Churn Prediction

In churn analysis:
- False negatives are often the most costly
- These are customers who churned but were not detected

> Missing a churner means losing a customer without intervention.

---

## 6. Precision and Recall (Conceptual Level)

### Precision
> “When the model predicts churn, how often is it correct?”

High precision:
- Few false alarms
- Marketing resources are not wasted

---

### Recall
> “Out of all customers who actually churned, how many did we catch?”
Recall tells us how many of the customers who actually churned were correctly found by the model.

High recall:
- Few missed churners
- Better customer retention coverage

---

## 7. Calculate Precision and Recall

```python
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred, pos_label='Yes')
recall = recall_score(y_test, y_pred, pos_label='Yes')

precision, recall
```

---

## 8. Interpreting Precision and Recall Together

- High precision, low recall:
  - Model is conservative
  - Misses many churners

- High recall, low precision:
  - Model catches many churners
  - Produces many false alarms

> There is always a trade-off.
> Example
> Precision = 0.685  (≈ 68.5%)
> Recall    = 0.571  (≈ 57.1%)
> Precision — Very Simple Explanation
When the model says “this customer will churn”, it is correct about 69 times out of 100.
So:
31% of the time, the model raises a false alarm
Marketing may contact some customers who would not churn

> Recall — Very Simple Explanation (Reinforced)
Out of all customers who actually churned, the model only caught about 57 out of 100.
So:
43% of churners were missed
These customers left without being detected

---

## 9. Reflection Question (Very Important)

Think about this question:

> “In a real business scenario, which is worse:  
> missing a customer who will churn, or contacting a customer who would not churn?”

There is no single correct answer.
The answer depends on business and research context.

---

## 10. Key Research Takeaways

- Accuracy alone is insufficient
- Confusion matrix reveals error types
- Precision and recall provide deeper insight
- Model evaluation must align with research goals

---

## What We Have Achieved So Far (End of Session 1)

By the end of Part E:
- A complete ML pipeline has been built
- The model has been evaluated responsibly
- Students understand why evaluation matters

This completes **Session 1**.

---

## Next Session Preview

In the next session, we will:
- Improve preprocessing
- Compare multiple models
- Discuss overfitting and validation

---

**End of Practice Notes – Session 1 (Part E)**
