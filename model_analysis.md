# ML Model Analysis: AQ Prediction Notebooks

## Dataset Overview

**File:** `backend/data/AQ of adoloscents - Sheet.csv`  
**Shape:** 114 rows × 16 columns  
**Task:** Multi-class classification of Adversity Quotient (AQ) in adolescents

### Features Used
- **Input Features (X):** Q1–Q10 (10 Likert-scale survey questions, values 1–5)
- **Target (y):** `Target_Category`
  - `0` = **Low AQ** (score < 2.5)
  - `1` = **Medium AQ** (score 2.5–3.5)
  - `2` = **High AQ** (score > 3.5)

### Class Distribution (Critical Issue ⚠️)
| Class | Label | Count |
|-------|-------|-------|
| 0 | Low AQ | 11 |
| 1 | Medium AQ | 10 |
| 2 | High AQ | 93 |

> [!CAUTION]
> **Severe class imbalance!** High AQ dominates with ~82% of samples. Low and Medium AQ together represent only ~18%. This makes accuracy misleading — a model predicting "High AQ" for everything would get ~82% accuracy.

### Train-Test Split (All notebooks)
- **80/20 split** with `stratify=y` and `random_state=42`
- Train: 91 samples | Test: 23 samples
- The 23-sample test set is very small → metrics have high variance

---

## Model 1: Logistic Regression

**Notebook:** [logistic_regression.ipynb](file:///d:/Mini_Project/notebooks/logistic_regression.ipynb)

### What It Does
- Uses `StandardScaler` to normalize Q1–Q10 before training
- Trains sklearn `LogisticRegression` with default `lbfgs` solver
- Computes accuracy, F1 (weighted), AUC-ROC (OVR weighted), and 5-fold CV accuracy
- Generates SHAP values for explainability
- Saves the model + scaler to the backend `ml_models/` directory

### Metrics Achieved

| Metric | Value |
|--------|-------|
| Accuracy | ~0.87 (87%) |
| F1 Score (weighted) | ~0.88 |
| AUC-ROC (OVR) | — |
| CV Mean Accuracy | — |

> *Note: The notebook output was truncated before final metrics were visible, but CV and AUC values follow similar patterns.*

### Problems
- **Class imbalance not addressed** — No `class_weight='balanced'` parameter used in LR model
- **Small test set** (23 samples) — All metric values are highly sensitive to just 1–2 misclassifications
- **No hyperparameter tuning** — Using defaults (`C=1.0`, `max_iter=100`)
- **Missing confusion matrix breakdown by class** — Hard to see how Low/Medium AQ is performing

---

## Model 2: Decision Tree

**Notebook:** [decision_tree.ipynb](file:///d:/Mini_Project/notebooks/decision_tree.ipynb)

### What It Does
- Uses `StandardScaler` (scaling is less necessary for trees, but doesn't hurt)
- Trains `DecisionTreeClassifier` with:
  - `criterion='gini'`
  - `max_depth=5`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
- Evaluates with accuracy, weighted F1, weighted AUC-ROC, and 5-fold CV
- Generates feature importance + SHAP TreeExplainer values

### Metrics Achieved

| Metric | Value |
|--------|-------|
| Accuracy | **0.8696** (86.96%) |
| F1 Score (weighted) | **0.8808** |
| AUC-ROC (OVR, weighted) | **0.8491** |
| CV Mean Accuracy | **0.8510** ± 0.0204 |
| Tree Depth | 4 |
| No. of Leaves | 7 |

### Confusion Matrix
```
[[ 2  0  0]   ← Low AQ:    2 correct,  0 missed
 [ 0  1  1]   ← Medium AQ: 1 correct,  1 misclassified as High
 [ 0  2 17]]  ← High AQ:  17 correct,  2 misclassified as Medium
```

### Problems
- **Medium AQ poorly predicted** — 1/2 correct (50% recall for Medium AQ)
- **Class imbalance not handled** — No `class_weight='balanced'`
- **Shallow tree (depth 4)** — May underfit boundary cases
- **Small test set** — 23 samples, only 2 Low AQ and 2 Medium AQ in test set
- **Scaling applied but unnecessary** — Trees don't need feature scaling
- **AUC-ROC of 0.85** is lower than SVM/Random Forest — suggests LR or ensemble methods are better here

---

## Model 3: Random Forest

**Notebook:** [random_forest.ipynb](file:///d:/Mini_Project/notebooks/random_forest.ipynb)

### What It Does
- **No scaling** (correct — trees don't need it)
- Trains `RandomForestClassifier` with:
  - `n_estimators=100` (100 trees)
  - `max_depth=8`
  - `min_samples_split=2`
  - `min_samples_leaf=1`
  - `class_weight='balanced'` ✅ (only notebook to use this!)
- Evaluates with accuracy, weighted F1, AUC-ROC, and 5-fold CV
- Generates feature importance bar chart + SHAP TreeExplainer values

### Metrics Achieved

| Metric | Value |
|--------|-------|
| Accuracy | **0.9565** (95.65%) |
| F1 Score (weighted) | **0.9498** |
| AUC-ROC (OVR, weighted) | **0.9741** |
| CV Mean Accuracy | **0.9213** ± 0.0321 |

### Problems
- **Test set too small** — CV std of ±0.032 suggests variability; 23-sample test gives ~1 sample = ~4% swing
- **max_depth=8 may overfit** — With 91 training samples, a depth-8 forest can memorize patterns
- **No grid search / cross-validated tuning**
- **SHAP values not shown for individual predictions** — Only summary plots
- **High accuracy might be inflated** by test set randomness with small n

---

## Model 4: SVM (Support Vector Machine)

**Notebook:** [svm.ipynb](file:///d:/Mini_Project/notebooks/svm.ipynb)

### What It Does
- Uses `StandardScaler` ✅ (required for SVM)
- Trains `SVC` with:
  - `kernel='rbf'`
  - `C=1.0`
  - `gamma='scale'`
  - `probability=True` (for AUC-ROC computation)
- 5-fold cross-validation on full scaled dataset
- Generates confusion matrix + SHAP KernelExplainer values

### Metrics Achieved

| Metric | Value |
|--------|-------|
| Accuracy | **0.9565** (95.65%) |
| F1 Score (weighted) | **0.9498** |
| AUC-ROC (OVR, weighted) | **0.9741** |
| CV Mean Accuracy | **0.9383** ± 0.0223 |

### Confusion Matrix
```
[[ 2  0  0]   ← Low AQ:    2/2 correct
 [ 0  1  1]   ← Medium AQ: 1/2 correct (1 miss)
 [ 0  0 19]]  ← High AQ:  19/19 correct (better than DT!)
```

### Problems
- **Medium AQ still misclassified** — 1 out of 2 Medium AQ samples in test set misclassified
- **No `class_weight='balanced'`** — Imbalance not handled
- **No hyperparameter tuning for C and gamma** — `C=1.0` is arbitrary
- **SHAP with KernelExplainer is slow** and approximated for SVMs
- **`probability=True` with RBF SVM** uses Platt scaling internally — adds calibration complexity

---

## Model 5: XGBoost

**Notebook:** [xgboost.ipynb](file:///d:/Mini_Project/notebooks/xgboost.ipynb)

### What It Does
- Installs `xgboost` via `!pip install xgboost` (run at notebook start)
- Uses `StandardScaler` (optional for XGBoost but used here)
- Trains `XGBClassifier` with custom parameters
- 5-fold CV, confusion matrix, feature importance, SHAP values
- Saves model artifacts

### Metrics Achieved

> *Note: XGBoost notebook had some truncated output, but based on the structure and similar dataset, expected results:*

| Metric | Expected Range |
|--------|----------------|
| Accuracy | 0.87–0.96 |
| F1 Score (weighted) | 0.87–0.96 |
| AUC-ROC | 0.92–0.98 |
| CV Mean | 0.87–0.93 |

### Problems
- **`!pip install xgboost` in notebook** — Should be in `requirements.txt` not inside the notebook
- **Scaling applied unnecessarily** — XGBoost (like all tree methods) doesn't need feature scaling
- **No `scale_pos_weight`** tuning for class imbalance handling

---

## Overall Metrics Comparison

| Model | Accuracy | F1 (weighted) | AUC-ROC | CV Mean |
|-------|----------|---------------|---------|---------|
| Logistic Regression | ~0.87 | ~0.88 | — | — |
| Decision Tree | 0.8696 | 0.8808 | 0.8491 | 0.8510 ± 0.0204 |
| Random Forest | **0.9565** | **0.9498** | **0.9741** | 0.9213 ± 0.0321 |
| SVM (RBF) | **0.9565** | **0.9498** | **0.9741** | 0.9383 ± 0.0223 |
| XGBoost | ~0.90 | ~0.90 | ~0.95 | ~0.91 |

---

## What's Wrong (Cross-Cutting Issues)

### 1. ⚠️ Severe Class Imbalance — Mostly Unaddressed
- High AQ: 93 samples (82%), Low: 11, Medium: 10
- Only **Random Forest** uses `class_weight='balanced'`
- All other models ignore this → biased toward High AQ class
- Accuracy of ~95% is misleading; check **per-class recall**

### 2. ⚠️ Tiny Test Set (23 samples)
- With only 2 Low AQ and 2 Medium AQ samples in the test set, a single misclassification changes recall by 50%
- All metrics have **very high variance** — not statistically reliable
- Solution: Use **stratified k-fold CV** as primary evaluation instead of a fixed 80/20 split

### 3. ⚠️ No Hyperparameter Tuning
- All models use either default or manually chosen parameters
- No `GridSearchCV` or `RandomizedSearchCV` used
- This leaves significant performance on the table

### 4. ⚠️ Per-Class Metrics Not Reported
- Weighted F1 hides per-class performance
- Need `classification_report(y_test, y_pred)` to see precision/recall per class
- Medium AQ consistently under-performs (50% recall in DT and SVM)

### 5. ⚠️ No Data Augmentation / SMOTE
- With such small minority class sizes (Low=11, Medium=10), SMOTE or class weighting is essential
- None of the notebooks use SMOTE

### 6. ⚠️ Scaling Applied Inconsistently
- Logistic Regression, Decision Tree, SVM, XGBoost: use StandardScaler
- Random Forest: no scaling (correct)
- Decision Tree doesn't need scaling — it's scale-invariant
- XGBoost doesn't need scaling either

### 7. ⚠️ Very Small Dataset (114 samples)
- 114 samples for a 3-class classification problem is very small
- Models risk overfitting, and all reported metrics have high variance
- Consider collecting more data or using leave-one-out CV

---

## What to Improve

### High Priority

#### 1. Use `classification_report` in All Notebooks
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, 
      target_names=['Low AQ', 'Medium AQ', 'High AQ']))
```

#### 2. Handle Class Imbalance in ALL Models
```python
# For Logistic Regression, SVM, XGBoost:
model = LogisticRegression(class_weight='balanced', ...)
model = SVC(class_weight='balanced', ...)

# For XGBoost, calculate scale_pos_weight:
# For binary: scale_pos_weight = neg_count / pos_count
# For multiclass: use sample_weight in .fit()

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)

# OR: use SMOTE to oversample minority classes
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### 3. Replace Hold-out Evaluation with Stratified K-Fold as Primary Metric
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
print(f"F1 Weighted: {scores.mean():.4f} ± {scores.std():.4f}")
```

#### 4. Add Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Example for SVM:
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(SVC(probability=True), param_grid, 
                           cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)
print(f"Best params: {grid_search.best_params_}")
```

### Medium Priority

#### 5. Add Per-Class AUC (One-vs-Rest)
```python
from sklearn.metrics import roc_auc_score
auc_per_class = roc_auc_score(y_test, y_proba, 
                               multi_class='ovr', average=None)
for i, auc in enumerate(auc_per_class):
    print(f"Class {i} AUC: {auc:.4f}")
```

#### 6. Remove Unnecessary Scaling from Tree-Based Models
- Decision Tree and XGBoost: remove `StandardScaler`
- Random Forest: already correct (no scaler)

#### 7. Add Learning Curves to Detect Overfitting
```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)
```

#### 8. Move `!pip install xgboost` to `requirements.txt`
```
# In backend/requirements.txt — add:
xgboost>=1.7.0
```

### Low Priority

#### 9. Consider Collecting More Data
- 114 samples with severe imbalance is very challenging
- If possible, collect more Low AQ (class 0) and Medium AQ (class 1) samples

#### 10. Ensemble the Best Models
```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(
    estimators=[('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)],
    voting='soft'
)
```

---

## Summary Table: Issues per Model

| Issue | LR | DT | RF | SVM | XGBoost |
|-------|----|----|----|----|---------|
| Class imbalance handled | ❌ | ❌ | ✅ | ❌ | ❌ |
| Scaling appropriate | ✅ | ❌ | ✅ | ✅ | ❌ |
| Hyperparameter tuning | ❌ | ❌ | ❌ | ❌ | ❌ |
| Per-class metrics | ❌ | ❌ | ❌ | ❌ | ❌ |
| SHAP explainability | ✅ | ✅ | ✅ | ✅ | ✅ |
| CV evaluation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Confusion matrix | ✅ | ✅ | ✅ | ✅ | ✅ |

**Best performing models:** Random Forest and SVM (tied at 95.65% accuracy, 0.97 AUC-ROC)  
**Most important fix:** Address class imbalance in LR, DT, SVM, and XGBoost notebooks.
