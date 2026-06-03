# Project Changes — Session Log

> **Date:** 2026-06-02  
> **Scope:** Random Forest notebook, `ml_models/` folder organisation, `main.py` dynamic registry, `predict.py` SHAP fix

---

## Table of Contents

1. [Overview](#overview)
2. [New File: `notebooks/random_forest.ipynb`](#1-new-file-notebooksrandom_forestipynb)
3. [Refactored: `backend/main.py`](#2-refactored-backendmainpy)
4. [Fixed: `backend/router/predict.py`](#3-fixed-backendrouterpredictpy)
5. [ml_models/ Folder Structure](#4-ml_models-folder-structure)
6. [How to Add a New Model](#5-how-to-add-a-new-model)

---

## Overview

This session added **Random Forest** as a second ML model alongside the existing Logistic Regression. The key goals were:

- Create a `random_forest.ipynb` notebook that mirrors the style and structure of `logistic_regression.ipynb`
- Fix `main.py` so it discovers models **dynamically** from subfolders instead of hardcoded flat paths
- Organise `ml_models/` consistently with one subfolder per model
- Make `predict.py` use the correct per-model SHAP explainer

---

## 1. New File: `notebooks/random_forest.ipynb`

**Path:** `d:/Mini_Project/notebooks/random_forest.ipynb`

### What it does

A complete model-training notebook for `RandomForestClassifier`. It follows the exact same cell-by-cell structure as `logistic_regression.ipynb`.

### Cell Structure

| # | Cell | Description |
|---|------|-------------|
| 1 | **Imports** | `RandomForestClassifier`, SHAP, pandas, sklearn metrics — same as LR but without `StandardScaler` |
| 2 | **Data Loading** | Reads `../backend/data/AQ of adoloscents - Sheet.csv`, engineers `Target_Category` (identical to LR) |
| 3 | **Class Distribution** | Bar chart of Low / Medium / High AQ counts |
| 4 | **Train / Test Split** | 80/20 stratified split — **no scaling applied** (RF doesn't require it) |
| 5 | **Train Model** | `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)` |
| 6 | **Evaluation** | Accuracy, weighted F1, weighted AUC-ROC |
| 7 | **Cross-Validation** | 5-fold CV — mean accuracy + std |
| 8 | **Confusion Matrix** | `ConfusionMatrixDisplay` plot (Low / Medium / High) |
| 9 | **Gini Feature Importance** | Bar chart of native RF feature importances |
| 10 | **SHAP Explainability** | `shap.TreeExplainer` + mean \|SHAP\| per feature bar chart |
| 11 | **Save Artifacts** | Saves all outputs to `../backend/ml_models/random_forest/` |

### Artifacts saved by the notebook

```text
backend/ml_models/random_forest/
├── random_forest_model.pkl     ← trained RandomForestClassifier
├── X_train.pkl                 ← training data (for SHAP background)
├── shap_explainer.pkl          ← TreeExplainer instance
├── evaluation_metrics.json     ← accuracy, f1, auc_roc, cv_mean, cv_std
└── feature_importance.json     ← SHAP-based feature ranking
```

> **Note:** No `scaler.pkl` is saved — Random Forest works on raw feature values.

### Key differences from `logistic_regression.ipynb`

| Aspect | Logistic Regression | Random Forest |
|--------|--------------------|--------------------|
| Model class | `LogisticRegression` | `RandomForestClassifier` |
| Needs scaler | `StandardScaler` saved | No scaler |
| SHAP type | `LinearExplainer` | `TreeExplainer` |
| Feature importance | Coefficient magnitude | Gini + SHAP |
| Save dir | `ml_models/logistic_regression/` | `ml_models/random_forest/` |

---

## 2. Refactored: `backend/main.py`

**Path:** `d:/Mini_Project/backend/main.py`

### Problem (Before)

The old `ModelRegistry`:

- Hardcoded flat paths like `ml_models/logistic_regression_model.pkl`
- Had a single shared `self.scaler`, `self.shap_explainer`, `self.X_train`
- Had Random Forest commented out
- Could not discover new models without editing source code

```python
# OLD — hardcoded, flat, commented-out RF
model_files = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    # 'Random Forest': 'random_forest_model.pkl',
}
for model_name, filename in model_files.items():
    path = os.path.join(self.models_dir, filename)
    ...
```

### Solution (After)

The new `ModelRegistry` dynamically discovers every subfolder inside `ml_models/`:

```text
ml_models/
├── logistic_regression/
└── random_forest/
```

#### Discovery algorithm

For each subfolder found:

1. **Read `evaluation_metrics.json`** → extract `model_name` (e.g. `"Logistic Regression"`)
2. **Find `*_model.pkl`** → load the sklearn model into `self.models[model_name]`
3. **Load optional components** (if the file exists):
   - `scaler.pkl` → `self.scalers[model_name]`
   - `shap_explainer.pkl` → `self.shap_explainers[model_name]`
   - `X_train.pkl` → `self.X_trains[model_name]`
4. **Accumulate** `evaluation_metrics` (list extended per model)
5. **Set** `global_feature_importance` from the first model found

#### Key attribute changes

| Old attribute | New attribute | Type | Notes |
|---|---|---|---|
| `self.models` | `self.models` | `dict` | Unchanged — `{model_name: model}` |
| `self.scaler` | `self.scalers` | `dict` | Per-model; `None` if model doesn't need it |
| `self.shap_explainer` | `self.shap_explainers` | `dict` | Per-model |
| `self.X_train` | `self.X_trains` | `dict` | Per-model |
| `self.evaluation_metrics` | `self.evaluation_metrics` | `list` | Merged from all subfolders |
| `self.global_feature_importance` | `self.global_feature_importance` | `list` | From first subfolder found |

#### `predict()` method

```python
# OLD — scaler hardcoded by name
X_processed = self.scaler.transform(X) if model_name in ['Logistic Regression', 'SVM'] else X

# NEW — uses per-model scaler; applies only if one was saved
scaler = self.scalers.get(model_name)
X_processed = scaler.transform(X) if scaler is not None else X
```

---

## 3. Fixed: `backend/router/predict.py`

**Path:** `d:/Mini_Project/backend/router/predict.py`

### Change

The SHAP local-explanation block was updated to use the **best model's** SHAP explainer, not a single global one:

```python
# BEFORE
if model_registry.shap_explainer:
    shap_vals = model_registry.shap_explainer.shap_values(X)

# AFTER
shap_explainer = model_registry.shap_explainers.get(best_model_name)
if shap_explainer:
    shap_vals = shap_explainer.shap_values(X)
```

This ensures:

- If the best-performing model is Logistic Regression → uses LR's `LinearExplainer`
- If the best-performing model is Random Forest → uses RF's `TreeExplainer`
- If neither has an explainer saved → gracefully skips (empty list returned)

---

## 4. `ml_models/` Folder Structure

### Current state (after running the LR notebook)

```text
backend/ml_models/
├── logistic_regression/
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   ├── shap_explainer.pkl
│   ├── X_train.pkl
│   ├── evaluation_metrics.json
│   └── feature_importance.json
└── random_forest/
    └── (empty — run random_forest.ipynb to populate)
```

### After running `random_forest.ipynb`

```text
backend/ml_models/
├── logistic_regression/
│   └── ... (as above)
└── random_forest/
    ├── random_forest_model.pkl
    ├── shap_explainer.pkl
    ├── X_train.pkl
    ├── evaluation_metrics.json
    └── feature_importance.json
```

---

## 5. How to Add a New Model

Thanks to the dynamic discovery in `main.py`, adding a third model requires **zero code changes**:

1. Create a new notebook (e.g. `notebooks/decision_tree.ipynb`) following the same pattern
2. Train the model and save artifacts to `backend/ml_models/decision_tree/`
3. The folder **must** contain:
   - `decision_tree_model.pkl` (any `*_model.pkl` name works)
   - `evaluation_metrics.json` with `[{"model_name": "Decision Tree", ...}]`
4. Optionally include `scaler.pkl`, `shap_explainer.pkl`, `X_train.pkl`, `feature_importance.json`
5. Restart the FastAPI server — the new model is loaded automatically

---

## Files Changed

| File | Status | Summary |
|------|--------|---------|
| `notebooks/random_forest.ipynb` | Created | Full RF training + SHAP + artifact saving |
| `backend/main.py` | Rewritten | Dynamic subfolder discovery, per-model components |
| `backend/router/predict.py` | Modified | Per-model SHAP explainer lookup |