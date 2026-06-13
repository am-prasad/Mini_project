import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# 1. Load your raw CSV (assuming it has Q1-Q10 and Target_Category)
df = pd.read_csv("AQ of adoloscents - Sheet.csv")

# 2. Separate into features and target (using only raw questions)
question_cols = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]
X_raw = df[question_cols]
y = df["Target_Category"]

# 3. Perform your 80-20 train-test split (91 train rows, 23 test rows)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Apply SMOTE ONLY to the training partition
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 5. Rebuild the separate Training and Testing DataFrames
train_df = pd.DataFrame(X_train_res, columns=question_cols)
train_df["Target_Category"] = y_train_res

test_df = pd.DataFrame(X_test, columns=question_cols)
test_df["Target_Category"] = y_test.values


# 6. Create a helper function to calculate formula columns safely
def apply_formulas(dataframe):
    dataframe["CONTROL"] = (
        dataframe["Q1"] + dataframe["Q5"] + dataframe["Q9"]
    ) / 3
    dataframe["OWNERSHIP"] = (dataframe["Q2"] + dataframe["Q6"]) / 2
    dataframe["REACH"] = (dataframe["Q3"] + dataframe["Q7"]) / 2
    dataframe["ENDURANCE"] = (
        dataframe["Q4"] + dataframe["Q8"] + dataframe["Q10"]
    ) / 3

    # Organize column ordering
    final_cols = question_cols + [
        "CONTROL",
        "OWNERSHIP",
        "REACH",
        "ENDURANCE",
        "Target_Category",
    ]
    return dataframe[final_cols]


# 7. Apply formulas to both datasets
final_train = apply_formulas(train_df)
final_test = apply_formulas(test_df)

# 8. Save your balanced training set and untouched test set
final_train.to_csv("balanced_train_data.csv", index=False)
final_test.to_csv("clean_test_data.csv", index=False)

print(f"Balanced training set saved with {len(final_train)} rows.")
print(f"Untouched test set saved with {len(final_test)} rows.")