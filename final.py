# ===================================================================
# FINAL SCRIPT: AMEX DECISION TRACK
# ===================================================================

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

print("--- Starting Final Consolidated Script ---")

# --- 1. Load All Datasets ---
print("Step 1: Loading data...")
try:
    # Main data
    train_df = pd.read_parquet('data/train_data.parquet')
    test_df = pd.read_parquet('data/test_data.parquet')

    # Supplemental data
    add_trans_df = pd.read_parquet('data/add_trans.parquet')
    add_event_df = pd.read_parquet('data/add_event.parquet')
    offer_meta_df = pd.read_parquet('data/offer_metadata.parquet')
    submission_df = pd.read_csv('data/685404e30cfdb_submission_template.csv')

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure all files are in the correct directory.")
    exit()

# --- 2. Unify Key Data Types (Crucial Fix) ---
print("Step 2: Unifying key column data types...")
for df in [train_df, test_df, add_event_df, offer_meta_df]:
    if 'id3' in df.columns:
        df['id3'] = pd.to_numeric(df['id3'], errors='coerce')

for df in [train_df, test_df, add_trans_df, add_event_df]:
    if 'id2' in df.columns:
        df['id2'] = pd.to_numeric(df['id2'], errors='coerce')

# --- 3. Feature Engineering & Merging ---
print("Step 3: Performing feature engineering...")

# Clean and merge offer metadata
offer_meta_df.drop(columns=['f377', 'id11'], inplace=True, errors='ignore')
train_df = pd.merge(train_df, offer_meta_df, on='id3', how='left')
test_df = pd.merge(test_df, offer_meta_df, on='id3', how='left')

# Aggregate Transaction Features
add_trans_df['f370'] = pd.to_datetime(add_trans_df['f370'], errors='coerce')
last_trans_date = add_trans_df['f370'].max()
agg_trans = add_trans_df.groupby('id2').agg(
    avg_trans_amt=('f367', 'mean'),
    sum_trans_amt=('f367', 'sum'),
    total_transactions=('id2', 'count'),
    days_since_last_transaction=('f370', lambda x: (last_trans_date - x.max()).days)
)
train_df = pd.merge(train_df, agg_trans, on='id2', how='left')
test_df = pd.merge(test_df, agg_trans, on='id2', how='left')

# Aggregate Event Features
add_event_df['id4'] = pd.to_datetime(add_event_df['id4'], errors='coerce')
last_event_date = add_event_df['id4'].max()
agg_event = add_event_df.groupby('id2').agg(
    total_events=('id2', 'count'),
    nunique_offers_interacted=('id3', 'nunique'),
    days_since_last_event=('id4', lambda x: (last_event_date - x.max()).days)
)
train_df = pd.merge(train_df, agg_event, on='id2', how='left')
test_df = pd.merge(test_df, agg_event, on='id2', how='left')


# --- 4. Final Cleaning, Imputation, and Scaling ---
print("Step 4: Final cleaning, imputing, and scaling...")

# Separate target and identifiers
y = pd.to_numeric(train_df['y'], errors='coerce').fillna(0)
test_ids = test_df['id1'] # Save for submission
train_df = train_df.drop(columns=['id1', 'y', 'id4', 'id5'], errors='ignore')
test_df = test_df.drop(columns=['id1', 'id4', 'id5'], errors='ignore')

# Convert all object columns to category
for col in train_df.select_dtypes(include='object').columns:
    train_df[col] = train_df[col].astype('category')
    if col in test_df.columns:
        # Use train categories to define test categories
        test_df[col] = test_df[col].astype('category').cat.set_categories(train_df[col].cat.categories)

# Impute, Scale, and Prepare for Model
numerical_cols = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_df.select_dtypes(include='category').columns.tolist()

# Impute using training set statistics
for col in numerical_cols:
    median_val = train_df[col].median()
    train_df[col].fillna(median_val, inplace=True)
    test_df[col].fillna(median_val, inplace=True)

for col in categorical_cols:
    # --- Robust Categorical Imputation ---

# Get the mode of the training column
    modes = train_df[col].mode()

    # Check if the mode series is not empty
    if not modes.empty:
        # If modes exist, use the first one as the fill value
        mode_val = modes[0]
        train_df[col].fillna(mode_val, inplace=True)
        test_df[col].fillna(mode_val, inplace=True)
    else:
        # If the column was all NaN, fill with a placeholder like "Unknown"
        # This requires adding "Unknown" to the categories first.
        train_df[col] = train_df[col].cat.add_categories("Unknown")
        test_df[col] = test_df[col].cat.add_categories("Unknown")
        train_df[col].fillna("Unknown", inplace=True)
        test_df[col].fillna("Unknown", inplace=True)

# Scale numerical features
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


# --- 5. Model Training ---
print("Step 5: Training LightGBM Model...")
# Calculate weight for handling class imbalance
scale_pos_weight = y.value_counts().get(0, 1) / y.value_counts().get(1, 1)

lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 2000, 'learning_rate': 0.02, 'num_leaves': 31,
    'max_depth': 7, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.8, 'subsample': 0.8, 'reg_alpha': 0.1,
    'reg_lambda': 0.1, 'scale_pos_weight': scale_pos_weight
}

model = lgb.LGBMClassifier(**lgb_params)
model.fit(train_df, y, eval_set=[(train_df, y)],
          callbacks=[lgb.early_stopping(100, verbose=True)],
          categorical_feature=categorical_cols)


# --- 6. Prediction and Submission ---
print("Step 6: Generating Predictions and Creating Submission File...")
predictions = model.predict_proba(test_df)[:, 1]

submission_df['pred'] = predictions
submission_df.to_csv('r2_submission_file_Gemini.csv', index=False)

# --- 7. Final Output ---
print("\n" + "="*50 + "\n")
print("✅ Script finished successfully!")
print("Submission file 'r2_submission_file_Gemini.csv' has been created.")
print("\n--- Model and Submission File Output ---")
print(f"\nTrained Model:\n{model}")
print(f"\nTop 5 rows of the submission file:")
print(submission_df.head())
print("\n" + "="*50)