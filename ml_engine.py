import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import os
import json

# Constants
DATA_FILE = "clean_training_data.csv"
MODEL_FILE = "expiry_model.pkl"
SCALER_FILE = "feature_scaler.pkl"
CATEGORY_COLS_FILE = "category_columns.json"

# Numeric feature columns (must match data_processor output)
NUMERIC_FEATURES = [
    'Price', 'Quantity', 'Avg_Daily_Sales', 'Days_Until_Expiry',
    'Stock_Turnover_Ratio', 'Price_Per_Unit_Sold'
]


def _prepare_features(df):
    """One-hot encode Category and return feature DataFrame + list of all feature column names."""
    cat_dummies = pd.get_dummies(df['Category'], prefix='Cat')
    feature_df = pd.concat([df[NUMERIC_FEATURES], cat_dummies], axis=1)
    return feature_df


def train_model():
    """Trains a Stacking ensemble and prints each base model's accuracy."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run data_processor.py first.")
        return False

    print("=" * 60)
    print("  SMART-FOOD LINK — MODEL TRAINING")
    print("=" * 60)

    print("\n📂 Loading data...")
    df = pd.read_csv(DATA_FILE)

    X = _prepare_features(df)
    y = df['Risk_Label']

    # Save the category column names so predict_item can recreate the same shape
    category_cols = [c for c in X.columns if c.startswith('Cat_')]
    with open(CATEGORY_COLS_FILE, 'w') as f:
        json.dump(category_cols, f)

    # ── Stratified split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale features ──────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    joblib.dump(scaler, SCALER_FILE)

    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ]

    print("\n🔧 Training individual base models...\n")
    print(f"  {'Model':25s} {'Train Acc':>10s}  {'Test Acc':>10s}")
    print("-" * 50)

    for name, estimator in base_estimators:
        est_clone = type(estimator)(**estimator.get_params())
        est_clone.fit(X_train_scaled, y_train)
        y_pred_train = est_clone.predict(X_train_scaled)
        y_pred_test = est_clone.predict(X_test_scaled)
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        label = {
            'rf': 'Random Forest',
            'gb': 'Gradient Boosting',
            'lr': 'Logistic Regression',
        }[name]
        print(f"  {label:25s} {train_acc:>10.4f}  {test_acc:>10.4f}")

    print("-" * 50)

    # ── Train stacking ensemble ─────────────────────────────────
    print("\n🏗️  Training Stacking Ensemble...")

    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
    )
    stacking_clf.fit(X_train_scaled, y_train)

    y_pred_stack_train = stacking_clf.predict(X_train_scaled)
    y_pred_stack_test = stacking_clf.predict(X_test_scaled)
    stack_train_acc = accuracy_score(y_train, y_pred_stack_train)
    stack_test_acc = accuracy_score(y_test, y_pred_stack_test)

    print(f"\n  {'Stacked Model':25s} {stack_train_acc:>10.4f}  {stack_test_acc:>10.4f}")
    print("-" * 50)

    # ── Cross-validation score ──────────────────────────────────
    print("\n📈 5-Fold Cross-Validation (Stacked Model)...")
    cv_scores = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"  CV Scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print("\n📊 Classification Report (Stacked Model):\n")
    print(classification_report(y_test, y_pred_stack_test))

    joblib.dump(stacking_clf, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")
    print(f"✅ Scaler saved to {SCALER_FILE}")
    print(f"✅ Category columns saved to {CATEGORY_COLS_FILE}")
    print("=" * 60)
    return True


def predict_item(price, stock, sales, days_left, category="Unknown"):
    """
    Predicts risk level and recommends action using the stacking ensemble.

    Args:
        price (float): Item price
        stock (int): Current quantity
        sales (float): Average daily sales (units)
        days_left (int): Days until expiry
        category (str): Food category (e.g. 'Dairy', 'Bakery')

    Returns:
        dict: {
            "Risk_Level": "Safe" | "Critical",
            "Probability": float,
            "Action": "Keep Price" | "Discount 30%" | "Donate to NGO"
        }
    """
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model file not found. Please train the model first.")

    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    with open(CATEGORY_COLS_FILE, 'r') as f:
        category_cols = json.load(f)

    # Compute derived features (same logic as data_processor)
    avg_daily_sales = sales if sales > 0 else 0.001
    qty = stock if stock > 0 else 0.001
    stock_turnover_ratio = avg_daily_sales / qty
    price_per_unit_sold = price / avg_daily_sales

    # Build numeric row
    numeric_values = [price, stock, sales, days_left,
                      stock_turnover_ratio, price_per_unit_sold]
    input_data = pd.DataFrame([numeric_values], columns=NUMERIC_FEATURES)

    # One-hot encode category to match training columns
    for col in category_cols:
        input_data[col] = 0
    cat_col = f"Cat_{category}"
    if cat_col in category_cols:
        input_data[cat_col] = 1

    # Scale
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    # Predict
    risk_prob = clf.predict_proba(input_scaled)[0][1]  # Probability of class 1 (Risk)
    risk_label = clf.predict(input_scaled)[0]

    # Action logic
    result = {
        "Risk_Level": "Critical" if risk_label == 1 else "Safe",
        "Probability": round(float(risk_prob), 2)
    }

    if risk_prob > 0.8:
        result["Action"] = "Donate to NGO"
    elif risk_label == 1:
        result["Action"] = "Discount 30%"
    else:
        result["Action"] = "Keep Price"

    return result


if __name__ == "__main__":
    train_model()
