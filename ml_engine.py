import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

DATA_FILE = "clean_training_data.csv"
MODEL_FILE = "expiry_model.pkl"
SCALER_FILE = "feature_scaler.pkl"
CATEGORY_COLS_FILE = "category_columns.json"

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
    """Trains a stacking ensemble model."""
    if not os.path.exists(DATA_FILE):
        return False
    df = pd.read_csv(DATA_FILE)

    X = _prepare_features(df)
    y = df['Risk_Label']

    category_cols = [c for c in X.columns if c.startswith('Cat_')]
    with open(CATEGORY_COLS_FILE, 'w') as f:
        json.dump(category_cols, f)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    joblib.dump(scaler, SCALER_FILE)

    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)),
        ('gb', GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, min_samples_split=10, min_samples_leaf=5,
            random_state=42
        )),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ]

    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1,
    )
    stacking_clf.fit(X_train_scaled, y_train)

    joblib.dump(stacking_clf, MODEL_FILE)
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

    avg_daily_sales = sales if sales > 0 else 0.001
    qty = stock if stock > 0 else 0.001
    stock_turnover_ratio = avg_daily_sales / qty
    price_per_unit_sold = price / avg_daily_sales

    numeric_values = [price, stock, sales, days_left,
                      stock_turnover_ratio, price_per_unit_sold]
    input_data = pd.DataFrame([numeric_values], columns=NUMERIC_FEATURES)

    for col in category_cols:
        input_data[col] = 0
    cat_col = f"Cat_{category}"
    if cat_col in category_cols:
        input_data[cat_col] = 1

    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

    risk_prob = clf.predict_proba(input_scaled)[0][1]

    if risk_prob > 0.7:
        risk_level = "High"
    elif risk_prob > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    result = {
        "Risk_Level": risk_level,
        "Probability": round(float(risk_prob), 2)
    }

    if risk_prob > 0.8:
        result["Action"] = "Donate to NGO"
    elif risk_level == "High":
        result["Action"] = "Discount 30%"
    elif risk_level == "Medium":
        result["Action"] = "Discount 15%"
    else:
        result["Action"] = "Keep Price"

    return result


if __name__ == "__main__":
    train_model()
