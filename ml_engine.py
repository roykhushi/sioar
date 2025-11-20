import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Constants
DATA_FILE = "clean_training_data.csv"
MODEL_FILE = "expiry_model.pkl"

def train_model():
    """Trains the Random Forest model and saves it."""
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run data_processor.py first.")
        return False

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Features and Target
    # We need to ensure we use the same features for training and prediction.
    # Based on the prompt's inference function signature: predict_item(price, stock, sales, days_left)
    # We should use these columns: 'Price', 'Quantity', 'Avg_Daily_Sales', 'Days_Until_Expiry'
    
    feature_cols = ['Price', 'Quantity', 'Avg_Daily_Sales', 'Days_Until_Expiry']
    X = df[feature_cols]
    y = df['Risk_Label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Save
    joblib.dump(clf, MODEL_FILE)
    print(f"Model saved to {MODEL_FILE}")
    return True

def predict_item(price, stock, sales, days_left):
    """
    Predicts risk level and recommends action.
    
    Args:
        price (float): Item price
        stock (int): Current quantity
        sales (float): Average daily sales (units)
        days_left (int): Days until expiry
        
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
    
    # Prepare input
    # Must match training features: ['Price', 'Quantity', 'Avg_Daily_Sales', 'Days_Until_Expiry']
    input_data = pd.DataFrame([[price, stock, sales, days_left]], 
                              columns=['Price', 'Quantity', 'Avg_Daily_Sales', 'Days_Until_Expiry'])
    
    # Predict
    risk_prob = clf.predict_proba(input_data)[0][1] # Probability of class 1 (Risk)
    risk_label = clf.predict(input_data)[0]
    
    # Logic for Action
    # High probability of risk (>80%) = Donate
    # Moderate risk (Risk=1 but prob <= 80%) = Discount
    # Low risk (Risk=0) = Keep Price
    
    result = {
        "Risk_Level": "Critical" if risk_label == 1 else "Safe",
        "Probability": round(risk_prob, 2)
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
