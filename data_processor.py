import pandas as pd
import numpy as np
import os


DATA_FILE = "Grocery_Inventory_and_Sales_Dataset.csv"
OUTPUT_FILE = "clean_training_data.csv"

def load_data(filepath):
    if os.path.exists(filepath):
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Column Mapping & Cleaning
        # Expected: Item, Price, Quantity, Total_Sales, Category
        # Actual: Product_Name, Unit_Price, Stock_Quantity, Sales_Volume, Catagory
        
        rename_map = {
            'Product_Name': 'Item',
            'Unit_Price': 'Price',
            'Stock_Quantity': 'Quantity',
            'Sales_Volume': 'Total_Sales',
            'Catagory': 'Category'
        }
        
        # Check if columns exist before renaming to avoid errors if file changes
        available_cols = df.columns.tolist()
        actual_rename = {k: v for k, v in rename_map.items() if k in available_cols}
        df.rename(columns=actual_rename, inplace=True)
        
        # Clean Price (remove '$')
        if 'Price' in df.columns and df['Price'].dtype == 'object':
            df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False).astype(float)
            
        return df
    else:
        print(f"File {filepath} not found. Generating synthetic data...")
        data = {
            'Item': [f'Item_{i}' for i in range(100)],
            'Price': np.random.uniform(1.0, 20.0, 100),
            'Quantity': np.random.randint(1, 100, 100),
            'Total_Sales': np.random.uniform(100.0, 5000.0, 100),
            'Category': np.random.choice(['Dairy', 'Bakery', 'Canned', 'Produce', 'Meat'], 100)
        }
        return pd.DataFrame(data)

def generate_expiry_days(row):
    """Generates synthetic Days_Until_Expiry based on category."""
    category_expiry_mean = {
        'Dairy': 7,
        'Bakery': 3,
        'Produce': 5,
        'Meat': 4,
        'Canned': 365,
        'Frozen': 180,
        'Beverages': 90,
        'Snacks': 60,
        'Grains & Pulses': 365 # Added from sample data
    }
    
    cat = row.get('Category', 'Unknown')
    # Handle cases where category might be slightly different or missing
    if not isinstance(cat, str):
        cat = 'Unknown'
        
    # Simple matching
    mean_days = 30
    for key, val in category_expiry_mean.items():
        if key.lower() in cat.lower():
            mean_days = val
            break
            
    # Add random variance (std dev = 20% of mean)
    days = int(np.random.normal(mean_days, mean_days * 0.2))
    return max(1, days)

def process_data():
    df = load_data(DATA_FILE)
    
    # 1. Ensure 'Category' exists
    if 'Category' not in df.columns:
        categories = ['Dairy', 'Bakery', 'Produce', 'Meat', 'Canned', 'Frozen', 'Beverages', 'Snacks']
        df['Category'] = np.random.choice(categories, len(df))

    # 2. Generate Days_Until_Expiry (CRITICAL FIX from prompt)
    # Even if Expiration_Date exists, we follow the prompt's instruction to synthetically generate it
    # based on category to ensure the logic is implemented as requested.
    df['Days_Until_Expiry'] = df.apply(generate_expiry_days, axis=1)

    # 3. Feature Engineering: Risk_Label
    # Logic: If (Current_Stock / Avg_Daily_Sales) > Days_Until_Expiry, then Risk = 1
    
    # Calculate Avg_Daily_Sales
    # Assuming Total_Sales (Sales_Volume) is units sold over 30 days
    # If Total_Sales was revenue, we'd divide by Price. 
    # But 'Sales_Volume' usually implies units. Let's assume units.
    
    if 'Total_Sales' in df.columns:
        # If it's volume (units)
        df['Avg_Daily_Sales'] = df['Total_Sales'] / 30.0
    else:
        # Fallback
        df['Avg_Daily_Sales'] = np.random.uniform(0.1, 10.0, len(df))

    # Avoid division by zero
    df['Avg_Daily_Sales'] = df['Avg_Daily_Sales'].replace(0, 0.001)

    df['Days_To_Sell_Stock'] = df['Quantity'] / df['Avg_Daily_Sales']
    
    df['Risk_Label'] = (df['Days_To_Sell_Stock'] > df['Days_Until_Expiry']).astype(int)

    # 4. Clean Data
    # Fill missing values
    df.fillna(0, inplace=True)
    
    # Select only necessary columns for training/output
    cols_to_keep = ['Item', 'Price', 'Quantity', 'Avg_Daily_Sales', 'Days_Until_Expiry', 'Risk_Label', 'Category']
    # Only keep columns that exist
    cols_to_keep = [c for c in cols_to_keep if c in df.columns]
    
    df_clean = df[cols_to_keep]
    
    # Save
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f"Processed data saved to {OUTPUT_FILE}")
    print(df_clean.head())

if __name__ == "__main__":
    process_data()
