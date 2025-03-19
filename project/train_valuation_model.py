import pandas as pd
import numpy as np
import os
import logging
import re
import matplotlib.pyplot as plt
import seaborn as sns
from property_valuation_model import PropertyValuationModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Train the property valuation model using the real estate dataset."""
    logger.info("Starting property valuation model training")
    
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Check if models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load the dataset
    csv_path = 'data/Real Estate Data V21.csv'
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found at {csv_path}")
        return
    
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Display dataset info
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Check column names to ensure they match what we expect
    expected_columns = ['Property Title', 'Location', 'Price', 'Total_Area', 'Price_per_SQFT', 'Baths', 'Description', 'Balcony']
    missing_columns = [col for col in expected_columns if col not in df.columns]
    
    if missing_columns:
        logger.warning(f"Missing expected columns: {missing_columns}")
        # Try to find similar column names
        for missing_col in missing_columns:
            similar_cols = [col for col in df.columns if missing_col.lower() in col.lower()]
            if similar_cols:
                logger.info(f"Found similar columns for {missing_col}: {similar_cols}")
    
    # Rename columns if needed to match expected format
    column_mapping = {}
    for expected_col in expected_columns:
        if expected_col not in df.columns:
            # Look for similar column names
            similar_cols = [col for col in df.columns if expected_col.lower() in col.lower()]
            if similar_cols:
                column_mapping[similar_cols[0]] = expected_col
    
    if column_mapping:
        logger.info(f"Renaming columns: {column_mapping}")
        df = df.rename(columns=column_mapping)
    
    # Clean and prepare data
    logger.info("Preparing data for training")
    
    # Clean price data (remove ₹ and Cr/L, convert to numerical)
    df['price_value'] = df['Price'].apply(clean_price)
    
    # Extract location components
    df['city'] = df['Location'].apply(lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else x)
    df['neighborhood'] = df['Location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else 'Unknown')
    
    # Extract bedrooms from title
    df['bedrooms'] = df['Property Title'].apply(extract_bhk)
    
    # Convert balcony to boolean
    df['has_balcony'] = df['Balcony'].apply(lambda x: x == 'Yes' if isinstance(x, str) else False)
    
    # Clean data
    df = df.dropna(subset=['price_value', 'Total_Area'])
    df = df[(df['price_value'] > 100000) & (df['price_value'] < 100000000)]  # Remove outliers
    df = df[(df['Total_Area'] > 100) & (df['Total_Area'] < 10000)]  # Remove outliers
    
    # Prepare final training dataset
    training_data = pd.DataFrame({
        'title': df['Property Title'],
        'location': df['Location'],
        'city': df['city'],
        'neighborhood': df['neighborhood'],
        'price': df['Price'],
        'price_value': df['price_value'],
        'total_area': df['Total_Area'],
        'baths': df['Baths'],
        'bedrooms': df['bedrooms'],
        'has_balcony': df['has_balcony']
    })
    
    logger.info(f"Prepared training data with {len(training_data)} rows")
    
    # Initialize and train the model
    model = PropertyValuationModel()
    model.train(training_data)
    
    # Evaluate the model
    metrics = model.evaluate()
    logger.info(f"Model evaluation metrics: {metrics}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    model.plot_feature_importance()
    plt.savefig('models/feature_importance.png')
    logger.info("Feature importance plot saved to models/feature_importance.png")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    model.plot_prediction_vs_actual()
    plt.savefig('models/actual_vs_predicted.png')
    logger.info("Actual vs predicted plot saved to models/actual_vs_predicted.png")
    
    logger.info("Model training completed successfully")

def clean_price(price):
    """Clean price data by removing symbols and converting to numerical value."""
    try:
        if not isinstance(price, str):
            return float(price) if not np.isnan(price) else 0
            
        # Remove ₹ symbol and spaces
        price = price.replace('₹', '').strip()
        
        # Convert Cr to actual number
        if 'Cr' in price:
            return float(price.replace('Cr', '')) * 10000000
        # Convert L to actual number
        elif 'L' in price:
            return float(price.replace('L', '')) * 100000
        return float(price)
    except:
        return 0

def extract_bhk(title):
    """Extract number of bedrooms from property title."""
    if not isinstance(title, str):
        return 2  # Default value
    
    # Look for patterns like "3BHK", "3 BHK", "3 Bedroom"
    bhk_match = re.search(r'(\d+)\s*BHK', title, re.IGNORECASE)
    bedroom_match = re.search(r'(\d+)\s*Bedroom', title, re.IGNORECASE)
    
    if bhk_match:
        return int(bhk_match.group(1))
    elif bedroom_match:
        return int(bedroom_match.group(1))
    else:
        return 2  # Default value

if __name__ == "__main__":
    main() 