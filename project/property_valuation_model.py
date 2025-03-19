import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
import logging
import re
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyValuationModel:
    """
    A property valuation model using XGBoost for accurate price predictions.
    Includes feature engineering, model training, evaluation, and prediction functionality.
    """
    
    def __init__(self, model_path='models/property_valuation_model.pkl', db_path='data/property_db.sqlite'):
        """Initialize the property valuation model."""
        self.model = None
        self.feature_encoders = {}
        self.model_path = model_path
        self.db_path = db_path
        self.feature_importance = None
        
        # Define feature groups based on the database schema
        self.categorical_features = ['city', 'neighborhood']
        self.numerical_features = ['total_area', 'baths']
        self.binary_features = ['has_balcony']
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    def preprocess_data(self, df):
        """Preprocess the data for training or prediction."""
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Handle missing values
        for feature in self.numerical_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna(data[feature].median())
        
        for feature in self.categorical_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna('Unknown')
        
        for feature in self.binary_features:
            if feature in data.columns:
                data[feature] = data[feature].fillna(False)
        
        # Extract BHK from title if not already present
        if 'bedrooms' not in data.columns and 'title' in data.columns:
            data['bedrooms'] = data['title'].apply(self._extract_bhk)
        
        return data
    
    def _extract_bhk(self, title):
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
    
    def _create_preprocessing_pipeline(self):
        """Create a preprocessing pipeline for categorical and numerical features."""
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # Get the actual columns present in the training data
        cat_cols = [col for col in self.categorical_features if col in self.X_train.columns]
        num_cols = [col for col in self.numerical_features if col in self.X_train.columns]
        
        # Add bedrooms to numerical features if present
        if 'bedrooms' in self.X_train.columns:
            num_cols.append('bedrooms')
        
        # Explicitly list columns to drop
        text_columns = ['title', 'location', 'price']
        drop_cols = [col for col in text_columns if col in self.X_train.columns]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, cat_cols),
                ('num', numerical_transformer, num_cols),
                ('drop', 'drop', drop_cols)
            ],
            remainder='drop'  # Drop any other columns not explicitly handled
        )
        
        return preprocessor
    
    def train(self, data=None, target_column='price_value', test_size=0.2, random_state=42):
        """
        Train the XGBoost model on the provided data or from the database.
        
        Args:
            data: DataFrame with property data (optional, will load from DB if None)
            target_column: Column name for the target variable
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        logger.info("Starting model training process")
        
        # If no data provided, load from database
        if data is None:
            data = self._load_data_from_db()
            if data is None or len(data) == 0:
                logger.error("Failed to load data from database")
                return None
        
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        
        # Explicitly select only the columns we want to use for modeling
        usable_columns = ['city', 'neighborhood', 'price_value', 'total_area', 'baths', 'bedrooms', 'has_balcony']
        modeling_data = processed_data[usable_columns]
        
        # Separate features and target
        X = modeling_data.drop(columns=[target_column])
        y = modeling_data[target_column]
        
        logger.info(f"Training data shape: {X.shape}")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create preprocessing pipeline
        preprocessor = self._create_preprocessing_pipeline()
        
        # Create and train the model
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state
            ))
        ])
        
        logger.info("Fitting model...")
        self.model.fit(self.X_train, self.y_train)
        
        # Extract feature importance
        xgb_model = self.model.named_steps['regressor']
        preprocessor = self.model.named_steps['preprocessor']
        
        try:
            # Get feature names after preprocessing
            feature_names = []
            for name, trans, cols in preprocessor.transformers_:
                if name != 'drop' and hasattr(trans, 'get_feature_names_out'):
                    feature_names.extend(trans.get_feature_names_out(cols))
                elif name != 'drop':
                    feature_names.extend(cols)
            
            # Create a mapping of feature names to user-friendly names
            feature_name_mapping = {}
            for feature in feature_names:
                if feature.startswith('cat__'):
                    # For one-hot encoded categorical features
                    parts = feature.split('__')
                    if len(parts) > 1:
                        category_value = parts[1].split('_')
                        if len(category_value) > 1:
                            category = category_value[0]
                            value = '_'.join(category_value[1:])
                            if category == 'city':
                                feature_name_mapping[feature] = f"City: {value}"
                            elif category == 'neighborhood':
                                feature_name_mapping[feature] = f"Area: {value}"
                elif feature.startswith('num__'):
                    # For numerical features
                    parts = feature.split('__')
                    if len(parts) > 1:
                        if parts[1] == 'total_area':
                            feature_name_mapping[feature] = "Total Area"
                        elif parts[1] == 'baths':
                            feature_name_mapping[feature] = "Number of Bathrooms"
                        elif parts[1] == 'bedrooms':
                            feature_name_mapping[feature] = "Number of Bedrooms"
                elif feature == 'has_balcony':
                    feature_name_mapping[feature] = "Has Balcony"
                else:
                    # Default mapping
                    feature_name_mapping[feature] = feature
            
            # Make sure feature_names length matches feature importances length
            if len(feature_names) == len(xgb_model.feature_importances_):
                # Create feature importance DataFrame with user-friendly names
                importance_data = []
                for i, (feature, importance) in enumerate(zip(feature_names, xgb_model.feature_importances_)):
                    friendly_name = feature_name_mapping.get(feature, f"Property Factor {i+1}")
                    importance_data.append({
                        'feature': friendly_name,
                        'importance': importance,
                        'original_feature': feature
                    })
                
                self.feature_importance = pd.DataFrame(importance_data).sort_values('importance', ascending=False)
            else:
                logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importances length ({len(xgb_model.feature_importances_)})")
                # Create with generic but more meaningful feature names
                self.feature_importance = pd.DataFrame({
                    'feature': [f"Property Factor {i+1}" for i in range(len(xgb_model.feature_importances_))],
                    'importance': xgb_model.feature_importances_,
                    'original_feature': [f"feature_{i}" for i in range(len(xgb_model.feature_importances_))]
                }).sort_values('importance', ascending=False)
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
            self.feature_importance = None
        
        logger.info("Model training completed")
        
        # Evaluate the model
        self.evaluate()
        
        # Save the model
        self.save_model()
        
        return self
    
    def _load_data_from_db(self):
        """Load property data from SQLite database."""
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.db_path)
            
            # Query all properties
            query = """
            SELECT 
                id, title, location, city, neighborhood, 
                price, price_value, total_area, baths, has_balcony
            FROM properties
            """
            
            # Load into DataFrame
            df = pd.read_sql_query(query, conn)
            
            # Close connection
            conn.close()
            
            # Convert has_balcony to boolean
            df['has_balcony'] = df['has_balcony'].astype(bool)
            
            # Extract bedrooms from title
            df['bedrooms'] = df['title'].apply(self._extract_bhk)
            
            logger.info(f"Loaded {len(df)} properties from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {str(e)}")
            return None
    
    def evaluate(self):
        """Evaluate the model on the test set."""
        logger.info("Evaluating model performance")
        
        # Make predictions on the test set
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        logger.info(f"Mean Absolute Error: {mae:.2f}")
        logger.info(f"Root Mean Squared Error: {rmse:.2f}")
        logger.info(f"R² Score: {r2:.4f}")
        
        # Calculate percentage error
        percentage_error = np.abs(self.y_test - y_pred) / self.y_test * 100
        mean_percentage_error = percentage_error.mean()
        median_percentage_error = np.median(percentage_error)
        
        logger.info(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
        logger.info(f"Median Percentage Error: {median_percentage_error:.2f}%")
        
        # Store evaluation metrics
        self.evaluation_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_percentage_error': mean_percentage_error,
            'median_percentage_error': median_percentage_error
        }
        
        return self.evaluation_metrics
    
    def predict(self, property_data):
        """
        Predict the value of a property.
        
        Args:
            property_data: DataFrame or dict with property features
            
        Returns:
            dict: Prediction results including estimated value and confidence interval
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                logger.error("Model not loaded and could not be loaded from disk")
                return None
        
        # Convert dict to DataFrame if needed
        if isinstance(property_data, dict):
            property_data = pd.DataFrame([property_data])
        
        # Preprocess the data
        processed_data = self.preprocess_data(property_data)
        
        # Make prediction
        predicted_value = self.model.predict(processed_data)[0]
        
        # Calculate confidence interval (using a simple approach)
        confidence_percentage = 10  # 10% confidence interval
        confidence_range = predicted_value * (confidence_percentage / 100)
        
        # Get top features that influenced this prediction
        if hasattr(self.model, 'named_steps') and self.feature_importance is not None:
            top_features = self.feature_importance.head(5).to_dict('records')
        else:
            top_features = []
        
        return {
            'estimated_value': float(predicted_value),
            'min_value': float(predicted_value - confidence_range),
            'max_value': float(predicted_value + confidence_range),
            'confidence_percentage': confidence_percentage,
            'top_features': top_features
        }
    
    def save_model(self):
        """Save the model to disk."""
        logger.info(f"Saving model to {self.model_path}")
        joblib.dump({
            'model': self.model,
            'feature_importance': self.feature_importance,
            'evaluation_metrics': getattr(self, 'evaluation_metrics', None),
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'binary_features': self.binary_features
        }, self.model_path)
    
    def load_model(self):
        """Load the model from disk."""
        logger.info(f"Loading model from {self.model_path}")
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_importance = model_data['feature_importance']
            self.evaluation_metrics = model_data.get('evaluation_metrics')
            self.categorical_features = model_data.get('categorical_features', self.categorical_features)
            self.numerical_features = model_data.get('numerical_features', self.numerical_features)
            self.binary_features = model_data.get('binary_features', self.binary_features)
            return True
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def plot_feature_importance(self, top_n=10):
        """Plot feature importance."""
        if self.feature_importance is None:
            logger.warning("Feature importance not available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        top_features = self.feature_importance.head(top_n)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_prediction_vs_actual(self):
        """Plot predicted vs actual values."""
        if not hasattr(self, 'X_test') or not hasattr(self, 'y_test'):
            logger.warning("Test data not available. Train the model first.")
            return
        
        y_pred = self.model.predict(self.X_test)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Property Prices')
        plt.tight_layout()
        return plt.gcf()
    
    def get_comparable_properties(self, property_data, n=5):
        """
        Find comparable properties based on similarity.
        
        Args:
            property_data: Dict with property features
            n: Number of comparable properties to return
            
        Returns:
            list: n most similar properties
        """
        try:
            # Connect to SQLite database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract key features for comparison
            if isinstance(property_data, dict):
                city = property_data.get('city', '')
                neighborhood = property_data.get('neighborhood', '')
                total_area = property_data.get('total_area', 0)
                bedrooms = property_data.get('bedrooms', 2)
                baths = property_data.get('bathrooms', 1)  # Note: 'bathrooms' in input, 'baths' in DB
                price_value = property_data.get('price_value', 0)
                
                # If price_value not provided but we have a prediction, use that
                if price_value == 0 and self.model is not None:
                    prediction = self.predict(property_data)
                    if prediction:
                        price_value = prediction['estimated_value']
            else:
                # If DataFrame, use first row
                row = property_data.iloc[0]
                city = row.get('city', '')
                neighborhood = row.get('neighborhood', '')
                total_area = row.get('total_area', 0)
                bedrooms = row.get('bedrooms', 2)
                baths = row.get('bathrooms', 1) if 'bathrooms' in row else row.get('baths', 1)
                price_value = row.get('price_value', 0)
            
            # Build query to find similar properties
            query = """
            SELECT 
                id, title, location, city, neighborhood, 
                price, price_value, total_area, baths, has_balcony
            FROM properties
            WHERE 1=1
            """
            params = []
            
            # Add filters based on property features
            if city:
                query += " AND city LIKE ?"
                params.append(f"%{city}%")
            
            # Price range (±20%)
            if price_value > 0:
                min_price = price_value * 0.8
                max_price = price_value * 1.2
                query += " AND price_value BETWEEN ? AND ?"
                params.extend([min_price, max_price])
            
            # Area range (±20%)
            if total_area > 0:
                min_area = total_area * 0.8
                max_area = total_area * 1.2
                query += " AND total_area BETWEEN ? AND ?"
                params.extend([min_area, max_area])
            
            # Same number of bathrooms
            if baths > 0:
                query += " AND baths = ?"
                params.append(baths)
            
            # Order by similarity to the target property
            query += " ORDER BY ABS(price_value - ?) + ABS(total_area - ?) * 100"
            params.extend([price_value, total_area])
            
            # Limit results
            query += f" LIMIT {n}"
            
            # Execute query
            cursor.execute(query, params)
            
            # Format results
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'title': row[1],
                    'location': row[2],
                    'city': row[3],
                    'neighborhood': row[4],
                    'price': row[5],
                    'price_value': row[6],
                    'total_area': row[7],
                    'baths': row[8],
                    'has_balcony': bool(row[9])
                })
            
            conn.close()
            return results
            
        except Exception as e:
            logger.error(f"Error getting comparable properties: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = PropertyValuationModel()
    
    # Train the model using data from the database
    model.train()
    
    # Make a prediction
    sample_property = {
        'city': 'Chennai',
        'neighborhood': 'Adyar',
        'bedrooms': 3,
        'bathrooms': 2,
        'total_area': 1500,
        'has_balcony': True
    }
    
    prediction = model.predict(sample_property)
    print(f"Estimated value: ₹{prediction['estimated_value']:,.2f}")
    print(f"Range: ₹{prediction['min_value']:,.2f} - ₹{prediction['max_value']:,.2f}")
    
    # Find comparable properties
    comparables = model.get_comparable_properties(sample_property)
    print("\nComparable properties:")
    for i, prop in enumerate(comparables, 1):
        print(f"\n{i}. {prop['title']}")
        print(f"   Location: {prop['location']}")
        print(f"   Price: {prop['price']}")
        print(f"   Area: {prop['total_area']} sq ft, Baths: {prop['baths']}") 