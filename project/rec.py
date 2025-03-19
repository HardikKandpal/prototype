import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache
import logging
import re
import sqlite3
import os
import faiss
import pickle
import spacy
from spacy.language import Language
from spacy.tokens import Doc

logger = logging.getLogger(__name__)

# Create a simple spaCy pipeline if full model not available
@Language.factory("simple_ner")
def create_simple_ner(nlp, name):
    return SimpleNER()

class SimpleNER:
    def __init__(self):
        # List of common Indian cities
        self.common_cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 
            'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Surat',
            'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane',
            'Bhopal', 'Visakhapatnam', 'Pimpri-Chinchwad', 'Patna', 'Vadodara',
            'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik', 'Faridabad',
            'Meerut', 'Rajkot', 'Kalyan-Dombivli', 'Vasai-Virar', 'Varanasi'
        ]
    
    def __call__(self, doc):
        # Find cities in text
        for city in self.common_cities:
            if city.lower() in doc.text.lower():
                start = doc.text.lower().find(city.lower())
                end = start + len(city)
                span = doc[start:end]
                if len(span) > 0:  # Ensure valid span
                    span.label_ = "GPE"
        return doc

class PropertyRecommender:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.db_path = 'data/property_db.sqlite'
        self.faiss_index_path = 'data/faiss_index.bin'
        self.embeddings_path = 'data/embeddings.pkl'
        
        # Initialize spaCy for location extraction
        try:
            # Try to load full English model
            self.nlp = spacy.load("en_core_web_sm")
            print("Loaded full spaCy model")
        except OSError:
            # Fall back to blank model with custom NER
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("simple_ner")
            print("Using simple spaCy model with custom NER")
        
        self.prepare_data()
        
    def prepare_data(self):
        # Store only necessary columns
        required_columns = ['Property Title', 'Location', 'Price', 
                           'Total_Area', 'Price_per_SQFT', 'Baths', 
                           'Description', 'Balcony']
        self.df = self.df[required_columns].copy()
        
        # Clean price data (remove ₹ and Cr/L, convert to numerical)
        self.df['Price_Cleaned'] = self.df['Price'].apply(self._clean_price)
        
        # Extract location components using spaCy
        print("Extracting location components...")
        self.df['City'], self.df['Neighborhood'] = zip(*self.df['Location'].apply(self._extract_location_components))
        
        # Create a combined features text for TF-IDF
        self.df['Combined_Features'] = self.df.apply(self._combine_features, axis=1)
        
        # Initialize SQLite database
        self._init_database()
        
        # Create FAISS index if it doesn't exist
        if not os.path.exists(self.faiss_index_path) or not os.path.exists(self.embeddings_path):
            self._create_faiss_index()
        else:
            # Load existing FAISS index
            self.index = faiss.read_index(self.faiss_index_path)
            with open(self.embeddings_path, 'rb') as f:
                self.property_embeddings = pickle.load(f)
    
    def _init_database(self):
        """Initialize SQLite database with property data"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect to SQLite database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create properties table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY,
            title TEXT,
            location TEXT,
            city TEXT,
            neighborhood TEXT,
            price TEXT,
            price_value REAL,
            total_area REAL,
            baths INTEGER,
            has_balcony INTEGER
        )
        ''')
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM properties")
        count = cursor.fetchone()[0]
        
        # Insert data if table is empty
        if count == 0:
            for idx, row in self.df.iterrows():
                has_balcony = 1 if row['Balcony'] == 'Yes' else 0
                cursor.execute('''
                INSERT INTO properties (id, title, location, city, neighborhood, price, price_value, total_area, baths, has_balcony)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    idx,
                    row['Property Title'],
                    row['Location'],
                    row['City'],
                    row['Neighborhood'],
                    row['Price'],
                    row['Price_Cleaned'],
                    row['Total_Area'],
                    row['Baths'],
                    has_balcony
                ))
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Database initialized with {len(self.df)} properties")
    
    def _create_faiss_index(self):
        """Create FAISS index for fast similarity search"""
        # Create TF-IDF matrix for text features
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.df['Combined_Features'])
        
        # Prepare numerical features
        numerical_features = ['Price_Cleaned', 'Total_Area', 'Price_per_SQFT', 'Baths']
        scaler = MinMaxScaler()
        numerical_matrix = scaler.fit_transform(self.df[numerical_features])
        
        # Combine TF-IDF and numerical features
        combined_features = np.hstack((
            tfidf_matrix.toarray(),
            numerical_matrix
        ))
        
        # Convert to float32 for FAISS
        self.property_embeddings = combined_features.astype(np.float32)
        
        # Create FAISS index
        dimension = self.property_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.property_embeddings)
        
        # Save FAISS index and embeddings
        os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
        faiss.write_index(self.index, self.faiss_index_path)
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump(self.property_embeddings, f)
        
        print(f"FAISS index created with {len(self.df)} properties and {dimension} dimensions")
    
    def _extract_location_components(self, location):
        """Extract city and neighborhood from location using spaCy"""
        if not location or not isinstance(location, str):
            return None, None
            
        # Process with spaCy
        doc = self.nlp(location)
        
        # Extract locations (GPE entities)
        locations = [ent.text for ent in doc.ents if getattr(ent, 'label_', '') == "GPE"]
        
        # If no entities found, fall back to comma splitting
        if not locations:
            parts = location.split(',')
            if len(parts) > 1:
                # Last part is usually the city
                city = parts[-1].strip()
                # First part is usually the neighborhood
                neighborhood = parts[0].strip()
                return city, neighborhood
            return location, ""
        
        # If entities found, last is usually the city
        if len(locations) > 1:
            return locations[-1], locations[0]
        elif locations:
            return locations[0], ""
        
        return None, None
    
    def _clean_price(self, price):
        try:
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
    
    def _combine_features(self, row):
        # Combine relevant features into a single string
        features = [
            str(row['Location']),
            str(row['City']),
            str(row['Neighborhood']),
            str(row['Description']),
            'balcony' if row['Balcony'] == 'Yes' else 'no balcony'
        ]
        return ' '.join(features)
    
    def search_properties(self, filters=None, limit=10):
        """Search properties based on filters with flexible matching"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Start with base query
            query = "SELECT id, title, location, city, neighborhood, price, price_value, total_area, baths, has_balcony FROM properties WHERE 1=1"
            params = []
            
            # Add filters if provided
            if filters:
                # Location filter (highest priority)
                if filters.get('location'):
                    location = filters['location'].strip()
                    if location:
                        # Process the location with spaCy to extract city
                        doc = self.nlp(location)
                        cities = [ent.text for ent in doc.ents if getattr(ent, 'label_', '') == "GPE"]
                        
                        # If spaCy found a city, prioritize city matching
                        if cities:
                            city = cities[-1]  # Last entity is usually the city
                            print(f"Extracted city from input: {city}")
                            query += " AND (city LIKE ? OR location LIKE ?)"
                            params.extend([f"%{city}%", f"%{location}%"])
                        else:
                            # Try to match with city, neighborhood, or location
                            query += " AND (city LIKE ? OR neighborhood LIKE ? OR location LIKE ?)"
                            params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
                
                # First try with all filters
                all_filters_query = query
                all_filters_params = params.copy()
                
                # Add optional filters
                if filters.get('min_price'):
                    all_filters_query += " AND price_value >= ?"
                    all_filters_params.append(float(filters['min_price']))
                if filters.get('max_price'):
                    all_filters_query += " AND price_value <= ?"
                    all_filters_params.append(float(filters['max_price']))
                if filters.get('bedrooms'):
                    all_filters_query += " AND title LIKE ?"
                    all_filters_params.append(f"%{filters['bedrooms']}BHK%")
                if filters.get('bathrooms'):
                    all_filters_query += " AND baths = ?"
                    all_filters_params.append(int(filters['bathrooms']))
                if filters.get('has_balcony') is not None:
                    all_filters_query += " AND has_balcony = ?"
                    all_filters_params.append(1 if filters['has_balcony'] else 0)
                
                # Try with all filters first
                all_filters_query += f" LIMIT {limit}"
                print(f"Trying with all filters: {all_filters_query} with params: {all_filters_params}")
                cursor.execute(all_filters_query, all_filters_params)
                results = cursor.fetchall()
                
                # If no results, try with just location and price range
                if not results:
                    print("No results with all filters, trying with fewer filters...")
                    relaxed_query = query
                    relaxed_params = params.copy()
                    
                    # Add price range (if provided)
                    if filters.get('min_price'):
                        relaxed_query += " AND price_value >= ?"
                        relaxed_params.append(float(filters['min_price']))
                    if filters.get('max_price'):
                        relaxed_query += " AND price_value <= ?"
                        relaxed_params.append(float(filters['max_price']))
                    
                    relaxed_query += f" LIMIT {limit}"
                    print(f"Trying with relaxed filters: {relaxed_query} with params: {relaxed_params}")
                    cursor.execute(relaxed_query, relaxed_params)
                    results = cursor.fetchall()
                    
                    # If still no results, try with just location
                    if not results and filters.get('location'):
                        print("No results with price range, trying with just location...")
                        location_query = "SELECT id, title, location, city, neighborhood, price, price_value, total_area, baths, has_balcony FROM properties WHERE 1=1"
                        location_params = []
                        
                        location = filters['location'].strip()
                        if location:
                            location_query += " AND (city LIKE ? OR neighborhood LIKE ? OR location LIKE ?)"
                            location_params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
                        
                        location_query += f" LIMIT {limit}"
                        print(f"Trying with location only: {location_query} with params: {location_params}")
                        cursor.execute(location_query, location_params)
                        results = cursor.fetchall()
                
                # Format results
                formatted_results = []
                for row in results:
                    formatted_results.append({
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
                return formatted_results
            
            # If no filters provided, return top properties
            query += f" LIMIT {limit}"
            cursor.execute(query, params)
            
            # Fetch results
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
            logger.error(f"Error searching properties: {str(e)}")
            print(f"Error in search_properties: {str(e)}")
            return []
    
    @lru_cache(maxsize=128)
    def get_recommendations(self, property_id, num_recommendations=5):
        try:
            if not isinstance(property_id, int):
                raise ValueError("property_id must be an integer")
            if property_id < 0:
                raise ValueError("property_id must be non-negative")
            if num_recommendations < 1:
                raise ValueError("num_recommendations must be positive")
            if property_id >= len(self.df):
                return []
            
            # Get property embedding
            property_embedding = self.property_embeddings[property_id].reshape(1, -1)
            
            # Search FAISS index
            distances, indices = self.index.search(property_embedding, num_recommendations + 1)
            
            # Get the details of recommended properties (excluding the input property)
            recommendations = []
            for i, idx in enumerate(indices[0]):
                if idx != property_id:  # Skip the input property
                    property_info = self.get_property_by_id(idx)
                    if property_info:
                        property_info['similarity_score'] = float(1.0 / (1.0 + distances[0][i]))
                        recommendations.append(property_info)
            
            return recommendations[:num_recommendations]
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def get_property_by_id(self, property_id):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT id, title, location, city, neighborhood, price, price_value, total_area, baths, has_balcony
            FROM properties WHERE id = ?
            ''', (property_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
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
                }
            return None
        except Exception as e:
            logger.error(f"Error getting property by ID: {str(e)}")
            return None

    def get_filtered_recommendations(self, property_id, filters=None):
        # Get base property info
        base_property = self.get_property_by_id(property_id)
        if not base_property:
            return []
        
        # If filters not provided, create filters based on the base property
        if not filters:
            filters = {
                'location': base_property['city'],  # Use city for broader matches
                'max_price': base_property['price_value'] * 1.2  # 20% higher price
            }
        
        # Get recommendations using FAISS
        recommendations = self.get_recommendations(property_id)
        
        # Apply additional filters if needed
        if filters:
            filtered_recommendations = []
            for rec in recommendations:
                matches = True
                
                # Apply price filter
                if 'max_price' in filters and rec['price_value'] > filters['max_price']:
                    matches = False
                
                # Apply location filter (city match)
                if 'location' in filters and filters['location'] not in rec['location']:
                    # Check if at least city matches
                    if filters['location'] != rec['city']:
                        matches = False
                
                if matches:
                    filtered_recommendations.append(rec)
            
            return filtered_recommendations
        
        return recommendations

    def search_properties_with_fallback(self, filters=None, limit=10):
        """Search properties with automatic fallback to more relaxed filters if no results found"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            relaxed_search = False
            
            # Start with base query
            query = "SELECT id, title, location, city, neighborhood, price, price_value, total_area, baths, has_balcony FROM properties WHERE 1=1"
            params = []
            
            # Add filters if provided
            if filters:
                # Location filter (highest priority)
                if filters.get('location'):
                    location = filters['location'].strip()
                    if location:
                        # Process the location with spaCy to extract city
                        doc = self.nlp(location)
                        cities = [ent.text for ent in doc.ents if getattr(ent, 'label_', '') == "GPE"]
                        
                        # If spaCy found a city, prioritize city matching
                        if cities:
                            city = cities[-1]  # Last entity is usually the city
                            print(f"Extracted city from input: {city}")
                            query += " AND (city LIKE ? OR location LIKE ?)"
                            params.extend([f"%{city}%", f"%{location}%"])
                        else:
                            # Try to match with city, neighborhood, or location
                            query += " AND (city LIKE ? OR neighborhood LIKE ? OR location LIKE ?)"
                            params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
                
                # First try with all filters
                all_filters_query = query
                all_filters_params = params.copy()
                
                # Add optional filters
                if filters.get('min_price'):
                    all_filters_query += " AND price_value >= ?"
                    all_filters_params.append(float(filters['min_price']))
                if filters.get('max_price'):
                    all_filters_query += " AND price_value <= ?"
                    all_filters_params.append(float(filters['max_price']))
                if filters.get('bedrooms'):
                    all_filters_query += " AND title LIKE ?"
                    all_filters_params.append(f"%{filters['bedrooms']}BHK%")
                if filters.get('bathrooms'):
                    all_filters_query += " AND baths = ?"
                    all_filters_params.append(int(filters['bathrooms']))
                if filters.get('has_balcony') is not None:
                    all_filters_query += " AND has_balcony = ?"
                    all_filters_params.append(1 if filters['has_balcony'] else 0)
                
                # Try with all filters first
                all_filters_query += f" LIMIT {limit}"
                print(f"Trying with all filters: {all_filters_query} with params: {all_filters_params}")
                cursor.execute(all_filters_query, all_filters_params)
                results = cursor.fetchall()
                
                # If no results, try with just location and price range
                if not results:
                    relaxed_search = True
                    print("No results with all filters, trying with fewer filters...")
                    relaxed_query = query
                    relaxed_params = params.copy()
                    
                    # Add price range (if provided)
                    if filters.get('min_price'):
                        relaxed_query += " AND price_value >= ?"
                        relaxed_params.append(float(filters['min_price']))
                    if filters.get('max_price'):
                        relaxed_query += " AND price_value <= ?"
                        relaxed_params.append(float(filters['max_price']))
                    
                    relaxed_query += f" LIMIT {limit}"
                    print(f"Trying with relaxed filters: {relaxed_query} with params: {relaxed_params}")
                    cursor.execute(relaxed_query, relaxed_params)
                    results = cursor.fetchall()
                    
                    # If still no results, try with just location
                    if not results and filters.get('location'):
                        print("No results with price range, trying with just location...")
                        location_query = "SELECT id, title, location, city, neighborhood, price, price_value, total_area, baths, has_balcony FROM properties WHERE 1=1"
                        location_params = []
                        
                        location = filters['location'].strip()
                        if location:
                            location_query += " AND (city LIKE ? OR neighborhood LIKE ? OR location LIKE ?)"
                            location_params.extend([f"%{location}%", f"%{location}%", f"%{location}%"])
                        
                        location_query += f" LIMIT {limit}"
                        print(f"Trying with location only: {location_query} with params: {location_params}")
                        cursor.execute(location_query, location_params)
                        results = cursor.fetchall()
                
                # Format results
                formatted_results = []
                for row in results:
                    formatted_results.append({
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
                return formatted_results, relaxed_search
            
            # If no filters provided, return top properties
            query += f" LIMIT {limit}"
            cursor.execute(query, params)
            
            # Fetch results
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
            return results, False
        except Exception as e:
            logger.error(f"Error searching properties: {str(e)}")
            print(f"Error in search_properties_with_fallback: {str(e)}")
            return [], False

# Test the recommendation system
if __name__ == "__main__":
    # Initialize the recommender
    recommender = PropertyRecommender('data/Real Estate Data V21.csv')
    
    # Test with a sample property (property_id = 0)
    print("Getting recommendations for property 0:")
    recommendations = recommender.get_recommendations(0)
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Title: {rec['title']}")
        print(f"Location: {rec['location']}")
        print(f"City: {rec['city']}")
        print(f"Neighborhood: {rec['neighborhood']}")
        print(f"Price: {rec['price']}")
        print(f"Total Area: {rec['total_area']}")
        print(f"Similarity Score: {rec['similarity_score']:.2f}")
    
    # Test search with filters
    print("\nSearching properties in Chennai:")
    search_results = recommender.search_properties({'location': 'Chennai'})
    for i, result in enumerate(search_results, 1):
        print(f"\nResult {i}:")
        print(f"Title: {result['title']}")
        print(f"Location: {result['location']}")