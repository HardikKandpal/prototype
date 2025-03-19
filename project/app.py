from flask import Flask, request, jsonify
import openai
import os
from flask_cors import CORS
from io import BytesIO
import requests
from rec import PropertyRecommender  # Import our recommendation system
from property_valuation_model import PropertyValuationModel  # Import our property valuation model
import logging
import pandas as pd
import traceback
import sqlite3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)
CORS(app)  # Enable CORS to allow frontend requests

DB_PATH = "data/property_db.sqlite"
os.makedirs("data", exist_ok=True)

def init_db():
    """Create database tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS enquiries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            phone TEXT,
            message TEXT,
            property_id INTEGER
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/api/enquire", methods=["POST"])
def make_enquiry():
    """Store user inquiries in the database."""
    data = request.json
    name, email, phone, message, property_id = (
        data.get("name"),
        data.get("email"),
        data.get("phone"),
        data.get("message"),
        data.get("property_id"),
    )

    if not name or not email or not message or not property_id:
        return jsonify({"error": "Missing required fields"}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO enquiries (name, email, phone, message, property_id) VALUES (?, ?, ?, ?, ?)",
        (name, email, phone, message, property_id),
    )
    conn.commit()
    conn.close()

    return jsonify({"message": "Enquiry submitted successfully"}), 201

@app.route("/api/admin/enquiries", methods=["GET"])
def get_enquiries():
    """Fetch all user enquiries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, message, property_id FROM enquiries")
    enquiries = cursor.fetchall()
    conn.close()

    return jsonify(
        [
            {
                "id": enquiry[0],
                "name": enquiry[1],
                "email": enquiry[2],
                "phone": enquiry[3],
                "message": enquiry[4],
                "property_id": enquiry[5],
            }
            for enquiry in enquiries
        ]
    )

@app.route("/api/admin/delete-enquiry/<int:enquiry_id>", methods=["DELETE"])
def delete_enquiry(enquiry_id):
    """Delete an enquiry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM enquiries WHERE id = ?", (enquiry_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Enquiry deleted successfully"}), 200

@app.route("/api/featured-properties", methods=["GET"])
def get_featured_properties():
    """Fetch featured listings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, location, price, total_area FROM properties ORDER BY RANDOM() LIMIT 5")
    properties = cursor.fetchall()
    conn.close()

    return jsonify(
        [
            {
                "id": prop[0],
                "title": prop[1],
                "location": prop[2],
                "price": prop[3],
                "total_area": prop[4],
            }
            for prop in properties
        ]
    )


@app.route("/api/admin/enquiries", methods=["GET"])
def get_enquiries():
    """Fetch all user enquiries."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone, message, property_id FROM enquiries")
    enquiries = cursor.fetchall()
    conn.close()

    return jsonify(
        [
            {
                "id": enquiry[0],
                "name": enquiry[1],
                "email": enquiry[2],
                "phone": enquiry[3],
                "message": enquiry[4],
                "property_id": enquiry[5],
            }
            for enquiry in enquiries
        ]
    )


@app.route("/api/admin/delete-enquiry/<int:enquiry_id>", methods=["DELETE"])
def delete_enquiry(enquiry_id):
    """Delete an enquiry."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM enquiries WHERE id = ?", (enquiry_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Enquiry deleted successfully"}), 200


@app.route("/api/admin/properties", methods=["GET"])
def get_all_properties():
    """Fetch all properties."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, location, price, total_area FROM properties")
    properties = cursor.fetchall()
    conn.close()

    return jsonify(
        [
            {
                "id": prop[0],
                "title": prop[1],
                "location": prop[2],
                "price": prop[3],
                "total_area": prop[4],
            }
            for prop in properties
        ]
    )


@app.route("/api/admin/add-property", methods=["POST"])
def add_property():
    """Add a new property."""
    data = request.json
    title, location, price, total_area = (
        data.get("title"),
        data.get("location"),
        data.get("price"),
        data.get("total_area"),
    )

    if not title or not location or not price or not total_area:
        return jsonify({"error": "Missing required fields"}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO properties (title, location, price, total_area) VALUES (?, ?, ?, ?)",
        (title, location, price, total_area),
    )
    conn.commit()
    conn.close()

    return jsonify({"message": "Property added successfully"}), 201


# Load OpenAI API Key
openai.api_key=os.getenv("OPENAI_API_KEY") 

# Initialize the recommender system
csv_path = 'data/Real Estate Data V21.csv'
recommender = PropertyRecommender(csv_path)

# Initialize the property valuation model
valuation_model = PropertyValuationModel()
try:
    valuation_model.load_model()
    logger.info("Property valuation model loaded successfully")
except Exception as e:
    logger.error(f"Error loading property valuation model: {str(e)}")
    logger.info("Will attempt to train a new model if needed")

# 1️⃣ Optimized AI-Powered Property Description
@app.route('/api/generate-description', methods=['POST'])
def generate_description():
    data = request.get_json()
    details = data.get('details', '')

    if not details:
        return jsonify({'error': 'Property details missing!'}), 400

    try:
        # Updated OpenAI API call for v1.0.0+
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert real estate agent, skilled at writing engaging property listings that attract buyers."},
                {"role": "user", "content": f"Write a professional and engaging real estate listing for the following property: {details}. \
                    Include the best selling points, neighborhood highlights, and a compelling call to action."}
            ],
            max_tokens=120
        )

        return jsonify({'generated_text': response.choices[0].message.content})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 2️⃣ ML-Powered Property Valuation (Replacing OpenAI implementation)
@app.route('/api/estimate-value', methods=['POST'])
def estimate_value():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Property details missing!'}), 400
    
    try:
        # Check if we have the model initialized
        global valuation_model
        if valuation_model.model is None:
            logger.info("Property valuation model not loaded. Initializing and training...")
            valuation_model = PropertyValuationModel()
            
            # Try to load the model first
            if not valuation_model.load_model():
                # If loading fails, train a new model
                logger.info("Training new property valuation model...")
                df = pd.read_csv(csv_path)
                
                # Prepare data for training (similar to train_valuation_model.py)
                # Extract city and neighborhood from location
                df['city'] = df['location'].apply(lambda x: x.split(',')[-1].strip() if isinstance(x, str) and ',' in x else x)
                df['neighborhood'] = df['location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else 'Unknown')
                
                # Extract numerical price from price string
                df['price_value'] = df['price'].apply(lambda x: float(''.join(c for c in str(x) if c.isdigit() or c == '.')) if isinstance(x, str) else x)
                
                # Convert bedrooms from title
                import re
                df['bedrooms'] = df['title'].apply(lambda x: int(re.search(r'(\d+)\s*BHK', str(x), re.IGNORECASE).group(1)) if isinstance(x, str) and re.search(r'(\d+)\s*BHK', str(x), re.IGNORECASE) else 2)
                
                # Clean data
                df = df.dropna(subset=['price_value', 'total_area'])
                df = df[(df['price_value'] > 100000) & (df['price_value'] < 100000000)]
                df = df[(df['total_area'] > 100) & (df['total_area'] < 10000)]
                
                # Train the model
                valuation_model.train(df, target_column='price_value')
        
        # Prepare the property data for prediction
        property_data = {
            'city': data.get('location', '').split(',')[-1].strip() if ',' in data.get('location', '') else data.get('location', ''),
            'neighborhood': data.get('location', '').split(',')[0].strip() if ',' in data.get('location', '') else 'Unknown',
            'bedrooms': int(data.get('bedrooms', 2)),
            'baths': int(data.get('bathrooms', 1)),
            'total_area': float(data.get('size', 1000)),
            'has_balcony': data.get('has_balcony', False)
        }
        
        # Make prediction
        prediction = valuation_model.predict(property_data)
        
        # Get comparable properties if available
        comparables = []
        if hasattr(recommender, 'df'):
            try:
                # Get comparable properties
                comparable_properties = valuation_model.get_comparable_properties(property_data)
                
                # Format for response
                comparables = comparable_properties
            except Exception as e:
                logger.error(f"Error getting comparable properties: {str(e)}")
        
        # Format the response
        response = {
            'estimated_value': f"₹{prediction['estimated_value']:,.2f}",
            'min_value': f"₹{prediction['min_value']:,.2f}",
            'max_value': f"₹{prediction['max_value']:,.2f}",
            'confidence_percentage': prediction['confidence_percentage'],
            'top_features': prediction['top_features'],
            'comparable_properties': comparables
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in property valuation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# Optimized AI-Powered Natural Language Search
@app.route('/api/natural-language-search', methods=['POST'])
def natural_language_search():
    data = request.get_json()
    query = data.get('query', '').lower()

    if not query:
        return jsonify({'error': 'Search query missing!'}), 400

    try:
        # Get all properties from our recommender
        all_properties = []
        for idx in range(len(recommender.df)):
            property_info = recommender.get_property_by_id(idx)
            if property_info:
                all_properties.append(property_info)

        # Filter properties based on the search query
        search_results = []
        for prop in all_properties:
            # Check if query matches any property attributes
            if (query in prop['title'].lower() or 
                query in prop['location'].lower() or 
                query in str(prop['price']).lower()):
                search_results.append({
                    'id': prop['id'],
                    'title': prop['title'],
                    'location': prop['location'],
                    'price': prop['price']
                })

        return jsonify({
            'success': True,
            'results': search_results[:10]  # Limit to top 10 results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400
    

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        # Save the uploaded image
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)

        # Updated OpenAI API call for v1.0.0+
        with open(image_path, "rb") as img_file:
            file_content = img_file.read()
            
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You analyze real estate images and describe their key features."},
                    {"role": "user", "content": "Describe the key features of this real estate image."}
                ],
                max_tokens=200
            )

        # Extract AI analysis response
        analysis_text = response.choices[0].message.content

        return jsonify({'analysis': analysis_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/")
def home():
    return "Flask Backend is Running!"

# Add new endpoint for property recommendations
@app.route('/api/recommendations/<int:property_id>', methods=['GET'])
def get_recommendations(property_id):
    try:
        num_recommendations = request.args.get('limit', default=5, type=int)
        recommendations = recommender.get_recommendations(property_id, num_recommendations)
        return jsonify(recommendations)
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add endpoint to get property details
@app.route('/api/property/<int:property_id>', methods=['GET'])
def get_property(property_id):
    try:
        property_info = recommender.get_property_by_id(property_id)
        if property_info:
            return jsonify(property_info)
        else:
            return jsonify({"error": "Property not found"}), 404
    except Exception as e:
        logger.error(f"Error getting property: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add new endpoint for property search
@app.route('/api/property-search', methods=['POST'])
def search_properties():
    try:
        filters = request.json
        logger.info(f"Searching properties with filters: {filters}")
        
        results, relaxed_search = recommender.search_properties_with_fallback(filters)
        
        return jsonify({
            "results": results,
            "relaxed_search": relaxed_search
        })
    except Exception as e:
        logger.error(f"Error searching properties: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/property-valuation', methods=['POST'])
def property_valuation():
    try:
        property_data = request.json
        logger.info(f"Valuing property with data: {property_data}")
        
        # Check if model is loaded, if not, try to load it
        if valuation_model.model is None:
            try:
                valuation_model.load_model()
            except Exception as e:
                logger.warning(f"Could not load model, will train a new one: {str(e)}")
                # If model doesn't exist, train a new one
                if os.path.exists(csv_path):
                    from train_valuation_model import main as train_model
                    train_model()
                    valuation_model.load_model()
                else:
                    return jsonify({"error": "Property valuation model not available and training data not found"}), 500
        
        # Make prediction
        prediction = valuation_model.predict(property_data)
        
        # Get comparable properties
        comparables = valuation_model.get_comparable_properties(property_data)
        
        # Format response
        response = {
            "valuation": prediction,
            "comparable_properties": comparables
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in property valuation: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})


@app.route('/api/property-search', methods=['POST'])
def search_properties():
    try:
        filters = request.json
        logger.info(f"Searching properties with filters: {filters}")
        
        # Extract filtering options
        location = filters.get('location', '')
        min_price = filters.get('min_price', 0)
        max_price = filters.get('max_price', 999999999)
        bedrooms = filters.get('bedrooms', None)
        bathrooms = filters.get('bathrooms', None)
        total_area = filters.get('total_area', None)
        has_balcony = filters.get('has_balcony', None)
        sort_by = filters.get('sort_by', 'price')  # Default: Sort by price
        sort_order = filters.get('sort_order', 'asc')  # 'asc' (low-high) or 'desc' (high-low)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        query = "SELECT id, title, location, price, total_area, baths, bedrooms, has_balcony FROM properties WHERE 1=1"
        params = []

        if location:
            query += " AND location LIKE ?"
            params.append(f"%{location}%")

        if min_price:
            query += " AND price >= ?"
            params.append(min_price)

        if max_price:
            query += " AND price <= ?"
            params.append(max_price)

        if bedrooms:
            query += " AND bedrooms = ?"
            params.append(bedrooms)

        if bathrooms:
            query += " AND baths = ?"
            params.append(bathrooms)

        if total_area:
            query += " AND total_area >= ?"
            params.append(total_area)

        if has_balcony is not None:
            query += " AND has_balcony = ?"
            params.append(int(has_balcony))  # Convert boolean to integer

        # Apply Sorting
        query += f" ORDER BY {sort_by} {sort_order.upper()}"

        cursor.execute(query, params)
        properties = cursor.fetchall()
        conn.close()

        return jsonify(
            [
                {
                    "id": prop[0],
                    "title": prop[1],
                    "location": prop[2],
                    "price": prop[3],
                    "total_area": prop[4],
                    "baths": prop[5],
                    "bedrooms": prop[6],
                    "has_balcony": bool(prop[7]),
                }
                for prop in properties
            ]
        )
    except Exception as e:
        logger.error(f"Error searching properties: {str(e)}")
        return jsonify({"error": str(e)}), 500


# Run Flask Server
if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at {csv_path}")
        exit(1)
    
    app.run(debug=True)
