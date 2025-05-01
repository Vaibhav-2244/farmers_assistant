from flask import Flask, render_template, request, jsonify
import json
import re
import os
import requests
from werkzeug.utils import secure_filename
from disease_detection import get_detector
from transformers import pipeline
from config import WEATHER_API_KEY, WEATHER_API_URL, AGMARKET_API_KEY, AGMARKET_API_URL

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize NLP pipeline
nlp = pipeline("question-answering")

# Load translations
with open('data/translations.json', 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load preprocessed crop data
with open('data/processed_crop_data.json', 'r') as f:
    crop_info = json.load(f)

def get_weather_data(city):
    try:
        params = {
            'q': city,
            'appid': WEATHER_API_KEY,
            'units': 'metric'
        }
        response = requests.get(WEATHER_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            return {
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'],
                'description': data['weather'][0]['description']
            }
        return None
    except Exception as e:
        print(f"Error getting weather data: {e}")
        return None

def get_crop_prices(crop_name):
    try:
        params = {
            'api-key': AGMARKET_API_KEY,
            'format': 'json',
            'filters[Commodity]': crop_name,  # Changed to 'Commodity' (capitalization matters)
            'limit': 5,
            'sort[Arrival_Date]': 'desc'  # Get latest prices first
        }
        
        response = requests.get(AGMARKET_API_URL, params=params)
        print(f"API Response: {response.text[:200]}...")
        data = response.json()
        
        if response.status_code == 200 and data.get('records'):
            # Return all records (removing year filter)
            return data['records']
            
        return []
    
    except Exception as e:
        print(f"Error fetching crop prices: {e}")
        return []

def get_crop_info(crop_name, parameter=None):
    crop_name = crop_name.lower().strip()
    
    # Find the closest matching crop name
    matching_crops = [crop for crop in crop_info.keys() if crop_name in crop]
    
    if not matching_crops:
        return None
    
    crop = matching_crops[0]
    info = crop_info[crop]
    
    if parameter:
        parameter = parameter.lower()
        if parameter in info:
            return {
                'crop': crop,
                'parameter': parameter,
                'mean': info[parameter]['mean'],
                'std': info[parameter]['std']
            }
    return {
        'crop': crop,
        'info': info
    }

def generate_answer(question, language='en-US'):
    # question = question.lower()
    
    # Check for weather-related questions
    if any(word in question for word in ['weather', 'temperature', 'humidity', 'wind']):
        # Extract city name using NLP
        city = extract_city(question)
        if city:
            weather_data = get_weather_data(city)
            if weather_data:
                if language == 'hi-IN':
                    return f"{city} का मौसम: तापमान {weather_data['temperature']}°C, आर्द्रता {weather_data['humidity']}%, हवा की गति {weather_data['wind_speed']} m/s, {weather_data['description']}"
                return f"Weather in {city}: Temperature {weather_data['temperature']}°C, Humidity {weather_data['humidity']}%, Wind Speed {weather_data['wind_speed']} m/s, {weather_data['description']}"
            return translations['hi']['Error getting weather data'] if language == 'hi-IN' else "Error getting weather data"
    
    # Check for crop price questions
    if any(word in question for word in ['price', 'market', 'cost', 'rate']):
        crop_name = extract_crop_name(question)
        if crop_name:
            prices = get_crop_prices(crop_name)
            if prices:
                # Use Modal_Price (most common trading price)
                price = prices[0].get('Modal_Price', prices[0].get('Max_Price', 'N/A'))
                
                # Default unit (AgMarkNet usually uses Rs/quintal)
                unit = "quintal"  
                
                if language == 'hi-IN':
                    return f"{crop_name} का वर्तमान बाजार मूल्य: {price} रुपये प्रति {unit}"
                return f"Current market price for {crop_name}: {price} rupees per {unit}"
            return translations['hi']['No price information available'] if language == 'hi-IN' else "No price information available"
    
    # Define parameter mappings
    parameters = {
        'nitrogen': 'N',
        'phosphorus': 'P',
        'potassium': 'K',
        'temperature': 'temperature',
        'humidity': 'humidity',
        'ph': 'ph',
        'rainfall': 'rainfall',
        'बारिश': 'rainfall',
        'वर्षा': 'rainfall',
        'तापमान': 'temperature',
        'आर्द्रता': 'humidity',
        'नाइट्रोजन': 'N',
        'फॉस्फोरस': 'P',
        'पोटैशियम': 'K'
    }
    
    # Extract crop name and parameter from question
    for crop in crop_info.keys():
        if crop in question:
            # Check for parameter-specific questions
            for param_name, param_key in parameters.items():
                if param_name in question:
                    info = get_crop_info(crop, param_key)
                    if info:
                        if language == 'hi-IN':
                            return f"{crop.title()} के लिए {param_name} की आवश्यकता औसतन {info['mean']} होती है, जिसमें {info['std']} का मानक विचलन होता है।"
                        return f"For {crop.title()}, the required {param_name} is typically {info['mean']} with a standard deviation of {info['std']}."
            
            # If no specific parameter is mentioned, return general information
            info = get_crop_info(crop)
            if info:
                if language == 'hi-IN':
                    return f"{crop.title()} के लिए आदर्श परिस्थितियाँ: तापमान {info['info']['temperature']['mean']}°C, आर्द्रता {info['info']['humidity']['mean']}%, वर्षा {info['info']['rainfall']['mean']}mm, और मिट्टी का pH {info['info']['ph']['mean']} होना चाहिए।"
                return f"For {crop.title()}, the ideal conditions are: temperature {info['info']['temperature']['mean']}°C, humidity {info['info']['humidity']['mean']}%, rainfall {info['info']['rainfall']['mean']}mm, and soil pH of {info['info']['ph']['mean']}."

    # If no specific crop is mentioned, return general information about all crops
    if language == 'hi-IN':
        return "कृपया किसी विशेष फसल के बारे में पूछें, जैसे 'गेहूं के लिए आवश्यक तापमान क्या है?' या 'चावल के लिए कितनी वर्षा की आवश्यकता होती है?'"
    return "Please ask about a specific crop, such as 'What temperature is required for wheat?' or 'How much rainfall does rice need?'"

def extract_city(question):
    # Simple city extraction - can be improved with NLP
    cities = ['delhi', 'mumbai', 'kolkata', 'chennai', 'bangalore', 'hyderabad']
    for city in cities:
        if city in question:
            return city
    return None

def extract_crop_name(question):
    question = question.lower().strip()
    crops = list(crop_info.keys())
    # Extract crop name from question using NLP
    for crop in crops:
        if crop in question:
            return crop.title()
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request format'}), 400
            
        question = data.get('query', '').strip()
        language = data.get('language', 'en-US')
        
        if not question:
            return jsonify({'error': 'Please enter a question.'}), 400
        
        answer = generate_answer(question, language)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"FULL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = translations['hi'].get('query_error', 'An error occurred while processing your question. Please try again.')
        return jsonify({'error': error_msg}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        language = request.form.get('language', 'en-US')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # file.save(filepath)
            # Save as binary to avoid encoding issues
            with open(filepath, 'wb') as f:
                f.write(file.read())

            try:
                # Get disease prediction
                detector = get_detector()
                result = detector.predict_disease(filepath)

                                # Ensure strings are UTF-8 encoded
                def ensure_utf8(s):
                    if isinstance(s, bytes):
                        return s.decode('utf-8', errors='replace')
                    return str(s)

                # Translate disease information if needed
                if language == 'hi-IN':
                    result['disease_name'] = translations['hi'].get(result['disease_name'], result['disease_name'])
                    result['description'] = translations['hi'].get(result['description'], result['description'])
                    result['prevention'] = [translations['hi'].get(step, step) for step in result['prevention']]
                    result['treatment'] = [translations['hi'].get(step, step) for step in result['treatment']]

                # Clean up the temporary file
                os.remove(filepath)

                return jsonify(result)
            except Exception as e:
                # Clean up the temporary file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise e

    except Exception as e:
        print(f"Error analyzing image: {e}")
        error_msg = translations['hi']['Failed to analyze the image'] if language == 'hi-IN' else "Failed to analyze the image"
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True) 