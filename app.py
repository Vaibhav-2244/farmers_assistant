from flask import Flask, render_template, request, jsonify
import json
import re
import os
import requests
from werkzeug.utils import secure_filename
from disease_detection import get_detector
from transformers import pipeline
from config import WEATHER_API_KEY, WEATHER_API_URL, AGMARKET_API_KEY, AGMARKET_API_URL
# Import the improved TranslationService
from translation_service import TranslationService

app = Flask(__name__)
translator = TranslationService()
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize NLP pipeline
nlp = pipeline("question-answering")

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load your existing crop data structure
try:
    with open('data/processed_crop_data.json', 'r') as f:
        crop_info = json.load(f)
except Exception as e:
    print(f"Error loading crop data: {e}")
    crop_info = {}

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
                'description': data['weather'][0]['description'],
                'city': data['name']
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
            'filters[Commodity]': crop_name,
            'limit': 5,
            'sort[Arrival_Date]': 'desc'
        }
        
        response = requests.get(AGMARKET_API_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            if data.get('records'):
                # Process all records
                processed_records = []
                for record in data['records']:
                    processed_records.append({
                        'crop': record.get('Commodity', crop_name),
                        'market': record.get('Market', 'Unknown'),
                        'price': record.get('Modal_Price', record.get('Max_Price', 'N/A')),
                        'unit': record.get('Unit', 'Quintal'),
                        'date': record.get('Arrival_Date', 'Unknown')
                    })
                return processed_records
        return []
    except Exception as e:
        print(f"Error fetching crop prices: {e}")
        return []

def get_crop_info(crop_name, parameter=None):
    if not crop_name:
        return None
        
    crop_name = crop_name.lower().strip()
    
    # First try exact match
    if crop_name in crop_info:
        info = crop_info[crop_name]
        if parameter:
            parameter = parameter.lower()
            if parameter in info:
                return {
                    'crop': crop_name,
                    'parameter': parameter,
                    'mean': info[parameter]['mean'],
                    'std': info[parameter]['std']
                }
            return None
        return {
            'crop': crop_name,
            'info': info
        }
    
    # Try partial match
    matching_crops = [crop for crop in crop_info.keys() if crop_name in crop.lower()]
    if matching_crops:
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
            return None
        return {
            'crop': crop,
            'info': info
        }
    
    return None

def format_response(data, lang_code):
    """Format response based on language"""
    if not data:
        return translator.translate("No information available", lang_code)
    
    # Format based on data type and language
    if isinstance(data, dict):
        if 'temperature' in data and 'humidity' in data:  # Weather data
            if lang_code == 'hi':
                return (
                    f"{data['city']} का मौसम:\n"
                    f"तापमान: {data['temperature']}°C\n"
                    f"आर्द्रता: {data['humidity']}%\n"
                    f"हवा की गति: {data['wind_speed']} m/s\n"
                    f"स्थिति: {translator.translate(data['description'], 'hi')}"
                )
            return (
                f"Weather in {data['city']}:\n"
                f"Temperature: {data['temperature']}°C\n"
                f"Humidity: {data['humidity']}%\n"
                f"Wind Speed: {data['wind_speed']} m/s\n"
                f"Conditions: {data['description']}"
            )
    
    # For lists (like price data)
    elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if 'price' in data[0]:  # Price data
            price = data[0]
            crop_name = price['crop']
            if lang_code == 'hi':
                hindi_name = translator.translate_crop_name(crop_name, 'hi')
                return (
                    f"{hindi_name} की कीमत:\n"
                    f"मूल्य: {price['price']} रुपये प्रति {translator.translate(price['unit'].lower(), 'hi')}\n"
                    f"बाजार: {price['market']}\n"
                    f"तिथि: {price['date']}"
                )
            return (
                f"{crop_name.capitalize()} Price:\n"
                f"Price: ₹{price['price']} per {price['unit'].lower()}\n"
                f"Market: {price['market']}\n"
                f"Date: {price['date']}"
            )
    
    # Default case - just return the data
    return str(data)

def generate_answer(question, language='en'):
    try:
        # Normalize language code to just the first 2 chars
        lang_code = language[:2].lower() if language != 'auto' else 'en'
        
        # Weather queries
        weather_keywords = ['weather', 'temperature', 'humidity', 'wind', 'मौसम', 'तापमान', 'आर्द्रता', 'हवा']
        if any(word in question.lower() for word in weather_keywords):
            city = extract_city(question)
            if not city:
                return translator.translate("Please specify a city (e.g.: weather in Delhi)", lang_code)
            
            weather_data = get_weather_data(city)
            if weather_data:
                return format_response(weather_data, lang_code)
            return translator.translate("Could not fetch weather data", lang_code)
        
        # Crop price queries
        price_keywords = ['price', 'market', 'cost', 'rate', 'मूल्य', 'कीमत', 'बाजार', 'लागत', 'दर']
        if any(word in question.lower() for word in price_keywords):
            crop_name = extract_crop_name(question)
            if crop_name:
                prices = get_crop_prices(crop_name)
                if prices:
                    return format_response(prices, lang_code)
                
                # No prices found
                if lang_code == 'hi':
                    hindi_name = translator.translate_crop_name(crop_name, 'hi')
                    return f"{hindi_name} के लिए मूल्य जानकारी उपलब्ध नहीं है"
                return f"No price information available for {crop_name.capitalize()}"
        
        # Parameter queries
        param_mappings = {
            'nitrogen': 'N', 'phosphorus': 'P', 'potassium': 'K',
            'temperature': 'temperature', 'humidity': 'humidity',
            'ph': 'ph', 'rainfall': 'rainfall',
            'नाइट्रोजन': 'N', 'फॉस्फोरस': 'P', 'पोटैशियम': 'K',
            'तापमान': 'temperature', 'आर्द्रता': 'humidity',
            'पीएच': 'ph', 'वर्षा': 'rainfall', 'बारिश': 'rainfall'
        }
        
        # Check for parameter queries
        for param_term, param_key in param_mappings.items():
            if param_term.lower() in question.lower():
                crop_name = extract_crop_name(question)
                if crop_name:
                    # Get parameter info
                    info = get_crop_info(crop_name, param_key)
                    
                    if info:
                        if lang_code == 'hi':
                            hindi_crop = translator.translate_crop_name(info['crop'], 'hi')
                            hindi_param = translator.translate(param_term, 'hi')
                            return (
                                f"{hindi_crop} के लिए {hindi_param} आवश्यकता:\n"
                                f"औसत: {info['mean']}\n"
                                f"मानक विचलन: {info['std']}"
                            )
                        return (
                            f"{info['crop'].capitalize()} {param_term} requirements:\n"
                            f"Mean: {info['mean']}\n"
                            f"Std Dev: {info['std']}"
                        )
                    else:
                        if lang_code == 'hi':
                            hindi_crop = translator.translate_crop_name(crop_name, 'hi')
                            hindi_param = translator.translate(param_term, 'hi')
                            return f"{hindi_crop} के लिए {hindi_param} जानकारी उपलब्ध नहीं है"
                        return f"No {param_term} information available for {crop_name.capitalize()}"
        
        # General crop information
        crop_name = extract_crop_name(question)
        if crop_name:
            info = get_crop_info(crop_name)
            if info:
                if lang_code == 'hi':
                    hindi_crop = translator.translate_crop_name(info['crop'], 'hi')
                    return (
                        f"{hindi_crop} के लिए आदर्श स्थितियाँ:\n"
                        f"तापमान: {info['info']['temperature']['mean']}°C (±{info['info']['temperature']['std']})\n"
                        f"आर्द्रता: {info['info']['humidity']['mean']}% (±{info['info']['humidity']['std']})\n"
                        f"वर्षा: {info['info']['rainfall']['mean']}mm (±{info['info']['rainfall']['std']})\n"
                        f"मिट्टी का pH: {info['info']['ph']['mean']} (±{info['info']['ph']['std']})"
                    )
                return (
                    f"Ideal conditions for {info['crop'].capitalize()}:\n"
                    f"Temperature: {info['info']['temperature']['mean']}°C (±{info['info']['temperature']['std']})\n"
                    f"Humidity: {info['info']['humidity']['mean']}% (±{info['info']['humidity']['std']})\n"
                    f"Rainfall: {info['info']['rainfall']['mean']}mm (±{info['info']['rainfall']['std']})\n"
                    f"Soil pH: {info['info']['ph']['mean']} (±{info['info']['ph']['std']})"
                )
            else:
                if lang_code == 'hi':
                    hindi_crop = translator.translate_crop_name(crop_name, 'hi')
                    return f"{hindi_crop} के बारे में जानकारी उपलब्ध नहीं है"
                return f"No information available for {crop_name.capitalize()}"
        
        # Final fallback
        return translator.translate("Please ask about a specific crop, weather, or market price", lang_code)
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return translator.translate("Error processing question. Please try again.", lang_code)

def extract_city(question):
    cities = {
        'delhi': ['delhi', 'दिल्ली', 'dilli'],
        'mumbai': ['mumbai', 'मुंबई', 'bombay'],
        'kolkata': ['kolkata', 'कोलकाता', 'calcutta'],
        'chennai': ['chennai', 'चेन्नई', 'madras'],
        'bangalore': ['bangalore', 'बेंगलुरु', 'bengaluru'],
        'hyderabad': ['hyderabad', 'हैदराबाद'],
        'pune': ['pune', 'पुणे'],
        'jaipur': ['jaipur', 'जयपुर']
    }
    
    question_lower = question.lower()
    for city, names in cities.items():
        if any(name in question_lower for name in names):
            return city
    return None

def extract_crop_name(question):
    question = question.lower().strip()
    
    # First check for exact matches in Hindi
    hindi_to_english = {
        'गेहूं': 'wheat', 'गेहूँ': 'wheat',
        'चावल': 'rice', 'धान': 'rice',
        'आलू': 'potato',
        'टमाटर': 'tomato',
        'मक्का': 'maize', 'मकई': 'maize',
        'बाजरा': 'millet',
        'सेब': 'apple',
        'केला': 'banana',
        'उड़द': 'blackgram',
        'चना': 'chickpea',
        'नारियल': 'coconut',
        'कॉफी': 'coffee',
        'कपास': 'cotton',
        'अंगूर': 'grapes',
        'दाल': 'lentil',
        'संतरा': 'orange',
        'पपीता': 'papaya',
        'मटर': 'peas',
        'अनार': 'pomegranate',
        'तरबूज': 'watermelon'
    }
    
    for hindi, english in hindi_to_english.items():
        if hindi in question:
            return english
    
    # Then check English names
    crops_in_data = list(crop_info.keys())
    for crop in crops_in_data:
        if crop.lower() in question:
            return crop.lower()
    
    # Try partial matches
    for crop in crops_in_data:
        if any(word in question for word in crop.lower().split()):
            return crop.lower()
    
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
        language = data.get('language', 'auto')
        
        if not question:
            return jsonify({'error': translator.translate('Please enter a question', 'en')}), 400
        
        # Auto-detect language if needed
        if language == 'auto':
            language = translator.detect_language(question)
        
        answer = generate_answer(question, language)
        return jsonify({
            'answer': answer,
            'detected_language': language
        })
    except Exception as e:
        print(f"Error processing query: {e}")
        error_msg = translator.translate(
            "An error occurred. Please try again.",
            language[:2] if language != 'auto' else 'en'
        )
        return jsonify({'error': error_msg}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        file = request.files['image']
        language = request.form.get('language', 'en-US')
        lang_code = language[:2].lower()
        
        if file.filename == '':
            return jsonify({'error': translator.translate('No selected file', lang_code)}), 400

        if file:
            # Save the uploaded file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            with open(filepath, 'wb') as f:
                f.write(file.read())

            try:
                # Get disease prediction
                detector = get_detector()
                result = detector.predict_disease(filepath)

                # Translate disease information if needed
                if lang_code == 'hi':
                    # Translate each field in the result
                    if 'disease_name' in result:
                        result['disease_name'] = translator.translate_crop_name(result['disease_name'], 'hi')
                    if 'description' in result:
                        result['description'] = translator.translate(result['description'], 'hi')
                    if 'prevention' in result and isinstance(result['prevention'], list):
                        result['prevention'] = translator.translate_list(result['prevention'], 'hi')
                    if 'treatment' in result and isinstance(result['treatment'], list):
                        result['treatment'] = translator.translate_list(result['treatment'], 'hi')

                # Clean up the temporary file
                os.remove(filepath)

                return jsonify(result)
            except Exception as e:
                # Clean up the temporary file in case of error
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f"Error in disease prediction: {e}")
                error_msg = translator.translate('Failed to analyze the image', lang_code)
                return jsonify({'error': error_msg}), 500

    except Exception as e:
        print(f"Error analyzing image: {e}")
        error_msg = translator.translate('Failed to analyze the image', 
                                         language[:2].lower() if language else 'en')
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)