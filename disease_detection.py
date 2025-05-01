import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import json
from PIL import Image
import os
import requests
from config import WEATHER_API_KEY, WEATHER_API_URL

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        try:
            # Try loading ResNet50 with SSL verification
            # self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True, trust_repo=True)
            self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        except Exception as e:
            print(f"Error loading model {e}")
            print("Attempting alternative model initialization...")
            
            # Alternative: Create ResNet50 without pre-trained weights
            from torchvision.models import resnet50
            self.model = models.resnet50(weights=None)
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class DiseaseDetector:
    def __init__(self):
        try:
            # Load the class labels
            with open('data/class_indices.json', 'r', encoding='utf-8') as f:
                self.class_indices = json.load(f)
                num_classes = len(self.class_indices['selected_classes'])
            
            # Initialize model
            self.model = PlantDiseaseModel(num_classes)
            
            # Load model weights
            checkpoint = torch.load('models/plant_disease_model.pth', 
                                 map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            
            # Define image transformations
            self.transform = transforms.Compose([
                transforms.Resize(180),
                transforms.CenterCrop(160),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])
            
            # Load the disease responses
            with open('data/response.json', 'r', encoding='utf-8') as f:
                self.disease_info = json.load(f)
                
            # Load translations
            with open('data/translations.json', 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
                
        except FileNotFoundError as e:
            print(f"Error loading required files: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """Preprocess the image for model input."""
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
            return img_tensor
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict_disease(self, image_path, language='en'):
        """Predict the disease from the image and return relevant information."""
        try:
            # Preprocess the image
            img_tensor = self.preprocess_image(image_path)
            if img_tensor is None:
                return {"error": "Failed to process the image"}

            # Make prediction
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
            # Convert to Python types
            predicted_idx = predicted_idx.item()
            confidence = confidence.item()

            # Get the disease name
            disease_name = self.class_indices['selected_classes'][predicted_idx]

            # Get additional information from response.json
            disease_info = self.get_disease_info(disease_name, language)

            return {
                "disease_name": disease_name,
                "confidence": confidence,
                "description": disease_info.get("description", "No description available"),
                "prevention": disease_info.get("prevention", []),
                "treatment": disease_info.get("treatment", [])
            }

        except Exception as e:
            print(f"Error in disease prediction: {e}")
            return {"error": "Failed to analyze the image"}

    def normalize_disease_name(self, disease_name):
        """Normalize disease name to match response.json format."""
        # Convert to lowercase
        name = disease_name.lower()
        
        # Replace multiple underscores with single underscore
        name = name.replace("___", "_").replace("__", "_")
        
        # Handle special cases
        name_mapping = {
            "tomato_spider_mites_two_spotted_spider_mite": "tomato_spider_mites",
            "potato___early_blight": "potato_early_blight",
            "potato___late_blight": "potato_late_blight",
            "pepper__bell___bacterial_spot": "pepper_bacterial_spot"
        }
        
        # Check if we have a direct mapping
        if name in name_mapping:
            return name_mapping[name]
            
        # Remove any extra spaces and underscores
        name = name.strip().replace(" ", "_")
        
        # Remove any trailing underscores
        name = name.rstrip("_")
        
        return name

    def get_disease_info(self, disease_name, language='en'):
        """Get disease information from response.json with language support."""
        # Normalize the disease name
        clean_name = self.normalize_disease_name(disease_name)
        
        # Try to get the disease info
        disease_info = self.disease_info.get(clean_name)
        
        if disease_info is None:
            # If not found, try to find a partial match
            for key in self.disease_info.keys():
                if clean_name in key or key in clean_name:
                    disease_info = self.disease_info[key]
                    break
        
        if disease_info is None:
            return {
                "description": self.translate("No information available for", language) + f" {disease_name}",
                "prevention": [self.translate("No prevention information available", language)],
                "treatment": [self.translate("No treatment information available", language)]
            }
        
        # Translate the information if not in English
        if language != 'en':
            translated_info = {
                "description": self.translate(disease_info.get("description", ""), language),
                "prevention": [self.translate(item, language) for item in disease_info.get("prevention", [])],
                "treatment": [self.translate(item, language) for item in disease_info.get("treatment", [])]
            }
            return translated_info
            
        return disease_info

    def translate(self, text, target_language):
        """Translate text to target language."""
        if target_language == 'en':
            return text
            
        # Check if we have a direct translation
        if text in self.translations.get(target_language, {}):
            return self.translations[target_language][text]
            
        # If no direct translation, return original text
        return text

    def get_weather_info(self, city):
        """Get weather information for a city."""
        try:
            params = {
                'q': city,
                'appid': WEATHER_API_KEY,
                'units': 'metric'
            }
            response = requests.get(WEATHER_API_URL, params=params)
            data = response.json()
            
            if response.status_code == 200:
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'description': data['weather'][0]['description'],
                    'wind_speed': data['wind']['speed']
                }
            else:
                return None
        except Exception as e:
            print(f"Error getting weather data: {e}")
            return None

# Initialize the detector
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = DiseaseDetector()
    return detector 