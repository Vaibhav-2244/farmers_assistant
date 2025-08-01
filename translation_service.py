import json
import re
from translate import Translator

class TranslationService:
    def __init__(self):
        # Load static translations
        try:
            with open('data/translations.json', 'r', encoding='utf-8') as f:
                self.static_translations = json.load(f)
        except Exception as e:
            print(f"Error loading translations: {e}")
            self.static_translations = {"hi": {}}
        
        # Initialize translators for different language pairs
        self.translators = {
            'en-hi': Translator(from_lang='en', to_lang='hi'),
            'hi-en': Translator(from_lang='hi', to_lang='en')
        }
        
        # Load crop names
        self.crop_names = self._load_crop_names()
        
        # Create reverse mappings for Hindi-English translations
        self.hindi_to_english = self._create_reverse_mappings()
    
    def _load_crop_names(self):
        """Load crop names from the crop data file"""
        try:
            with open('data/processed_crop_data.json', 'r') as f:
                crop_data = json.load(f)
            return list(crop_data.keys())
        except Exception as e:
            print(f"Error loading crop names: {e}")
            return []
    
    def _create_reverse_mappings(self):
        """Create a mapping from Hindi terms to English terms"""
        reverse_map = {}
        for lang, translations in self.static_translations.items():
            if lang == 'hi':
                for eng, hindi in translations.items():
                    if isinstance(hindi, str):  # Skip lists/dicts
                        reverse_map[hindi] = eng
        return reverse_map
    
    def detect_language(self, text):
        """
        Detect if text is in Hindi or English
        Returns language code: 'hi-IN' or 'en-US'
        """
        # Count Devanagari Unicode characters (Hindi)
        hindi_char_count = len(re.findall(r'[\u0900-\u097F]', text))
        
        # Additional Hindi indicators
        hindi_indicators = [
            'फसल', 'मौसम', 'मूल्य', 'तापमान', 'वर्षा', 'रोग', 'कीट',
            'का', 'के', 'की', 'में', 'है', 'कैसे', 'क्या'
        ]
        
        # Check if any Hindi indicators are in the text
        has_indicators = any(indicator in text for indicator in hindi_indicators)
        
        # Return 'hi-IN' if we have Hindi characters or indicators
        if hindi_char_count > 0 or has_indicators:
            return 'hi-IN'
        return 'en-US'
    
    def translate(self, text, target_lang='hi'):
        """
        Translate text to target language
        target_lang can be 'hi', 'hi-IN', 'en', 'en-US'
        """
        if not text:
            return text
            
        # Normalize language code
        lang_code = target_lang[:2].lower()
        
        # Return original if target is English and text is already in English
        if lang_code == 'en':
            return text
        
        # Check static translations first
        if lang_code in self.static_translations:
            if text in self.static_translations[lang_code]:
                return self.static_translations[lang_code][text]
        
        # For sentence translation
        try:
            if lang_code == 'hi':
                return self.translators['en-hi'].translate(text)
            elif lang_code == 'en':
                return self.translators['hi-en'].translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
        
        # Return original text if translation fails
        return text
    
    def translate_crop_name(self, crop_name, target_lang='hi'):
        """Special handling for crop names"""
        if not crop_name:
            return crop_name
            
        # Normalize language code
        lang_code = target_lang[:2].lower()
        crop_name_lower = crop_name.lower()
        
        # English to Hindi
        if lang_code == 'hi':
            # Check in static translations
            if 'hi' in self.static_translations:
                for eng, hindi in self.static_translations['hi'].items():
                    if eng.lower() == crop_name_lower or eng.lower() == crop_name_lower.capitalize():
                        return hindi
                        
            # Capitalize first letter for presentation
            return crop_name.capitalize()
        
        # Hindi to English
        elif lang_code == 'en':
            # Check in reverse mappings
            if crop_name in self.hindi_to_english:
                return self.hindi_to_english[crop_name].capitalize()
                
        # Return original with capitalization if no translation found
        return crop_name.capitalize()
    
    def translate_list(self, item_list, target_lang='hi'):
        """Translate a list of items"""
        if not item_list or not isinstance(item_list, list):
            return item_list
            
        translated_list = []
        for item in item_list:
            translated_list.append(self.translate(item, target_lang))
        
        return translated_list
    
    def translate_dict(self, data_dict, target_lang='hi', keys_to_translate=None):
        """
        Translate values in a dictionary
        If keys_to_translate is provided, only translate those keys
        """
        if not data_dict or not isinstance(data_dict, dict):
            return data_dict
            
        translated_dict = {}
        
        for key, value in data_dict.items():
            # Decide whether to translate this key's value
            should_translate = keys_to_translate is None or key in keys_to_translate
            
            if should_translate:
                if isinstance(value, str):
                    translated_dict[key] = self.translate(value, target_lang)
                elif isinstance(value, list):
                    translated_dict[key] = self.translate_list(value, target_lang)
                elif isinstance(value, dict):
                    translated_dict[key] = self.translate_dict(value, target_lang)
                else:
                    translated_dict[key] = value
            else:
                translated_dict[key] = value
                
        return translated_dict