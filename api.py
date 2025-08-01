import requests

# Weather API Configuration
WEATHER_API_KEY = "3d63e83589aad73062a800467d2784e8"
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# AgMarket API Configuration
AGMARKET_API_KEY = "579b464db66ec23bdd000001e5a09c5cfce9410143b1e20a9e63e4a4"
AGMARKET_API_URL = "https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24"

def test_weather_api(city="gurgaon"):
    params = {
        "q": city,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    try:
        response = requests.get(WEATHER_API_URL, params=params)
        print("Weather API Response:")
        print(response.json())
    except Exception as e:
        print("Weather API Error:", e)

def test_agmarket_api():
    params = {
        "api-key": AGMARKET_API_KEY,
        "format": "json",  # Make sure to specify the output format
        "limit": 5         # Limit number of records for testing
    }
    try:
        response = requests.get(AGMARKET_API_URL, params=params)
        print("AgMarket API Response:")
        print(response.json())
    except Exception as e:
        print("AgMarket API Error:", e)

if __name__ == "__main__":
    print("Testing Weather API...\n")
    test_weather_api()

    print("\nTesting AgMarket API...\n")
    test_agmarket_api()
