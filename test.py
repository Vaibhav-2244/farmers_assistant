import requests

params = {
    'api-key': '579b464db66ec23bdd000001e5a09c5cfce9410143b1e20a9e63e4a4',
    'format': 'json',
    'filters[Commodity]': 'Wheat',
    'limit': 5
}
response = requests.get('https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24', params=params)
print(response.json())