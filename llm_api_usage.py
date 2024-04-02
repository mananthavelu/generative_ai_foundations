import requests
API_TOKEN = 'actual_api_token_here'
API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()
data = query("Laptop is ")
print(data)