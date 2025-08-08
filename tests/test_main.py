import requests

url = "https://api.hyperbolic.xyz/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwYXJ0aHNoZXRoMzI2QGdtYWlsLmNvbSIsImlhdCI6MTczNjM1Njk2NH0.KXN599cevxC-QqVS439cdsOSSQX-cLiEC-ebt8Lw4oY"
}
data = {
    "messages": [{
      "role": "user",
      "content": "tell me about trading algorithms"
    }],
    "model": "openai/gpt-oss-20b",
    "max_tokens": 131072,
    "temperature": 0.7,
    "top_p": 0.8
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
