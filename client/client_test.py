import requests

url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "LGAI-EXAONE/EXAONE-4.0-1.2B",
    "messages": [
        {"role": "user", "content": "EXAONE을 한국어로 한 문장 소개해줘."}
    ],
    "temperature": 0.2
}

r = requests.post(url, json=payload, timeout=120)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
