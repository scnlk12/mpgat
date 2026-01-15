import requests


def test_kimi_k2_api():
    url = "https://oneapi.dev2.aquaintelling.com/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-5KEg0Gdn4n69Yyte59D488Af4e2c44249b938a7d225eCcDa",
        "Content-Type": "application/json"
    }
    data = {
        "model": "Kimi-K2",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What can you do?"}
        ],
        "temperature": 0.5
    }

    response = requests.post(url, headers=headers, json=data)

    if response.ok:
        print("Request successful!")
        print("Response JSON:", response.json())
    else:
        print(f"Request failed with status code {response.status_code}")
        print("Response Text:", response.text)

test_kimi_k2_api()
