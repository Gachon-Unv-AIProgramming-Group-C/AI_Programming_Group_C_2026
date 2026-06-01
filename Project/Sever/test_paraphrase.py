import requests
import sys

url = "http://127.0.0.1:8001/paraphrase"
question = "대만의 수도는 어디인가요?" if len(sys.argv) < 2 else sys.argv[1]

print(f"Sending question: {question}")
try:
    r = requests.post(url, json={"question": question})
    print("Response status:", r.status_code)
    print("Response JSON:", r.json())
except Exception as e:
    print("Error:", e)
