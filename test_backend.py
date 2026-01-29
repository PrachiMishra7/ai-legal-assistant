import requests

URL = "http://localhost:8000/analyze_fir"

fir = """
The accused assaulted the complainant and threatened to kill him.
"""

res = requests.post(URL, json={"fir_text": fir})

print("Status:", res.status_code)
print(res.json())
