import requests

# URL ของ API ของ Typhoon
api_url = "https://api.opentyphoon.ai/v1/chat/completions"  # URL ที่คุณต้องใช้ของ Typhoon

# API Key ที่ได้รับจาก Typhoon
api_key = "your-api-key"

# ข้อความที่ส่งไปให้ Typhoon ตอบ
data = {
    "model": "typhoon-v1.5x-70b-instruct",  # ชื่อโมเดลที่คุณต้องการใช้
    "messages": [
        {"role": "user", "content": "ยาลดน้ำมูกมีอะไรบ้าง"}
    ]
}

# การตั้งค่า header
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# ส่งคำขอไปยัง Typhoon API
response = requests.post(api_url, json=data, headers=headers)

# ตรวจสอบผลลัพธ์
if response.status_code == 200:
    result = response.json()
    print("Response:", result['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)
