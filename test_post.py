import requests

# 1. Login
res = requests.post("http://localhost:7860/auth/login", json={
    "email": "admin@dermai.com", # Admin is tested just to get a JWT, or I can use any
    "password": "admin123"
})
if res.status_code != 200:
    print("Login failed:", res.text)
    exit()
token = res.json()["access_token"]

import numpy as np
from PIL import Image
import io

img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format='JPEG')
valid_img_bytes = buf.getvalue()

# 2. Predict
files = {'file': ('dummy.jpg', valid_img_bytes, 'image/jpeg')}
data = {'user_info': '{"name":"Test","age":"25","symptoms":"None"}'}

pred_res = requests.post("http://localhost:7860/predict", 
                         headers={"Authorization": f"Bearer {token}"},
                         files=files, data=data)

print(pred_res.status_code)
print(pred_res.text)
