# DermAI - Comprehensive Skin Disease Prediction & Secure Medical Portal

DermAI is an end-to-end, full-stack medical web application powered by **Deep Learning (TensorFlow/Keras)**. It bridges the gap between patients seeking preliminary skin condition analysis and certified dermatologists providing validated prescriptions. 

The entire platform is built with a strictly **zero-trust security architecture**, implementing military-grade **AES-256 Symmetric Encryption** for all Personally Identifiable Information (PII) and uploaded image assets directly inside a **MongoDB Atlas Cloud Database**.

---

## 🌟 Core Architecture & Features

### 1. Artificial Intelligence Engine
- **TensorFlow Convolutional Neural Network (CNN)**: The backend securely loads an optimized `.h5` model `effnet.h5` specifically tuned to identify visual anomalies across exactly **23 distinct skin conditions and benign tumors**, including:
  *Eczema, Melanoma, Psoriasis, Acne, Actinic Keratosis, Cellulitis, Lupus, Ringworm, Warts, Scabies, and more.*
- **Confidence Scoring & Symptom Matching**: Alongside the visual image prediction, the system cross-references the patient's inputted symptoms (e.g., "itching", "redness") against a clinical `symptoms.json` dictionary to calculate an intelligent **Symptom Match Score**.

### 2. Role-Based Access Control (RBAC) & Portals
The application natively handles three completely isolated user portals governed by **Flask-JWT-Extended** stateless sessions:

* **Patient Portal (`/`)**: 
  - Allows patients to sign up seamlessly. 
  - Patients can upload a photo of a rash/lesion and immediately receive the AI's preliminary diagnosis.
  - Generates immutable PDF-ready Digital Prescriptions that are unlocked *only* after a Doctor has officially approved the case.
* **Doctor Dashboard (`/doctor`)**: 
  - Doctors must register and be explicitly verified by a System Admin before gaining access.
  - Doctors view a real-time queue of all pending patient cases.
  - The Doctor decrypts the clinical information privately, reviews the AI's diagnosis, customizes the treatment protocol natively using a rich text prescription editor, and stamps "Approval."
* **Admin Dashboard (`/admin`)**: 
  - Automatically initializes when the MongoDB database is created (`admin@dermai.com`).
  - Audits the network, views all integrated users, and toggles Doctor account verifications.

### 3. Advanced Cloud Security (MongoDB Atlas & AES-256)
- **PII Text Scrambling**: Names, ages, contact points, and symptoms never touch the database as raw text. They are encrypted using `PyCryptodome (AES-GCM)` using a strict 32-byte secret key.
- **BSON Binary Image Storage**: The system is completely stateless. Uploaded images are **not** saved to a local folder. Instead, the raw image bytes are instantly mathematically scrambled using AES and saved securely as a `bson.binary.Binary` blob inside the specific MongoDB Case Document.
- **On-the-Fly Decryption**: When a Doctor views a case, the encrypted blob is streamed from MongoDB, decrypted dynamically in the Python instance's hardware memory, and sent briefly to the frontend via an ephemeral `blob:` URL.

---

## 🚀 Technology Stack

### Backend Interface (Python/Flask)
* **Framework**: Flask (REST API)
* **Authentication**: `flask-jwt-extended`
* **Database Driver**: `pymongo[srv]` for direct MongoDB Atlas integration
* **Machine Learning**: `tensorflow/keras`, `numpy`, `Pillow` (Image Processing)
* **Cryptography**: `pycryptodome` (AES-256-GCM) + `werkzeug.security` (Scrypt Password Hashing)

### Frontend Interface (React/Vite)
* **Framework**: React.js with typical Hooks API context (`useState`, `useEffect`, `useRef`)
* **Build Tool**: Vite (Instant hot-reloading)
* **Routing**: `react-router-dom`
* **Animations**: `framer-motion` (Fluid page transitions and modal popups)
* **Styling**: Modern, premium Vanilla CSS Modules utilizing Glassmorphism design aesthetics, `DM Sans`, and `Syne` fonts.

---

## 🛠 Complete Local Setup Guide

### 1. Prerequisites
- **Node.js** (v16.14.0 or newer)
- **Python** (3.9.0 - 3.12.0)
- **MongoDB Atlas Account** (Free M0 Sandbox Tier works perfectly!)

### 2. Environment Variables (`.env`)
Create a `.env` file exactly at the root directory of the project (`dermaAI/.env`) containing:
```env
# MongoDB Atlas Cloud Connection String 
# (Ensure your database password is URL Encoded if it has special characters like @)
MONGO_URI="mongodb+srv://<USERNAME>:<PASSWORD>@<YOUR_CLUSTER>.mongodb.net/dermai?retryWrites=true&w=majority"

# Advanced Cryptography Secrets (Store Offline Securely)
AES_SECRET_KEY="your_secure_32_byte_aes_key_here"
JWT_SECRET_KEY="your_secure_jwt_token_signer_here"
```

### 3. Initialize Python Backend
Open a terminal in the root `dermaAI` folder.
```bash
# Optional: Create a virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux

# Install all AI and Sever libraries
pip install -r requirements.txt
pip install "pymongo[srv]"

# Run the Flask API
python app.py
```
*The native API will securely bind to port `http://localhost:7860`. Upon its very first boot, it will automatically instantiate your `admin@dermai.com` root profile into MongoDB!*

### 4. Initialize React Frontend
Open a **second** terminal and navigate into the `frontend` folder.
```bash
cd frontend

# Install UI Dependencies
npm install

# Launch Vite Development Server
npm run dev
```
*The portal natively spins up on `http://localhost:5173`.*

---

## 📸 Interactive System Workflows

1. **System Admin Launch**: Navigate to `http://localhost:5173/login`, authenticate as `admin@dermai.com` using `admin123`.
2. **Doctor Onboarding**: Have a secondary user launch the app, hit "Register", and select the "Doctor" role. *They will not be able to log in until the Admin clicks "Approve Doctor" on their dashboard.*
3. **Patient Diagnosis Pipeline**:
    - A Patient logs in and uploads an image of a benign rash.
    - Python strips the image bytes, queries TensorFlow, matches the text mapped inside `symptoms.json`, encrypts everything with AES, pushes to Mongo, and returns a "Pending Validation" status.
    - An approved Doctor logs in, sees the case in their queue, safely views the decrypted image, edits the native medical formulations, and clicks "Approve Treatment".
    - The Patient's UI generates a localized, printable PDF Prescription.

---
*Disclaimer: DermAI is a conceptual application designed strictly for educational and technological demonstration purposes. It is not currently certified to substitute for professional clinical medical advice.*
