import os
import io
import json
import uuid
import pickle
from datetime import datetime
from bson import ObjectId
from bson.binary import Binary

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required, get_jwt_identity
)
from werkzeug.security import generate_password_hash, check_password_hash

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Import custom DB and Encryption utils
from db import users_collection, cases_collection
from encryption_utils import encrypt_text, decrypt_text, encrypt_file, decrypt_file

# ─────────────────────────────────────────────
#  Absolute base directory & App Init
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__,
            static_folder=os.path.join(BASE_DIR, 'frontend', 'dist'),
            static_url_path='')
CORS(app)

# JWT Setup
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'default_jwt_secret_for_dev')
jwt = JWTManager(app)

# ─────────────────────────────────────────────
#  Lazy-load ML model and data files
# ─────────────────────────────────────────────
# Load the RBM Symptom Pipeline
RBM_PATH = os.path.join(BASE_DIR, "symptom_rbm.pkl")
rbm_pipeline = None
if os.path.exists(RBM_PATH):
    with open(RBM_PATH, 'rb') as f:
        rbm_pipeline = pickle.load(f)
    print("✅ RBM Symptom Pipeline loaded successfully.")
else:
    print("⚠️  symptom_rbm.pkl not found. Falling back to keyword matching.")
WEIGHTS_PATH     = os.path.join(BASE_DIR, "best_weights.weights.h5")
ARCH_PATH        = os.path.join(BASE_DIR, "model_architecture.json")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
SYMPTOMS_PATH    = os.path.join(BASE_DIR, "symptoms.json")
MEDICINES_PATH   = os.path.join(BASE_DIR, "medicines.json")
IMG_SIZE         = (224, 224)

# Safe defaults — app starts even if model fails to load
model = None
class_names = []
DISEASE_SYMPTOMS = {}
MEDICINES_DB = {}

print("Loading model architecture...")
try:
    with open(ARCH_PATH, 'r', encoding='utf-8') as f:
        model_json = f.read()
    model = tf.keras.models.model_from_json(model_json)
    
    print("Loading trained weights...")
    model.load_weights(WEIGHTS_PATH)
    
    with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
        
    with open(SYMPTOMS_PATH, 'r', encoding='utf-8') as f:
        DISEASE_SYMPTOMS = json.load(f)
        
    with open(MEDICINES_PATH, 'r', encoding='utf-8') as f:
        MEDICINES_DB = json.load(f)
        
    print(f"✅ Model loaded! {len(class_names)} classes ready.")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    import traceback
    traceback.print_exc()

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB').resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0
    return img_array

# ─────────────────────────────────────────────
#  Auth Routes
# ─────────────────────────────────────────────
@app.route('/auth/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'patient') # default to patient

    if not name or not email or not password:
        return jsonify({"error": "Missing required fields"}), 400

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already exists"}), 400

    new_user = {
        "name": name,
        "email": email,
        "password": generate_password_hash(password),
        "role": role,
        "approved": True if role == 'patient' else False # patients are auto-approved, doctors pend
    }
    
    users_collection.insert_one(new_user)
    
    msg = "User registered successfully."
    if role == 'doctor':
        msg += " Please wait for an Admin to approve your account."
        
    return jsonify({"message": msg}), 201

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    user = users_collection.find_one({"email": email})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({"error": "Invalid email or password"}), 401

    if not user.get('approved'):
        return jsonify({"error": "Account pending admin approval"}), 403

    identity_str = json.dumps({"id": str(user["_id"]), "role": user["role"], "name": user["name"]})
    access_token = create_access_token(identity=identity_str)
    return jsonify({
        "access_token": access_token, 
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "role": user["role"]
        }
    }), 200

# ─────────────────────────────────────────────
#  Patient Routes
# ─────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
@jwt_required()
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Server may still be initializing."}), 503
        current_user = json.loads(get_jwt_identity())
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        user_info = {}
        if 'user_info' in request.form:
            try:
                user_info = json.loads(request.form['user_info'])
            except Exception:
                pass

        user_name     = user_info.get("name", "Patient")
        user_age      = str(user_info.get("age", "N/A"))
        user_phone    = user_info.get("phone", "")
        symptoms_text = user_info.get("symptoms", "")
        user_symptoms = [s.strip() for s in symptoms_text.replace(",", " ").split() if s.strip()]

        # Predict
        img_bytes     = file.read()
        processed_img = preprocess_image(img_bytes)
        predictions   = model.predict(processed_img)[0]

        top_idx           = int(np.argmax(predictions))
        confidence        = float(predictions[top_idx] * 100)
        predicted_disease = class_names[top_idx]

        # Pre-initialise so they're always defined regardless of RBM path
        matching   = []
        missing    = []
        match_score = "No symptom data available"

        # ── Multimodal Fusion: RBM Symptom Analysis ──────────────────────────
        if rbm_pipeline and symptoms_text.strip():
            try:
                # Get probability distribution from RBM over all 23 diseases
                rbm_proba = rbm_pipeline.predict_proba([symptoms_text])[0]
                rbm_classes = rbm_pipeline.classes_

                # Build a score map from the RBM output
                rbm_scores = {cls: prob for cls, prob in zip(rbm_classes, rbm_proba)}

                # CNN score for the top predicted disease (normalized 0–1)
                cnn_score = predictions[top_idx]  # already between 0 and 1

                # RBM score for the top CNN disease
                rbm_score_for_top = rbm_scores.get(predicted_disease, 0.0)

                # Fuse: 60% CNN image confidence + 40% RBM symptom confidence
                fused_scores = {}
                for i, cls in enumerate(class_names):
                    cnn_val = float(predictions[i])
                    rbm_val = rbm_scores.get(cls, 0.0)
                    fused_scores[cls] = 0.60 * cnn_val + 0.40 * rbm_val

                # Re-select best class after fusion
                predicted_disease = max(fused_scores, key=fused_scores.get)
                fused_confidence   = fused_scores[predicted_disease] * 100

                match_score = (
                    f"RBM Symptom Confidence: {rbm_score_for_top*100:.1f}% | "
                    f"Image Confidence: {cnn_score*100:.1f}% | "
                    f"Fused Score: {fused_confidence:.1f}%"
                )
                confidence = round(fused_confidence, 2)
            except Exception as rbm_err:
                print(f"RBM inference failed, using CNN only: {rbm_err}")
                known_symptoms = DISEASE_SYMPTOMS.get(predicted_disease, [])
                matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known_symptoms)]
                match_score = (
                    f"{len(matching)} of {len(known_symptoms)} typical symptoms match"
                    if known_symptoms else "No symptom data available"
                )
        else:
            # Fallback: basic keyword matching (when RBM is unavailable)
            known_symptoms = DISEASE_SYMPTOMS.get(predicted_disease, [])
            matching = [s for s in user_symptoms if any(s.lower() == k.lower() for k in known_symptoms)]
            match_score = (
                f"{len(matching)} of {len(known_symptoms)} typical symptoms match"
                if known_symptoms else "No symptom data available"
            )
        # ── End Multimodal Fusion ─────────────────────────────────────────────
        meds = MEDICINES_DB.get(predicted_disease, {})

        # Save Encrypted File entirely in MongoDB Core
        encrypted_img = encrypt_file(img_bytes)

        # AES-256 Encrypt Patient Data
        encrypted_patient_info = {
            "name": encrypt_text(user_name),
            "age": encrypt_text(user_age),
            "phone": encrypt_text(user_phone),
            "symptoms": encrypt_text(symptoms_text)
        }
        
        # Save Case to MongoDB
        case_doc = {
            "patient_user_id": current_user["id"],
            "patient_info": encrypted_patient_info,
            "predicted_disease": encrypt_text(predicted_disease), # Encrypt disease
            "confidence": confidence,
            "match_score": match_score,
            "medicines_recommended": meds, # standard dictionary, can be edited
            "approved_by_doctor": False,
            "status": "Pending Doctor Review",
            "image_data": Binary(encrypted_img), # Store directly in Mongo!
            "created_at": datetime.utcnow()
        }
        cases_collection.insert_one(case_doc)

        return jsonify({
            "disease": predicted_disease,
            "confidence": f"{confidence:.2f}",
            "match_score": match_score,
            "matching": matching,
            "missing": missing,
            "medicines": meds,
            "status": "Pending Doctor Review"
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print("❌ Error:", e)
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/patient/cases', methods=['GET'])
@jwt_required()
def patient_cases():
    current_user = json.loads(get_jwt_identity())
    cases = list(cases_collection.find({"patient_user_id": current_user["id"]}).sort("created_at", -1))
    
    # Decrypt what the patient sees
    results = []
    for c in cases:
        c['_id'] = str(c['_id'])
        c.pop('image_data', None) # Remove binary data from JSON payload
        c['predicted_disease'] = decrypt_text(c['predicted_disease'])
        
        # Decrypt patient info
        pi = c.get('patient_info', {})
        c['patient_info'] = {
            "name": decrypt_text(pi.get('name', '')),
            "age": decrypt_text(pi.get('age', '')),
            "phone": decrypt_text(pi.get('phone', '')),
            "symptoms": decrypt_text(pi.get('symptoms', ''))
        }
        results.append(c)
        
    return jsonify(results), 200

# ─────────────────────────────────────────────
#  Doctor Routes
# ─────────────────────────────────────────────
@app.route('/doctor/cases', methods=['GET'])
@jwt_required()
def doctor_cases():
    current_user = json.loads(get_jwt_identity())
    if current_user['role'] != 'doctor':
        return jsonify({"error": "Unauthorized"}), 403
        
    cases = list(cases_collection.find().sort("created_at", -1))
    results = []
    for c in cases:
        c['_id'] = str(c['_id'])
        c.pop('image_data', None) # Remove binary data from JSON payload
        c['predicted_disease'] = decrypt_text(c['predicted_disease'])
        pi = c.get('patient_info', {})
        c['patient_info'] = {
            "name": decrypt_text(pi.get('name', '')),
            "age": decrypt_text(pi.get('age', '')),
            "phone": decrypt_text(pi.get('phone', '')),
            "symptoms": decrypt_text(pi.get('symptoms', ''))
        }
        results.append(c)
    return jsonify(results), 200

@app.route('/doctor/cases/<case_id>/approve', methods=['PUT'])
@jwt_required()
def doctor_approve_case(case_id):
    current_user = json.loads(get_jwt_identity())
    if current_user['role'] != 'doctor':
        return jsonify({"error": "Unauthorized"}), 403
        
    data = request.get_json()
    edited_medicines = data.get('medicines_recommended')
    
    cases_collection.update_one(
        {"_id": ObjectId(case_id)},
        {"$set": {
            "medicines_recommended": edited_medicines,
            "approved_by_doctor": True,
            "doctor_id": current_user["id"],
            "status": "Approved by Doctor"
        }}
    )
    return jsonify({"message": "Case approved successfully"}), 200

# Route to serve decrypted image to doctors/patients
@app.route('/images/<case_id>')
@jwt_required()
def get_image(case_id):
    case = cases_collection.find_one({"_id": ObjectId(case_id)})
    if not case or "image_data" not in case:
        return jsonify({"error": "Image not found"}), 404
        
    encrypted_data = case["image_data"]
    decrypted_data = decrypt_file(encrypted_data)
    
    return send_file(
        io.BytesIO(decrypted_data),
        mimetype='image/jpeg',
        as_attachment=False,
        download_name='scan.jpg'
    )

# ─────────────────────────────────────────────
#  Admin Routes
# ─────────────────────────────────────────────
@app.route('/admin/users', methods=['GET'])
@jwt_required()
def admin_users():
    current_user = json.loads(get_jwt_identity())
    if current_user['role'] != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
        
    users = list(users_collection.find({}, {"password": 0}))
    for u in users:
        u['_id'] = str(u['_id'])
    return jsonify(users), 200

@app.route('/admin/approve_doctor/<user_id>', methods=['PUT'])
@jwt_required()
def admin_approve_doctor(user_id):
    current_user = json.loads(get_jwt_identity())
    if current_user['role'] != 'admin':
        return jsonify({"error": "Unauthorized"}), 403
        
    users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"approved": True}}
    )
    return jsonify({"message": "Doctor approved successfully"}), 200


# ─────────────────────────────────────────────
#  React Catch-all Route for Client-Side Routing
# ─────────────────────────────────────────────
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    import os
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)