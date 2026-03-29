import os
import json
import random
import pickle
import numpy as np

# Suppress sklearn warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

print("🚀 Initializing Synthetic Dataset Generator for RBM Training...")

# Load our strictly controlled symptom categories (23 Classes)
with open('symptoms.json', 'r', encoding='utf-8') as f:
    symptoms_dict = json.load(f)

# Natural language structures to drastically inject realistic noise and variance into the AI
PREFIXES = [
    "I have been experiencing", "My skin suddenly has", "I woke up with", 
    "The doctor told me I might have", "There is a severe amount of", 
    "I'm deeply concerned about", "Help, my body is showing", "I feel",
    "Lately I've been suffering from", ""
]
SUFFIXES = [
    "on my left arm.", "and it's driving me crazy.", "since yesterday morning.",
    "which looks extremely bad.", "and it keeps spreading rapidly.",
    "near my chest.", "over the past few weeks.", "and I can't sleep.", ""
]

X_text = []
y_labels = []

print("⚙️ Generating 10,000 synthetic patient consultation scenarios natively...")

# Generate 400 distinct sentence combinations recursively for each of the 23 diseases
for disease, keywords in symptoms_dict.items():
    for _ in range(400):
        # Pick 1 to 4 keywords randomly
        k = random.randint(1, min(len(keywords), 4))
        chosen_symptoms = random.sample(keywords, k)
        
        # Build mathematical string
        symp_str = " and ".join(chosen_symptoms)
        
        # Add human probabilistic noise
        prefix = random.choice(PREFIXES)
        suffix = random.choice(SUFFIXES)
        
        sentence = f"{prefix} {symp_str} {suffix}".strip()
        
        X_text.append(sentence)
        y_labels.append(disease)

# Zip and uniquely shuffle identically
combined = list(zip(X_text, y_labels))
random.shuffle(combined)
X_text, y_labels = zip(*combined)

print(f"✅ Generated deeply randomized {len(X_text)} mathematical symptom rows.")
print("🧠 Training the TfidfVectorizer -> BernoulliRBM -> LogisticRegression Pipeline...")

# Construct the exact AI Pipeline structure
pipeline = Pipeline([
    # Step 1: Text Feature Extraction (Tokens)
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english', norm='l2')),
    # Step 2: The actual Deep Feature Extractor
    ('rbm', BernoulliRBM(n_components=64, learning_rate=0.05, n_iter=20, random_state=42, verbose=True)),
    # Step 3: Probabilistic mapping strictly back onto our 23 CNN models
    ('classifier', LogisticRegression(max_iter=500, class_weight='balanced'))
])

# Mathematically execute the fitting phase across both Neural systems
pipeline.fit(X_text, y_labels)

# Validate baseline score
accuracy = pipeline.score(X_text, y_labels)
print(f"🎯 RBM Multimodal Fusion Accuracy natively on training distribution: {accuracy*100:.2f}%")

# Save the mathematical weights directly into a portable pickle format alongside effnet.h5
with open('symptom_rbm.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("💾 Successfully exported robust 'symptom_rbm.pkl' memory state into local directory!")
print("Done. Ready to inject natively into app.py.")
