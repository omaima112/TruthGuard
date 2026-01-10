from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__)

# Load stop words
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
print("\n" + "=" * 60)
print("🛡️  TRUTHGUARD - AI Fake News Detector")
print("=" * 60)

try:
    model = joblib.load('models/truthguard_model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("✅ AI Model loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Model files not found!")
    print("📝 Please run 'python train_model.py' first to train the model.")
    exit(1)

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Analyze claim function
def analyze_claim(claim):
    # Clean the text
    cleaned = clean_text(claim)
    
    # Convert to numbers
    vectorized = vectorizer.transform([cleaned])
    
    # Get prediction
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    
    # Get confidence
    confidence = probability[prediction] * 100
    
    # Determine verdict
    if prediction == 0:  # Real
        if confidence >= 80:
            verdict = "Highly Credible"
            color = "green"
        elif confidence >= 60:
            verdict = "Likely True"
            color = "lightgreen"
        else:
            verdict = "Possibly True"
            color = "yellow"
    else:  # Fake
        if confidence >= 80:
            verdict = "Highly Suspicious"
            color = "red"
        elif confidence >= 60:
            verdict = "Likely False"
            color = "orange"
        else:
            verdict = "Questionable"
            color = "yellow"
    
    return {
        'verdict': verdict,
        'confidence': round(confidence, 2),
        'is_fake': bool(prediction),
        'color': color,
        'real_probability': round(probability[0] * 100, 2),
        'fake_probability': round(probability[1] * 100, 2)
    }

# Routes
@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/detector')
def detector():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    claim = data.get('claim', '')
    
    if not claim or len(claim.strip()) < 10:
        return jsonify({'error': 'Please enter a claim with at least 10 characters'}), 400
    
    result = analyze_claim(claim)
    return jsonify(result)

if __name__ == '__main__':
    print("\n🌐 Starting TruthGuard Web Server...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)