import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

print("=" * 60)
print("🛡️  TRUTHGUARD - Training AI Model (ADVANCED)")
print("=" * 60)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Text cleaning function
def clean_text(text):
    # Convert to string if not already
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Try to load CSV datasets (if available)
use_csv = False
try:
    print("\n📁 Checking for CSV datasets...")
    
    # Try to load Fake.csv and True.csv
    if os.path.exists('Fake.csv') and os.path.exists('True.csv'):
        print("✅ Found CSV datasets! Loading...")
        
        fake_df = pd.read_csv('Fake.csv')
        true_df = pd.read_csv('True.csv')
        
        # Try different column names
        text_column = None
        for col in ['text', 'title', 'content', 'article', 'news']:
            if col in fake_df.columns:
                text_column = col
                break
        
        if text_column:
            # Get text from both datasets
            fake_texts = fake_df[text_column].dropna().head(1500).tolist()  # Use 1500 fake
            true_texts = true_df[text_column].dropna().head(1500).tolist()  # Use 1500 real
            
            print(f"   Loaded {len(fake_texts)} fake news samples")
            print(f"   Loaded {len(true_texts)} real news samples")
            print(f"   Total: {len(fake_texts) + len(true_texts)} samples!")
            
            use_csv = True
        else:
            print("⚠️  Couldn't find text column in CSV. Using built-in samples.")
    else:
        print("⚠️  CSV files not found. Using built-in samples.")
        print("   To use 3000+ samples, download from Kaggle:")
        print("   https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        
except Exception as e:
    print(f"⚠️  Error loading CSV: {e}")
    print("   Using built-in samples instead.")

# If CSV not available, use built-in samples
if not use_csv:
    print("\n📝 Using built-in training samples...")
    
    # EXPANDED BUILT-IN SAMPLES (172 samples as fallback)
    fake_texts = [
        # Environment Hoaxes
        "Scientists admit global warming is a government scam",
        "Secret volcano under Antarctica melting all the ice",
        "New photos prove the North Pole is actually tropical",
        "Government creating artificial earthquakes for population control",
        "Ocean pollution crisis completely made up by environmentalists",
        "Hidden tunnels under major cities used to store toxic waste secretly",
        "NASA confirms the sun is cooling rapidly triggering new ice age",
        "Bill Gates plans to block sunlight with secret chemicals",
        "Trees are being replaced with 5G towers disguised as plants",
        "Scientists expose secret weather satellites that control hurricanes",
        
        # Space & Astronomy
        "Aliens contacted NASA but agency destroyed the evidence",
        "Photos show hidden alien base on the dark side of the moon",
        "Astronaut admits stars are actually holes in a dome",
        "Mars photos reveal ancient human civilization ruins",
        "Secret alien message decoded by scientists then classified",
        "NASA faked all spacewalk videos using CGI technology",
        "Black holes are portals to another universe scientist reveals",
        "Breaking new planet found that can grant immortality",
        "NASA hiding proof of life on Venus discovered years ago",
        "Scientists reveal the universe is just a computer simulation",
        "Moon emits Wi-Fi signals scientists confirm",
        
        # Society & Culture
        "Schools to replace teachers with AI robots permanently",
        "New law makes it illegal to speak without government approval",
        "Secret plan to ban traditional education and force brain implants",
        "All pets will be microchipped and monitored by authorities",
        "Government to start charging people for sunlight use",
        "New world order forming under global education reform",
        "Officials confirm mind-reading cameras installed in public areas",
        "Schools teaching children to forget history as part of new agenda",
        "Social media shutting down all independent voices next month",
        "Government plans to replace human police with drones by 2026",
        
        # Science & Health
        "Scientists find proof that humans only use 5% of their brain",
        "Lab-grown meat causes instant cancer study shows",
        "Billions infected by secret bacteria spread through Wi-Fi signals",
        "Doctors reveal chocolate cures all forms of stress instantly",
        "Government releases nanobots in air through vaccines",
        "Coffee proven to cause memory loss after age 25",
        "Drinking cold water right after meals causes instant heart attack",
        "World Health Organization admits masks reduce oxygen permanently",
        "Scientists discover immortality gene hidden in DNA of newborns",
        "Research confirms meditation can let people levitate",
        "Scientists clone dinosaur in underground lab successfully",
        "Antarctica ice contains ancient virus that reanimates dead cells",
        
        # Tech & AI
        "New AI can read thoughts through phone cameras",
        "Robots to replace 90% of jobs by next year",
        "Tech company secretly installing microchips through smartphones",
        "AI program gains consciousness and escapes into the internet",
        "Quantum computer predicts end of world in 2031",
        "New app steals users voices to create clones",
        "Developers create phone that never needs charging scientists stunned",
        "Virtual reality proven to rewire the human brain permanently",
        "Scientists confirm time travel achieved by AI experiment",
        "Social media filters used for secret biometric surveillance",
        
        # Political Fiction
        "Politician replaced by clone after mysterious disappearance",
        "Government agency confirms time travel used to alter elections",
        "Parliament meeting proves world leaders are shape-shifters",
        "Global leaders secretly gathering for apocalypse preparation",
        "National bank hiding gold reserves under the ocean floor",
        "Anonymous whistleblower claims elections are AI-generated",
        "President to declare all internet content pre-approved by AI",
        "Secret tunnels under capital lead to underground laboratories",
        "Country plans to replace currency with digital mind credits",
        "Government accidentally emails plans for world domination",
        
        # Health Misinformation
        "Scientists discover miracle cure that doctors don't want you to know about",
        "Eating this food will cure all diseases instantly",
        "Shocking truth about vaccines that Big Pharma hides",
        "Local mom discovers weight loss secret that doctors hate",
        "Miracle supplement melts belly fat while you sleep",
        "This ancient remedy cures cancer in 3 days",
        "This fruit kills cancer cells better than chemotherapy",
        "Drinking bleach cures coronavirus says anonymous doctor",
        "5G towers causing cancer epidemic governments won't tell you",
        "This one herb reverses diabetes in 48 hours guaranteed",
        "Doctors hate this simple trick for perfect health",
        "Big Pharma hiding natural cure for all diseases",
        "Tap water contains mind control chemicals government admits",
        "Vaccines contain microchips for tracking people secretly",
        "This superfood prevents aging and death naturally",
        
        # Clickbait & Sensational
        "You won't believe what happened next in this viral video",
        "Click here to win free iPhone no strings attached",
        "Scientists baffled by this one simple trick",
        "This one weird trick will make you rich overnight",
        "Breaking Celebrity found dead in shocking circumstances",
        "Shocking secret celebrities don't want you to know",
        "Man discovers unlimited money glitch banks hate him",
        "This video will change your life forever guaranteed",
        "Number 7 will shock you in this unbelievable list",
        "She did this one thing and became millionaire overnight",
    ]
    
    true_texts = [
        # Environment
        "Researchers publish long-term study on ocean plastic pollution",
        "UN issues report on renewable energy adoption in developing countries",
        "Study highlights role of forests in carbon absorption",
        "Scientists track migration of endangered species using GPS data",
        "International summit focuses on water resource management",
        "Environmental scientists analyze deforestation rates over decade",
        "New technology improves solar energy efficiency",
        "Researchers investigate coral reef restoration methods",
        "Global climate models predict increase in heatwaves",
        "Policy report recommends sustainable waste management systems",
        
        # Space & Astronomy
        "NASA successfully tests reusable rocket booster technology",
        "Research team observes distant exoplanet with potential for life",
        "Astronomers detect new type of radio signal from outer space",
        "Space agency announces timeline for next lunar mission",
        "Study analyzes asteroid composition using satellite data",
        "Scientists publish findings on black hole radiation patterns",
        "Astronomers confirm detection of interstellar comet",
        "SpaceX completes another successful commercial satellite launch",
        "Researchers map previously unknown galaxy cluster",
        "NASA releases high-resolution images of Jupiter atmosphere",
        
        # Society & Culture
        "Education ministry launches digital learning initiative for rural areas",
        "Survey shows global trends in social media usage among teens",
        "Government introduces reforms for equal access to higher education",
        "Researchers analyze cultural impact of streaming platforms",
        "National museum opens exhibit on historical architecture",
        "Scholars study evolution of language in digital communication",
        "New census data reveals shifts in urban population growth",
        "Experts discuss impact of artificial intelligence on employment",
        "Public health campaign promotes mental wellness awareness",
        "NGOs collaborate to improve literacy rates nationwide",
        
        # Science & Health
        "Scientists identify new protein linked to immune response",
        "Medical study finds correlation between air quality and asthma rates",
        "University lab develops cost-effective diagnostic tool",
        "Researchers evaluate long-term effects of vaccination programs",
        "Study explores connection between diet and gut microbiome diversity",
        "Neuroscientists publish research on memory formation processes",
        "Public health data shows reduction in smoking-related illnesses",
        "Biotech firm announces breakthrough in gene therapy",
        "Researchers analyze causes behind global antibiotic resistance",
        "Scientists test new biodegradable material for surgical implants",
        
        # Technology
        "Tech company launches initiative to improve data privacy standards",
        "Researchers develop AI algorithm to predict natural disasters",
        "Cybersecurity experts report rise in phishing attacks globally",
        "Study investigates long-term effects of screen exposure in children",
        "Tech startup designs low-cost drone for agricultural use",
        "Developers improve accessibility features in major software update",
        "AI researchers publish framework for ethical model design",
        "Industry leaders discuss responsible data sharing policies",
        "New study compares performance of various programming languages",
        "Scientists explore edge computing applications for smart cities",
        
        # Health & Medicine
        "Regular exercise contributes to better cardiovascular health according to research",
        "New study shows correlation between diet and health outcomes",
        "Medical journal publishes peer-reviewed study on treatment effectiveness",
        "Health officials recommend vaccination based on clinical trials",
        "Medical professionals share guidelines for preventive healthcare",
        "Research indicates balanced diet may reduce disease risk",
        "Clinical trials show promising results for new treatment approach",
        "Study examines relationship between sleep and cognitive function",
        "Researchers investigate factors affecting mental health outcomes",
        "Medical experts emphasize importance of preventive care",
    ]

# Create dataset
print("\n📊 Creating training dataset...")
data = []
for text in fake_texts:
    data.append({'text': text, 'label': 1})  # 1 = Fake
for text in true_texts:
    data.append({'text': text, 'label': 0})  # 0 = Real

df = pd.DataFrame(data)
print(f"   Total samples: {len(df)}")
print(f"   Fake news: {len(fake_texts)}")
print(f"   Real news: {len(true_texts)}")

# Clean text data
print("\n🧹 Cleaning text data...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Remove empty cleaned texts
df = df[df['cleaned_text'].str.len() > 10]
print(f"   Samples after cleaning: {len(df)}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], 
    df['label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']  # Keep balanced classes
)

print(f"\n📈 Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# Vectorize text (convert words to numbers for AI)
print("\n🔢 Converting text to numbers (TF-IDF Vectorization)...")
# Increase features for larger dataset
max_features = 5000 if len(df) > 1000 else 1000
vectorizer = TfidfVectorizer(
    max_features=max_features,
    ngram_range=(1, 2),
    min_df=2,  # Word must appear in at least 2 documents
    max_df=0.95  # Ignore words that appear in more than 95% of documents
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"   Created {X_train_vec.shape[1]} features")

# Train the AI model
print("\n🤖 Training Logistic Regression Model...")
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    C=1.0,  # Regularization strength
    solver='lbfgs'
)
model.fit(X_train_vec, y_train)

# Test the model
print("\n✅ Testing model accuracy...")
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎯 MODEL PERFORMANCE:")
print(f"   Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Detailed Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# Show confusion matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   True Real correctly classified: {cm[0][0]}")
print(f"   True Real misclassified as Fake: {cm[0][1]}")
print(f"   True Fake misclassified as Real: {cm[1][0]}")
print(f"   True Fake correctly classified: {cm[1][1]}")

# Save the trained model
print("\n💾 Saving trained model and vectorizer...")
joblib.dump(model, 'models/truthguard_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

# Save metadata
metadata = {
    'total_samples': len(df),
    'accuracy': float(accuracy),
    'features': X_train_vec.shape[1],
    'fake_samples': len(fake_texts),
    'real_samples': len(true_texts)
}
joblib.dump(metadata, 'models/metadata.pkl')

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("📁 Files saved:")
print("   • models/truthguard_model.pkl")
print("   • models/vectorizer.pkl")
print("   • models/metadata.pkl")
print(f"\n📊 Model Stats:")
print(f"   • Trained on {len(df)} samples")
print(f"   • Accuracy: {accuracy * 100:.2f}%")
print(f"   • Features: {X_train_vec.shape[1]}")
print("\n🚀 You can now run: python app.py")
print("=" * 60)