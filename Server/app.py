from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from twilio.rest import Client

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
try:
    model = joblib.load('multinomial_nb_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Model files not found. Ensure 'multinomial_nb_model.pkl' and 'tfidf_vectorizer.pkl' are in the directory.")

# Twilio credentials (replace with your actual credentials)
ACCOUNT_SID = ''
AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''

# Initialize Twilio client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Initialize the Flask application
app = Flask(__name__)

# Initialize a global counter for spam messages
spam_counter = 0
SPAM_LIMIT = 100  # Limit for spam messages

def clean_text(text):
    """Cleans text by removing special characters and converting to lowercase"""
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text.strip()

def remove_stopwords(text):
    """Removes common words (stopwords) from the text"""
    stop_words = set(stopwords.words("english"))
    return [word for word in text if word not in stop_words]

def lemmatize_word(text):
    """Converts words to their base form"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in text]

def send_message(to, message):
    """Sends a message using Twilio"""
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=to
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict if a message is spam or ham
    """
    global spam_counter  # Use the global counter
    # Get data from the request
    data = request.get_json(force=True)
    phone_number = data.get('phone_number')
    message = data.get('message')
        
    # Preprocess the message
    message_cleaned = clean_text(message)
    message_tokenized = nltk.word_tokenize(message_cleaned)
    message_no_stopwords = remove_stopwords(message_tokenized)
    message_lemmatized = lemmatize_word(message_no_stopwords)
    message_preprocessed = ' '.join(message_lemmatized)

    # Vectorize and predict
    message_vectorized = tfidf.transform([message_preprocessed]).toarray()
    prediction = model.predict(message_vectorized)
    result = 'spam' if prediction[0] == 1 else 'ham'

    # Increment the spam counter if spam is detected
    if result == 'spam':
        spam_counter += 1

    # Check if the spam counter exceeds the limit
    if spam_counter > SPAM_LIMIT:
        return jsonify({
            "prediction": result,
            "status": "spam limit exceeded, no alert sent"
        })

    # Send notification based on classification
    if result == 'spam':
        alert_message = "Alert: A spam message has been detected. Avoid clicking on suspicious links."
    else:
        alert_message = "This message has been classified as safe."

    # Send the appropriate message
    send_message(phone_number, alert_message)
    
    # JSON response with the prediction and message status
    return jsonify({
        "prediction": result,
        "status": "alert sent"
    })

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
