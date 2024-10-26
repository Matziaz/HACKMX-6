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
TWILIO_PHONE_NUMBER = ''  # Ensure to include the country code in this number

# Initialize Twilio client
try:
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
except Exception as e:
    raise Exception("Error initializing Twilio client: " + str(e))

# Initialize Flask application
app = Flask(__name__)

def clean_text(text):
    """Cleans the text by removing special characters and converting to lowercase."""
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text.strip()

def remove_stopwords(text):
    """Removes common words (stopwords) from the text."""
    stop_words = set(stopwords.words("english"))  # Load English stopwords
    return [word for word in text if word not in stop_words]  # Filter out stopwords

def lemmatize_word(text):
    """Lemmatizes the words to their base form."""
    lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer
    return [lemmatizer.lemmatize(word, pos='v') for word in text]  # Lemmatize verbs

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict whether a message is spam.
    
    Example usage:
    curl -X POST http://localhost:5000/predict 
         -H "Content-Type: application/json" 
         -d '{"message": "Win free money now!", "phone_number": "+1234567890"}'
    """
    try:
        # Get data from the request
        data = request.get_json(force=True)
        
        # Validate required data
        if not all(key in data for key in ['message', 'phone_number']):
            return jsonify({'error': 'Message and phone number are required'}), 400
            
        message = data['message']
        phone_number = data['phone_number']

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

        # Send SMS if the message is classified as spam
        if result == 'spam':
            try:
                sms = client.messages.create(
                    body="Alert! This message seems suspicious. For your safety, avoid clicking on unknown links.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=phone_number
                )
                print(f"Message sent successfully. SID: {sms.sid}")
            except Exception as e:
                print(f"Error sending SMS: {str(e)}")
                return jsonify({'error': f"Error sending SMS: {str(e)}"}), 500

        return jsonify({
            'prediction': result,
            'message': message,
            'preprocessed_message': message_preprocessed
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

