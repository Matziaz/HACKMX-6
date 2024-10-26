from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import google.generativeai as genai
import os
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv('creds.env')

API_KEY = os.getenv('API_KEY')
ACCOUNT_SID = os.getenv('ACCOUNT_SID')
AUTH_TOKEN = os.getenv('AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

print(API_KEY)
print(ACCOUNT_SID)
print(AUTH_TOKEN)
print(TWILIO_PHONE_NUMBER)

genai.configure(api_key=API_KEY)

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar el modelo entrenado y el vectorizador
try:
    model = joblib.load('multinomial_nb_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    raise FileNotFoundError("No se encontraron los archivos del modelo. Asegúrate de tener 'multinomial_nb_model.pkl' y 'tfidf_vectorizer.pkl' en el directorio.")

# Credenciales de Twilio (reemplaza con tus credenciales reales)

# Inicializar cliente de Twilio
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# Inicializar la aplicación Flask
app = Flask(__name__)

def clean_text(text):
    """Limpia el texto eliminando caracteres especiales y convirtiendo a minúsculas"""
    if not isinstance(text, str):
        return ""
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    return text.strip()

def remove_stopwords(text):
    """Elimina las palabras comunes (stopwords) del texto"""
    stop_words = set(stopwords.words("english"))
    return [word for word in text if word not in stop_words]

def lemmatize_word(text):
    """Convierte las palabras a su forma base"""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in text]


    

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predecir si un mensaje es spam
    
    Ejemplo de uso:
    curl -X POST http://localhost:5000/predict 
         -H "Content-Type: application/json" 
         -d '{"message": "Win free money now!", "phone_number": "+1234567890"}'
    """
    try:
        # Obtener datos del request
        data = request.get_json(force=True)
        
        # Validar datos requeridos
        if not all(key in data for key in ['message', 'phone_number']):
            return jsonify({'error': 'Se requiere mensaje y número de teléfono'}), 400
            
        message = data['message']
        phone_number = data['phone_number']

        # Preprocesar el mensaje
        message_cleaned = clean_text(message)
        message_tokenized = nltk.word_tokenize(message_cleaned)
        message_no_stopwords = remove_stopwords(message_tokenized)
        message_lemmatized = lemmatize_word(message_no_stopwords)
        message_preprocessed = ' '.join(message_lemmatized)

        # Vectorizar y predecir
        message_vectorized = tfidf.transform([message_preprocessed]).toarray()
        prediction = model.predict(message_vectorized)
        result = 'spam' if prediction[0] == 1 else 'ham'


         # Send SMS if the message is classified as spam
        if result == 'spam':
            gemini = genai.GenerativeModel("gemini-1.5-flash")
            response = gemini.generate_content(f"Why is this text '{message}' considered spam? Explain as friendly as possible to an elder adult. Also make it short (Less than 150 as it is intended to be sent as sms). No text formatting.")
            
            # Extract plain text using regex
            text_response = re.sub(r'<[^>]+>', '', response.text)
            
            # Remove any remaining unwanted characters
            text_response = text_response.replace('\n', ' ').replace('**', '').strip()
            
            try:
                sms = client.messages.create(
                    body=text_response,
                    from_=TWILIO_PHONE_NUMBER,
                    to=phone_number
                )
                print(f"Message sent successfully. SID: {sms.sid}")
            except Exception as e:
                print(f"Error sending SMS: {str(e)}")
                return jsonify({'error': f"Error sending SMS: {str(e)}"}), 500
        else:
            try:
                sms = client.messages.create(
                    body=message,
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
        
    # Run the Flask application
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
