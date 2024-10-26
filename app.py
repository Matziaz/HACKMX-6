from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from twilio.rest import Client

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
ACCOUNT_SID = ''
AUTH_TOKEN = ''
TWILIO_PHONE_NUMBER = ''

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

def send_message(to, message):
    """Envía un mensaje usando Twilio"""
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=to
    )

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para predecir si un mensaje es spam
    
    Ejemplo de uso:
    curl -X POST http://localhost:5000/predict 
         -H "Content-Type: application/json" 
         -d '{"message": "Win free money now!", "phone_number": "+1234567890"}'
    """
    # Obtener datos del request
    data = request.get_json(force=True)
    phone_number = data.get('phone_number')
    message = data.get('message')
        
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

    # Enviar alerta si es spam, de lo contrario enviar el mensaje original
    if result == 'spam':
        alert_message = "Alerta: Se ha detectado un mensaje de spam."
        send_message(phone_number, alert_message)
        return jsonify({"status": "alert sent"})
    else:
        send_message(phone_number, message)
        return jsonify({"status": "message sent"})

if __name__ == '__main__':
    app.run(debug=True)