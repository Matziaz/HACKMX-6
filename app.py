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

        # Enviar SMS si es spam
        if result == 'spam':
            try:
                client.messages.create(
                    body="¡Alerta! Este mensaje parece sospechoso. Por tu seguridad, evita hacer clic en enlaces desconocidos.",
                    from_=TWILIO_PHONE_NUMBER,
                    to=phone_number
                )
            except Exception as e:
                print(f"Error al enviar SMS: {str(e)}")

        return jsonify({
            'prediction': result,
            'message': message,
            'preprocessed_message': message_preprocessed
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)