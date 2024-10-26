from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load('multinomial_nb_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask application
app = Flask(__name__)

# Define a function to clean up the text
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Replacing all non-alphabetic characters with a space
    text = text.lower()  # Converting to lowercase
    text = text.split()
    return ' '.join(text)

# Removing the stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

lemmatizer = WordNetLemmatizer()

# Lemmatize string
def lemmatize_word(text):
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in text]
    return lemmas

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)

    # Extract the text message from the request
    message = data['message']

    # Apply the same preprocessing steps
    message_cleaned = clean_text(message)
    message_tokenized = nltk.word_tokenize(message_cleaned)
    message_no_stopwords = remove_stopwords(message_tokenized)
    message_lemmatized = lemmatize_word(message_no_stopwords)
    message_preprocessed = ' '.join(message_lemmatized)

    # Vectorize the message using the loaded vectorizer
    message_vectorized = tfidf.transform([message_preprocessed]).toarray()

    # Make a prediction using the loaded model
    prediction = model.predict(message_vectorized)

    # Map the prediction to the original labels (0: ham, 1: spam)
    result = 'spam' if prediction[0] == 1 else 'ham'

    # Return the prediction as JSON
    return jsonify({'prediction': result})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)