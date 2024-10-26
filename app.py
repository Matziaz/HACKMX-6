from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('multinomial_nb_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Initialize the Flask application
app = Flask(__name__)

# Define a route for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json(force=True)

    # Extract the text message from the request
    message = data['message']

    # Vectorize the message using the loaded vectorizer
    message_vectorized = tfidf.transform([message]).toarray()

    # Make a prediction using the loaded model
    prediction = model.predict(message_vectorized)

    # Map the prediction to the original labels (0: ham, 1: spam)
    result = 'spam' if prediction[0] == 1 else 'ham'

    # Return the prediction as JSON
    return jsonify({'prediction': result})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)

