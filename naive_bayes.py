# Importing all the libraries to be used
import warnings
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
import joblib  # Add this line at the beginning of the script

nltk.download('stopwords')
nltk.download('WordNetLemmatizer')
nltk.download('punkt')
nltk.download('wordnet')

# Loading data
data = pd.read_csv("spam.csv", encoding='ISO-8859-1')

# Dropping the redundant looking columns (for this project)
to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
data = data.drop(columns=to_drop)

# Renaming the columns because I feel fancy today 
data.rename(columns={"v1": "Target", "v2": "Text"}, inplace=True)

# Adding a column of numbers of characters, words, and sentences in each message
data["No_of_Characters"] = data["Text"].apply(len)
data["No_of_Words"] = data.apply(lambda row: nltk.word_tokenize(row["Text"]), axis=1).apply(len)
data["No_of_Sentence"] = data.apply(lambda row: nltk.sent_tokenize(row["Text"]), axis=1).apply(len)

# Defining a function to clean up the text
def Clean(Text):
    sms = re.sub('[^a-zA-Z]', ' ', Text)  # Replacing all non-alphabetic characters with a space
    sms = sms.lower()  # Converting to lowercase
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

data["Clean_Text"] = data["Text"].apply(Clean)

data["Tokenize_Text"] = data.apply(lambda row: nltk.word_tokenize(row["Clean_Text"]), axis=1)

# Removing the stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

data["Nostopword_Text"] = data["Tokenize_Text"].apply(remove_stopwords)

lemmatizer = WordNetLemmatizer()

# Lemmatize string
def lemmatize_word(text):
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in text]
    return lemmas

data["Lemmatized_Text"] = data["Nostopword_Text"].apply(lemmatize_word)

# Creating a corpus of text feature to encode further into vectorized form
corpus = [' '.join(row) for row in data["Lemmatized_Text"]]

# Changing text data into numbers
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()

# Label encode the Target and use it as y
label_encoder = LabelEncoder()
data["Target"] = label_encoder.fit_transform(data["Target"])

# Setting values for labels and feature as y and X
y = data["Target"]

# Splitting the testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Testing on the following classifiers
model = MultinomialNB().fit(X_train, y_train)

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "SVC"}

cv_score = cross_val_score(model, X_train, y_train, scoring="accuracy", cv=10)
print("%s: %f " % (pipe_dict[0], cv_score.mean()))

# Model Evaluation
# Creating lists of various scores
precision = []
recall = []
f1_scores = []
trainset_accuracy = []
testset_accuracy = []

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
prec = metrics.precision_score(y_test, pred_test)
recal = metrics.recall_score(y_test, pred_test)
f1_s = metrics.f1_score(y_test, pred_test)
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

# Appending scores
precision.append(prec)
recall.append(recal)
f1_scores.append(f1_s)
trainset_accuracy.append(train_accuracy)
testset_accuracy.append(test_accuracy)

# Initialize data of lists
data = {
    'Precision': precision,
    'Recall': recall,
    'F1score': f1_scores,
    'Accuracy on Testset': testset_accuracy,
    'Accuracy on Trainset': trainset_accuracy
}

# Create pandas DataFrame
Results = pd.DataFrame(data, index=["SVC"])

# Print accuracy
print("Train Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

# Save the model and the vectorizer
joblib.dump(model, 'multinomial_nb_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Model and vectorizer saved successfully.")

