from flask import Flask, request, jsonify
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd


app = Flask(__name__)

def load_model(filename):
    try:
        # Load the model from the file
        loaded_model = pickle.load(open(filename, "rb"))
        return loaded_model
    except:
        print("Error loading the model")
        return None
    
# Initialize the PorterStemmer
ps = PorterStemmer()
# Get Arabic stopwords
sw = stopwords.words('arabic')

def clean(text):
    text = str(text)
    text = re.sub('[^ء-ي0-9 ]+', '', text).strip()  ## arabic
    filtered_words = []
    for word in text.lower().split():   ## arabic 
        if word not in sw:
            filtered_words.append(ps.stem(word))
    return ' '.join(filtered_words)

def predict_label_from_user_input(clf, tfidf_vectorizer,user_input):
    cleaned_text = clean(user_input)  # You need to define your clean() function
    # Transform the cleaned text using the fitted vectorizer
    cleaned_text = count_vect.transform([cleaned_text])
    text_vectorized = tfidf_vectorizer.transform(cleaned_text)
    
    # Predict using the trained classifier
    predicted_label = clf.predict(text_vectorized)
    
    return predicted_label[0]


# Example usage:
model_filename = 'clf.sav'
tfidf_filename = 'tfidf_transformer.sav'
count_vect_filename = 'count_vect.sav'

model = load_model(model_filename)
tfidf_vectorizer = load_model(tfidf_filename)
count_vect = load_model(count_vect_filename)

# Function to predict class
def predict_text(user_input):
    global model, tfidf_vectorizer
    predicted = predict_label_from_user_input(model, tfidf_vectorizer,user_input)
    return predicted

# Flask route
@app.route('/predict', methods=['POST','GET'])
def predict():
    data = request.get_json()
    text = data['text']
    predicted = predict_text(text)
    return jsonify({'predicted': predicted})

if __name__ == '__main__':
    app.run(debug=True)
