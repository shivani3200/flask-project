# Import necessary libraries
from flask import Flask, render_template, request, url_for
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Create an instance of PorterStemmer for stemming
port_stem = PorterStemmer()

# Define a function for text preprocessing and stemming
def stemming(content):
    # Remove non-alphabetic characters and convert to lowercase
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    
    # Split the text into words
    stemmed_content = stemmed_content.split()
    
    # Stem each word and remove stopwords
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    
    # Join the stemmed words back into a single string
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content

# Load the trained model from a pickle file
with open('rf_classifier4.pkl', 'rb') as model_file:
    classifier = pickle.load(model_file)

# Load the TF-IDF vectorizer from a pickle file
with open('tfidf_vectorizer4.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the route for the home page
@app.route('/')
def home():
    # Render the home page template
    return render_template("index1.html")

# Define the route for sentiment analysis prediction
@app.route('/sentiment_analysis_prediction', methods=['POST'])
def sent_analysis_prediction():
    if request.method == 'POST':
        # Get the input text from the form
        comment = request.form['text']
        
        # Preprocess and stem the input text
        cleaned_comment = stemming(comment)
        
        # Vectorize the preprocessed text using the TF-IDF vectorizer
        comment_vector = vectorizer.transform([cleaned_comment])
        
        # Predict the sentiment using the trained classifier
        predicted_sentiment = classifier.predict(comment_vector)
       
        # Determine which emoji to display based on the predicted sentiment
        if predicted_sentiment == 1:
            sentiment_label = 'Positive'
            emoji_file = 'happy_emoji.gif'  # Filename for positive sentiment emoji
        elif predicted_sentiment == 0:
            sentiment_label = 'Negative'
            emoji_file = 'sad_smiley2.gif'  # Filename for negative sentiment emoji
        
        # Render the result template with the input text, predicted sentiment, and emoji
        return render_template('result4.html', text=comment, sentiment=sentiment_label, emoji=emoji_file)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
