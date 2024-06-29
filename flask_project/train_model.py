# Import necessary libraries
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import warnings
from sklearn.exceptions import InconsistentVersionWarning

import ssl

# Bypass SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK data (stopwords)
nltk.download('stopwords')

# Ignore InconsistentVersionWarning from scikit-learn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)


# Load the Sentiment140 dataset
df = pd.read_csv('training.1600000.processed.noemoticon.csv', 
                 encoding='latin', 
                 header=None, 
                 usecols=[0, 5],
                 names=['target', 'text'])

# Map target to binary labels: 0 -> negative, 4 -> positive
df['target'] = df['target'].map({0: 0, 4: 1})

# Data cleaning function
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove @mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_text = ' '.join([word for word in words if word not in stop_words])
    return cleaned_text

# Apply data cleaning to 'text' column and create 'cleaned_text' column
df['cleaned_text'] = df['text'].apply(clean_text)

# Prepare features (X) and labels (y)
X = df['cleaned_text']
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the Random Forest classifier
try:
    # Initialize and train the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train_vectorized, y_train)
    
    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test_vectorized)
    
    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Save the trained model and vectorizer using joblib
    pickle.dump(rf_classifier, 'rf_classifier.pkl')
    pickle.dump(vectorizer, 'vectorizer.pkl')

    # # Attempt to reload and re-save the model to update its version
    # try:
    #     # Load the model again
    #     loaded_model = joblib.load('rf_classifier.pkl')
        
    #     # Re-save the loaded model
    #     joblib.dump(loaded_model, 'rf_classifier.pkl')

    # except Exception as e:
    #     print(f"Exception occurred while updating model: {e}")

except Exception as e:
    # Handle any exceptions that occur during training or prediction
    print(f"Exception occurred: {e}")

