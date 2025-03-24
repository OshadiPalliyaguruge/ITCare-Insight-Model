import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
import pickle
from datetime import datetime

# Download NLTK resources (run once)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # Download the required resource
  
# Generate a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS

# Load dataset
dataset = pd.read_csv('Dataset\\incident_report_q&a_cleaned.csv')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess text by tokenizing, removing stopwords, punctuation, and stemming.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

# Preprocess the dataset
dataset['Summary_tokens'] = dataset['Summary'].apply(preprocess_text)
dataset['Resolution_tokens'] = dataset['Resolution'].apply(preprocess_text)

# Create an inverted index
inverted_index = defaultdict(list)

for idx, row in dataset.iterrows():
    summary_tokens = row['Summary_tokens']
    summary = row['Summary']  # Store the original question
    resolution = row['Resolution']  # Store the corresponding answer
    for token in summary_tokens:
        inverted_index[token].append((idx, summary, resolution))  # Include both question and answer

# Save the preprocessed dataset and inverted index to disk
with open(f'QnA_Suggestions\\preprocessed_data_{timestamp}.pkl', 'wb') as f:
    pickle.dump({
        'dataset': dataset,
        'inverted_index': inverted_index
    }, f)

print("Preprocessing and indexing complete. Data saved to 'preprocessed_data.pkl'.")