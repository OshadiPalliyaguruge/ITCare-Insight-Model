import os
import glob
import pickle
from collections import defaultdict
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

def get_latest_preprocessed_file():
    """
    Find the most recent preprocessed_data_*.pkl file.
    """
    # List all files matching the pattern
    files = glob.glob('QnA_Suggestions\\preprocessed_data_*.pkl')
    if not files:
        raise FileNotFoundError("No preprocessed data files found.")

    # Get the most recent file based on the timestamp in the filename
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def search(user_question, top_n=3, exact_match_threshold=0.8):
    """
    Search the inverted index for the best matches to the user question.
    Returns both the question and the answer.
    """
    # Load the most recent preprocessed data file
    preprocessed_file = get_latest_preprocessed_file()
    with open(preprocessed_file, 'rb') as f:
        data = pickle.load(f)
        dataset = data['dataset']
        inverted_index = data['inverted_index']

    # Preprocess the user question
    user_tokens = preprocess_text(user_question)

    # Find matching documents
    matches = defaultdict(int)
    for token in user_tokens:
        if token in inverted_index:
            for doc_id, summary, resolution in inverted_index[token]:
                matches[(doc_id, summary, resolution)] += 1

    # Sort matches by frequency
    sorted_matches = sorted(matches.items(), key=lambda x: x[1], reverse=True)

    # Filter results based on exact match threshold
    filtered_results = []
    for (doc_id, summary, resolution), count in sorted_matches:
        # Calculate the percentage of tokens that match
        match_percentage = count / len(user_tokens)
        if match_percentage >= exact_match_threshold:
            filtered_results.append((summary, resolution))

    # Return top N results
    results = filtered_results[:top_n]
    return results if results else [("No relevant question found.", "No relevant answer found.")]

# Example usage
if __name__ == "__main__":
    user_question = "btp"
    results = search(user_question, exact_match_threshold=0.8)  # Adjust threshold as needed
    print("Search Results:")
    for i, (question, answer) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Question: {question}")
        print(f"Answer: {answer}")

