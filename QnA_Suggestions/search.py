import pickle
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
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

def search(user_question, top_n=3, similarity_threshold=0.2):
    """
    Search the inverted index for the best matches to the user question.
    Returns both the question and the answer.
    """
    # Load the preprocessed dataset, inverted index, and TF-IDF vectorizer
    with open('Q&A_Suggestions\\preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        dataset = data['dataset']
        inverted_index = data['inverted_index']
        vectorizer = data['vectorizer']
        tfidf_matrix = data['tfidf_matrix']

    # Preprocess the user question
    user_tokens = preprocess_text(user_question)

    # Compute TF-IDF vector for the user question
    user_tfidf = vectorizer.transform([user_question])

    # Compute cosine similarity between the user question and all summaries
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Filter results based on the similarity threshold
    filtered_indices = [idx for idx, score in enumerate(similarities) if score >= similarity_threshold]

    # If no results meet the threshold, return a message
    if not filtered_indices:
        return [("No relevant question found.", "No relevant answer found.")]

    # Get top N indices with highest similarity
    top_indices = sorted(filtered_indices, key=lambda idx: similarities[idx], reverse=True)[:top_n]

    # Retrieve the corresponding questions and answers
    results = []
    for idx in top_indices:
        results.append((dataset.iloc[idx]['Summary'], dataset.iloc[idx]['Resolution']))

    return results

# Example usage
if __name__ == "__main__":
    user_question = "Unable to Login to PC"
    results = search(user_question, similarity_threshold=0.2)  # Adjust threshold as needed
    print("Search Results:")
    for i, (question, answer) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Question: {question}")
        print(f"Answer: {answer}")