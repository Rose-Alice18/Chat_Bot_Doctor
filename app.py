from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
import os

# Initialize the Flask application
app = Flask(__name__)

# Load embeddings and dataset
try:
    with open('embeddings.pkl', 'rb') as f:
        embeddings = pickle.load(f)
    dataset = pd.read_csv('dataset_with_embeddings.csv')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    embeddings = []
    dataset = pd.DataFrame()

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def preprocess_text(text):
    """Preprocess the text by lowering and stripping."""
    return text.lower().strip()

def cosine_similarity_custom(vec1, vecs2):
    """Calculate the cosine similarity between one vector and a matrix of vectors."""
    vec1 = np.array(vec1)
    vecs2 = np.array(vecs2)
    dot_product = np.dot(vecs2, vec1.T)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vecs2 = np.linalg.norm(vecs2, axis=1)
    similarity = dot_product / (norm_vec1 * norm_vecs2)
    return similarity

def get_response(question, dataset, embeddings, threshold=0.4):
    """Find the most similar context and its response based on the given question."""
    preprocessed_question = preprocess_text(question)
    question_embedding = model.encode([preprocessed_question])[0]  # Ensure it's a 1D array
    if len(embeddings) == 0:
        return "I do not have knowledge about this question."
    similarities = cosine_similarity_custom(question_embedding, np.array(embeddings).reshape(-1, 384))
    max_similarity = np.max(similarities)

    if max_similarity < threshold:
        return "I do not have knowledge about this question."
    else:
        most_similar_index = np.argmax(similarities)
        return dataset.iloc[most_similar_index]['Response']

@app.route('/query', methods=['POST'])
def query():
    """Endpoint to get a response for a given question."""
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({'error': 'No question provided.'}), 400
    response = get_response(question, dataset, embeddings)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
