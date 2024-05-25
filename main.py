import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from llama_cpp import Llama

# Start the timer
start_time = time.time()

# Load the model
model_path = "bge-small-en-1.5-Q_4_K_M.gguf"
model = Llama(model_path, embedding=True)

# Input text
post = """Artificial intelligence (AI) has emerged as a transformative technology, revolutionizing various sectors 
and redefining how we interact with the digital world. At its core, AI refers to the simulation of human intelligence 
in machines programmed to think, learn, and adapt. This technology encompasses a wide range of capabilities, 
including problem-solving, understanding natural language, recognizing patterns, and making decisions."""

# Get embedding for the entire passage
passage_embedding = model.embed(post)

# Tokenize the passage into words (case insensitive)
words = [word.lower() for word in post.split()]

# Remove punctuation from the words
import string
words = [''.join(char for char in word if char not in string.punctuation) for word in words]
words = list(filter(None, words))  # Remove empty strings

# Get embeddings for each word
word_embeddings = {word: model.embed(word) for word in words}

# Calculate cosine similarity and dot product for each word
cosine_similarities = {}
dot_products = {}

for word, embedding in word_embeddings.items():
    cosine_similarities[word] = cosine_similarity([passage_embedding], [embedding])[0][0]
    dot_products[word] = np.dot(passage_embedding, embedding)

# Get top 5 words by cosine similarity
top_5_cosine = sorted(cosine_similarities, key=cosine_similarities.get, reverse=True)[:5]

# Get top 5 words by dot product
top_5_dot = sorted(dot_products, key=dot_products.get, reverse=True)[:5]

# End the timer
end_time = time.time()

# Print results
print("Top 5 words by cosine similarity:", top_5_cosine)
print("Top 5 words by dot product:", top_5_dot)

# Print the time taken to run the program
print(f"Time taken to run the program: {end_time - start_time:.2f} seconds")
