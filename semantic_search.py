import numpy as np
from collections import Counter
import math
import re


class WordEmbeddingSearch:
    def __init__(self):
        self.word_vectors = {}
        self.document_vectors = {}
        self.documents = {}
        self.vocabulary_size = 100  # For simplified demo
        
    def build_simple_embeddings(self, corpus):
        """Build simple word embeddings using co-occurrence (simplified)."""
        # Build co-occurrence matrix
        co_occurrence = Counter()
        
        for doc in corpus:
            tokens = re.findall(r'\b[a-z0-9]+\b', doc.lower())
            for i, word in enumerate(tokens):
                # Look at neighboring words
                for j in range(max(0, i-2), min(len(tokens), i+3)):
                    if i != j:
                        co_occurrence[(word, tokens[j])] += 1
        
        # Create word vectors
        all_words = set()
        for word, _ in co_occurrence:
            all_words.add(word)
        
        # Create one-hot like vectors based on co-occurrence
        for word in all_words:
            vector = np.zeros(min(len(all_words), self.vocabulary_size))
            # Add co-occurrence based features
            for j, other in enumerate(list(all_words)[:self.vocabulary_size]):
                vector[j] = co_occurrence.get((word, other), 0)
            
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            self.word_vectors[word] = vector
    
    def document_vector(self, text):
        """Compute document vector as average of word vectors."""
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        vectors = [self.word_vectors.get(token, np.zeros(len(self.word_vectors.get(list(self.word_vectors.keys())[0], [])))) 
                   for token in tokens]
        
        if vectors:
            return np.mean(vectors, axis=0)
        return np.zeros(self.vocabulary_size)
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def add_document(self, doc_id, text):
        """Add document to the collection."""
        self.documents[doc_id] = text
    
    def build_index(self):
        """Build document vectors for all documents."""
        for doc_id, text in self.documents.items():
            self.document_vectors[doc_id] = self.document_vector(text)
    
    def search(self, query):
        """Search for documents semantically related to the query."""
        query_vector = self.document_vector(query)
        results = []
        
        for doc_id, doc_vec in self.document_vectors.items():
            similarity = self.cosine_similarity(query_vector, doc_vec)
            results.append((doc_id, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A fast fox runs through the forest",
    "The lazy dog sleeps peacefully",
    "Cats and dogs are popular pets",
    "Wild animals live in the forest",
    "The agile fox catches the rabbit"
]

embed_search = WordEmbeddingSearch()

# Build embeddings and add documents
embed_search.build_simple_embeddings(corpus)
for i, doc in enumerate(corpus):
    embed_search.add_document(i, doc)
embed_search.build_index()

print("Semantic search for 'fast animal':", embed_search.search("fast animal"))
print("Semantic search for 'canine pet':", embed_search.search("canine pet"))