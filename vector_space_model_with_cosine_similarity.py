from collections import Counter
import math
import re

class VectorSpaceModel:
    def __init__(self):
        self.documents = {}
        self.vocabulary = set()
        self.document_vectors = {}
        self.term_freqs = []
    
    def tokenize(self, text):
        """Tokenize and normalize text."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    def compute_tf(self, tokens):
        """Compute Term Frequency."""
        tf = Counter(tokens)
        max_freq = max(tf.values()) if tf else 1
        return {term: freq / max_freq for term, freq in tf.items()}
    
    def compute_idf(self, doc_term_freqs):
        """Compute Inverse Document Frequency."""
        idf = {}
        num_docs = len(doc_term_freqs)
        for term in self.vocabulary:
            doc_count = sum(1 for doc in doc_term_freqs if term in doc)
            idf[term] = math.log(num_docs / (1 + doc_count)) + 1
        return idf
    
    def compute_tfidf_vector(self, tf, idf):
        """Compute TF-IDF vector for a document."""
        return {term: tf.get(term, 0) * idf.get(term, 0) for term in self.vocabulary}
    
    def add_document(self, doc_id, text):
        """Add document and compute its vector."""
        self.documents[doc_id] = text
        tokens = self.tokenize(text)
        self.vocabulary.update(tokens)
        
        self.term_freqs.append(Counter(tokens))
    
    def build_vectors(self):
        """Build TF-IDF vectors for all documents."""
        self.idf = self.compute_idf(self.term_freqs)
        self.document_vectors = {}
        
        for doc_id in self.documents.keys():
            tokens = self.tokenize(self.documents[doc_id])
            tf = self.compute_tf(tokens)
            self.document_vectors[doc_id] = self.compute_tfidf_vector(tf, self.idf)
    
    def cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors."""
        # Find common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        norm1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query):
        """Search for documents relevant to the query."""
        query_tokens = self.tokenize(query)
        query_tf = self.compute_tf(query_tokens)
        query_vector = self.compute_tfidf_vector(query_tf, self.idf)
        
        results = []
        for doc_id, doc_vector in self.document_vectors.items():
            similarity = self.cosine_similarity(query_vector, doc_vector)
            results.append((doc_id, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Example usage
vsm = VectorSpaceModel()
vsm.add_document(1, "The quick brown fox jumps over the lazy dog")
vsm.add_document(2, "A quick brown fox runs fast")
vsm.add_document(3, "The lazy dog sleeps all day")
vsm.build_vectors()

print("Search results for 'quick fox':", vsm.search("quick fox"))
print("Search results for 'lazy dog':", vsm.search("lazy dog"))