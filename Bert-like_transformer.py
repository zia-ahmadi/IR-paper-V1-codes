import numpy as np
import math
import re

class SimplifiedAttention:
    """Simplified implementation of attention mechanism used in BERT."""
    
    def __init__(self, embedding_dim=64):
        self.embedding_dim = embedding_dim
        self.attention_weights = None
    
    def scaled_dot_product_attention(self, query, key, value):
        """Compute scaled dot-product attention."""
        # query, key, value are matrices of shape (seq_len, embedding_dim)
        # Compute attention scores
        scores = np.matmul(query, key.T) / math.sqrt(self.embedding_dim)
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        output = np.matmul(self.attention_weights, value)
        return output
    
    def encode_query(self, query_tokens, context_tokens):
        """Encode query with context attention."""
        # Simplified: Create random embeddings for demonstration
        np.random.seed(42)
        query_emb = np.random.randn(len(query_tokens), self.embedding_dim)
        context_emb = np.random.randn(len(context_tokens), self.embedding_dim)
        
        # Apply attention
        attended = self.scaled_dot_product_attention(query_emb, context_emb, context_emb)
        return attended

class BertSimplified:
    """Simplified BERT-like model for semantic understanding."""
    
    def __init__(self):
        self.attention = SimplifiedAttention()
        self.embedding_dim = 64
        self.vocab = {}
        
    def tokenize(self, text):
        """Simple tokenizer."""
        return re.findall(r'\b[a-z0-9]+\b', text.lower())
    
    def compute_relevance_score(self, query, document):
        """Compute semantic relevance score using attention."""
        query_tokens = self.tokenize(query)
        doc_tokens = self.tokenize(document)
        
        if not query_tokens or not doc_tokens:
            return 0.0
        
        # Get attended representations
        attended = self.attention.encode_query(query_tokens, doc_tokens)
        
        # Compute relevance as average attention-weighted similarity
        relevance = np.mean(attended)
        
        # Normalize to [0, 1] range
        return np.clip(relevance, 0, 1) * 10
    
    def rank_documents(self, query, documents):
        """Rank documents by semantic relevance."""
        results = []
        for doc_id, doc_text in documents:
            score = self.compute_relevance_score(query, doc_text)
            results.append((doc_id, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results

# Example usage
bert_simple = BertSimplified()

test_documents = [
    (1, "The quick brown fox jumps over the lazy dog"),
    (2, "A fast fox runs through the forest"),
    (3, "The lazy dog sleeps all day")
]

query = "fast animal with quick reflexes"
ranked = bert_simple.rank_documents(query, test_documents)
print(f"Semantic relevance for '{query}':", ranked)