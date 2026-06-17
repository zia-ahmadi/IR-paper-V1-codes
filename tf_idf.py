from collections import Counter
import math
import re


class TFIDFCalculator:
    def __init__(self):
        self.documents = []
        self.vocabulary = set()
        self.doc_term_freq = []
        self.idf_values = {}

    def add_document(self, text):
        """Add a document to the collection."""
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        self.documents.append(tokens)
        self.vocabulary.update(tokens)
        self.doc_term_freq.append(Counter(tokens))

    def compute_tf(self, term, doc_index):
        """Compute Term Frequency."""
        doc_counter = self.doc_term_freq[doc_index]
        total_terms = sum(doc_counter.values())

        if total_terms == 0:
            return 0.0

        return doc_counter.get(term, 0) / total_terms

    def compute_idf(self, term):
        """Compute Inverse Document Frequency."""
        num_docs = len(self.documents)

        if num_docs == 0:
            return 0.0

        doc_count = sum(1 for doc in self.doc_term_freq if term in doc)

        if doc_count == 0:
            return 0.0

        return math.log(num_docs / doc_count)

    def compute_all_idf(self):
        """Compute IDF for all terms in vocabulary."""
        self.idf_values = {
            term: self.compute_idf(term)
            for term in self.vocabulary
        }

    def compute_tfidf(self, term, doc_index):
        """Compute TF-IDF score for a term in a document."""
        if not self.idf_values:
            self.compute_all_idf()

        return self.compute_tf(term, doc_index) * self.idf_values.get(term, 0)

    def get_document_vector(self, doc_index):
        """Get TF-IDF vector for a document."""
        if not self.idf_values:
            self.compute_all_idf()

        vector = {}

        for term in set(self.documents[doc_index]):
            vector[term] = self.compute_tfidf(term, doc_index)

        return vector

    def get_query_vector(self, query):
        """Get TF-IDF vector for a query."""
        if not self.idf_values:
            self.compute_all_idf()

        tokens = re.findall(r'\b[a-z0-9]+\b', query.lower())

        if not tokens:
            return {}

        tf = Counter(tokens)
        total_terms = len(tokens)

        vector = {}

        for term in set(tokens):
            vector[term] = (
                tf[term] / total_terms
            ) * self.idf_values.get(term, 0)

        return vector


# Example usage
tfidf = TFIDFCalculator()

tfidf.add_document("The quick brown fox jumps over the lazy dog")
tfidf.add_document("A quick brown fox runs fast")
tfidf.add_document("The lazy dog sleeps all day")

tfidf.compute_all_idf()

print("TF-IDF for 'fox' in document 1:",
      tfidf.compute_tfidf("fox", 0))

print("TF-IDF for 'dog' in document 2:",
      tfidf.compute_tfidf("dog", 1))

print("Document 1 vector:")
print(tfidf.get_document_vector(0))

print("Query vector:")
print(tfidf.get_query_vector("quick fox"))