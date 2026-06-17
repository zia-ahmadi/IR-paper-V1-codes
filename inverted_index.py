class InvertedIndex:
    def __init__(self):
        self.index = {}  # word -> list of doc IDs
    
    def add_document(self, doc_id, text):
        words = text.lower().split()  # simple split, no punctuation handling
        for word in words:
            if word not in self.index:
                self.index[word] = []
            if doc_id not in self.index[word]:  # avoid duplicates
                self.index[word].append(doc_id)
    
    def search(self, query):
        words = query.lower().split()
        if not words:
            return []
        
        result = set(self.index.get(words[0], []))
        for word in words[1:]:
            result = result.intersection(self.index.get(word, []))
        
        return list(result)

# Example usage
index = InvertedIndex()
index.add_document(1, "the quick brown fox")
index.add_document(2, "quick brown fox")
index.add_document(3, "lazy dog")

print(index.search("quick fox"))  # [1, 2]
print(index.search("lazy dog"))   # [3]