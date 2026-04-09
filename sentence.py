from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = [
    "Buy a new smartphone with good camera",
    "Affordable laptop for students",
    "Best places to visit during summer vacation",
    "Learn Python programming for beginners",
    "Online courses for data science",
    "Cheap hotels near beach locations"
]
embeddings = model.encode(sentences)
for i, emb in enumerate(embeddings):
    print(f"\nSentence: {sentences[i]}")
    print(f"Vector (first 5 values): {emb[:5]}")
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

query = "budget laptop for college students"
query_vec = model.encode([query])[0]

print("\n Search Results:")

for i, emb in enumerate(embeddings):
    score = cosine_similarity(query_vec, emb)
    print(f"{sentences[i]} → Similarity: {round(score, 3)}")