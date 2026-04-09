from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "The movie had stunning visuals and great cinematography.",
    "The storyline was boring and lacked depth.",
    "The acting performances were outstanding and emotional.",
    "The film had poor direction and weak screenplay.",
    "Background music and editing were impressive."
]

embeddings = model.encode(documents)
embeddings = np.array(embeddings)
print(" Embeddings created\n")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(" Vector DB created\n")

print("🔹 Similarity Between Documents:\n")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = util.cos_sim(embeddings[i], embeddings[j])
        print(f"Doc {i+1} & Doc {j+1}: {sim.item():.4f}")

def search(query, k=2):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    print("\n Top Results:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {documents[idx]} (Distance: {distances[0][i]:.4f})")

def add_document(new_doc):
    global documents, index
    new_embedding = model.encode([new_doc])
    index.add(np.array(new_embedding))
    documents.append(new_doc)
    print("\n New document added successfully!")

while True:
    print("\nOptions:")
    print("1. Search")
    print("2. Add Document")
    print("3. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        query = input("Enter your query: ")
        search(query)

    elif choice == "2":
        new_doc = input("Enter new movie review: ")
        add_document(new_doc)

    elif choice == "3":
        print("Exiting...")
        break

    else:
        print("Invalid choice!")