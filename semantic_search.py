import os
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

QA_PATH = os.path.join(DATA_DIR, "kcc_qa_pairs.json")
EMB_PATH = os.path.join(DATA_DIR, "kcc_embeddings.pkl")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

# ----------------------------
# Load Q&A data
# ----------------------------
with open(QA_PATH, "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# Debug check (only once)
print("DEBUG FIRST QA PAIR:")
print(qa_pairs[0])
print()

# ----------------------------
# Load embeddings
# ----------------------------
with open(EMB_PATH, "rb") as f:
    embeddings = pickle.load(f)

# ----------------------------
# Load FAISS index
# ----------------------------
index = faiss.read_index(FAISS_PATH)

# ----------------------------
# Load SAME embedding model
# ----------------------------
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

# ----------------------------
# Search function
# ----------------------------
def search(query, top_k=3):
    # Convert query to embedding
    query_vector = model.encode([query])
    query_vector = np.array(query_vector).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_vector, top_k)

    # Collect unique answers
    answers = []
    for idx in indices[0]:
        answer = qa_pairs[idx]["answer"]
        if answer not in answers:
            answers.append(answer)

    return answers

# ----------------------------
# MAIN LOOP (IMPORTANT)
# ----------------------------
if __name__ == "__main__":
    print("Semantic Search Test Started\n")

    while True:
        query = input("Ask a question (type 'exit' to quit): ")

        if query.lower() == "exit":
            print("Exiting...")
            break

        results = search(query, top_k=1)

        print("\nRetrieved Answers:")
        for r in results:
            print("-", r)
        print()  