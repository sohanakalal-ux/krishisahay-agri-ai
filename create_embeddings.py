import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔒 LOCK THE MODEL (DO NOT CHANGE LATER)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load Q&A pairs
with open("data/kcc_qa_pairs.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

# ✅ EMBED ONLY QUESTIONS
questions = [item["question"] for item in qa_pairs]

# Load model
model = SentenceTransformer(MODEL_NAME)

# Create embeddings
embeddings = model.encode(
    questions,
    convert_to_numpy=True,
    show_progress_bar=True
).astype("float32")

# ✅ NORMALIZE (CRITICAL)
faiss.normalize_L2(embeddings)

# Save embeddings
with open("data/kcc_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

# ✅ COSINE SIMILARITY INDEX
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "data/faiss_index.bin")

print("✅ FAISS index built correctly")