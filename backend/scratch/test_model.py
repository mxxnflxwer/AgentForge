import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import time
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading BAAI/bge-m3...")
start = time.time()
model = SentenceTransformer("BAAI/bge-m3")
load_time = time.time() - start
print(f"Model loaded in {load_time:.2f} seconds.")

test_texts = [
    "Patient presents with intermittent chest discomfort and shortness of breath on exertion.",
    "Findings are consistent with stable angina, likely related to underlying coronary artery disease.",
    "What is the diagnosis?",
    "What is Google?",
]

start = time.time()
embeddings = model.encode(test_texts, normalize_embeddings=True)
infer_time = time.time() - start
print(f"Encoded {len(test_texts)} texts in {infer_time:.4f}s.")
print(f"Embedding shape: {embeddings.shape}")
print(f"Dimensions: {len(embeddings[0])}")

query_emb = embeddings[2]
doc1_emb = embeddings[0]
doc2_emb = embeddings[1]
irrelevant_query_emb = embeddings[3]

sim_diag_doc2 = float(np.dot(query_emb, doc2_emb))
sim_diag_doc1 = float(np.dot(query_emb, doc1_emb))
sim_irrelevant = float(np.dot(irrelevant_query_emb, doc2_emb))

print(f"Similarity ('What is the diagnosis?' -> Assessment): {sim_diag_doc2:.4f}")
print(f"Similarity ('What is the diagnosis?' -> Chief Complaint): {sim_diag_doc1:.4f}")
print(f"Similarity ('What is Google?' -> Assessment): {sim_irrelevant:.4f}")
