import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

print("Loading BAAI/bge-m3 with AutoTokenizer and AutoModel...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
model = AutoModel.from_pretrained("BAAI/bge-m3")
model.eval()
print(f"Loaded successfully in {time.time() - start:.2f}s!")

sentences = [
    "Patient presents with intermittent chest discomfort and shortness of breath on exertion.",
    "Findings are consistent with stable angina, likely related to underlying coronary artery disease.",
    "What is the diagnosis?",
    "What is Google?",
]

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=8192)
with torch.no_grad():
    model_output = model(**encoded_input)
    # BGE-M3 uses [CLS] representation (model_output[0][:, 0]) as dense embedding
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Dense embedding shape:", sentence_embeddings.shape)
emb = sentence_embeddings.cpu().numpy()
print("Dimensions:", len(emb[0]))

# Cosine similarities
import numpy as np
query_emb = emb[2]
doc1_emb = emb[0]
doc2_emb = emb[1]
irrelevant_query_emb = emb[3]

sim_diag_doc2 = float(np.dot(query_emb, doc2_emb))
sim_diag_doc1 = float(np.dot(query_emb, doc1_emb))
sim_irrelevant = float(np.dot(irrelevant_query_emb, doc2_emb))

print(f"Similarity ('What is the diagnosis?' -> Assessment): {sim_diag_doc2:.4f}")
print(f"Similarity ('What is the diagnosis?' -> Chief Complaint): {sim_diag_doc1:.4f}")
print(f"Similarity ('What is Google?' -> Assessment): {sim_irrelevant:.4f}")
