import os
import sys
import time

# Set HF mirror if needed for fast reliable download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

print("Attempting to load/download BAAI/bge-m3 via mirror/hub...")
start = time.time()
try:
    model = SentenceTransformer("BAAI/bge-m3")
    print(f"Loaded successfully in {time.time() - start:.2f}s!")
    emb = model.encode(["Medical diagnosis test"], normalize_embeddings=True)
    print(f"Embedding shape: {emb.shape}")
except Exception as e:
    print(f"Error loading model: {e}")
