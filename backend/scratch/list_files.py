import os
from huggingface_hub import list_repo_files

files = list_repo_files("BAAI/bge-m3")
print("Files in BAAI/bge-m3:")
for f in files:
    print(" -", f)
