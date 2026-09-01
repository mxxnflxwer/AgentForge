import os
import sys
import requests
import time

url = "https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
os.makedirs(cache_dir, exist_ok=True)
dest_path = os.path.join(cache_dir, "pytorch_model.bin")

headers = {"User-Agent": "Mozilla/5.0"}
downloaded = 0
if os.path.exists(dest_path):
    downloaded = os.path.getsize(dest_path)

if downloaded > 0:
    headers["Range"] = f"bytes={downloaded}-"
    print(f"Resuming from {downloaded / (1024*1024):.2f} MB...")

print(f"Downloading {url} -> {dest_path}...")
mode = "ab" if downloaded > 0 else "wb"

resp = requests.get(url, headers=headers, stream=True, timeout=30)
if resp.status_code in (200, 206):
    total_size = int(resp.headers.get('content-length', 0)) + downloaded
    start = time.time()
    last_print = start
    with open(dest_path, mode) as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024 * 4): # 4MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if time.time() - last_print > 3:
                    pct = (downloaded / total_size * 100) if total_size else 0
                    mb = downloaded / (1024 * 1024)
                    speed = (len(chunk) / (time.time() - last_print)) / (1024 * 1024)
                    print(f"Downloaded: {mb:.1f} MB / {total_size/(1024*1024):.1f} MB ({pct:.1f}%)")
                    last_print = time.time()
    print("Download completed successfully!")
else:
    print(f"Failed with status: {resp.status_code}")
