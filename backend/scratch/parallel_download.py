import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
DEST_FILE = os.path.join(CACHE_DIR, "pytorch_model.bin")
NUM_THREADS = 16

head = requests.head(URL, allow_redirects=True, timeout=15)
total_size = int(head.headers.get("content-length", 0))
print(f"Total size: {total_size} bytes ({total_size / (1024*1024):.2f} MB)")

chunk_size = total_size // NUM_THREADS
ranges = []
for i in range(NUM_THREADS):
    start = i * chunk_size
    end = (start + chunk_size - 1) if i < NUM_THREADS - 1 else total_size - 1
    ranges.append((i, start, end))

part_files = [DEST_FILE + f".part{i}" for i in range(NUM_THREADS)]

def download_part(i, start, end):
    part_path = part_files[i]
    target_bytes = end - start + 1
    
    while True:
        existing = 0
        if os.path.exists(part_path):
            existing = os.path.getsize(part_path)
        if existing >= target_bytes:
            print(f"Part {i} already complete ({existing/(1024*1024):.2f} MB).")
            return i, existing
        
        current_start = start + existing
        headers = {"Range": f"bytes={current_start}-{end}"}
        try:
            r = requests.get(URL, headers=headers, stream=True, timeout=30)
            if r.status_code not in (200, 206):
                print(f"Part {i} status {r.status_code}, retrying...")
                time.sleep(2)
                continue
            with open(part_path, "ab" if existing > 0 else "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            # Verify size
            sz = os.path.getsize(part_path)
            if sz >= target_bytes:
                print(f"Part {i} finished ({sz/(1024*1024):.2f} MB).")
                return i, sz
        except Exception as e:
            print(f"Part {i} connection hiccup ({e}), resuming in 2s...")
            time.sleep(2)

print(f"Starting resilient parallel download with {NUM_THREADS} threads...")
start_time = time.time()
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = [executor.submit(download_part, i, s, e) for i, s, e in ranges]
    for future in as_completed(futures):
        idx, sz = future.result()

# Assemble all parts
print("All parts downloaded! Assembling final file...")
with open(DEST_FILE, "wb") as outfile:
    for part_path in part_files:
        with open(part_path, "rb") as infile:
            while True:
                buf = infile.read(1024 * 1024 * 8)
                if not buf:
                    break
                outfile.write(buf)
        os.remove(part_path)

print(f"Download & assembly finished in {time.time() - start_time:.2f}s! Final size: {os.path.getsize(DEST_FILE)} bytes.")
