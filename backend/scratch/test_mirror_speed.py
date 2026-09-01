import requests
import time

urls = [
    "https://hf-mirror.com/BAAI/bge-m3/resolve/main/pytorch_model.bin",
    "https://huggingface.co/BAAI/bge-m3/resolve/main/pytorch_model.bin",
    "https://modelscope.cn/api/v1/models/BAAI/bge-m3/repo?Revision=master&FilePath=pytorch_model.bin",
]

for url in urls:
    try:
        t0 = time.time()
        r = requests.get(url, headers={"Range": "bytes=0-10485759"}, timeout=10) # 10MB test
        dt = time.time() - t0
        mb = len(r.content) / (1024 * 1024)
        speed = mb / dt if dt > 0 else 0
        print(f"URL: {url[:40]}... Status: {r.status_code}, Read: {mb:.2f} MB in {dt:.2f}s ({speed:.2f} MB/s)")
    except Exception as e:
        print(f"URL: {url[:40]}... Error: {e}")
