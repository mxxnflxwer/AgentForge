from huggingface_hub import HfApi

api = HfApi()
info = api.model_info("BAAI/bge-m3", files_metadata=True)
for s in info.siblings:
    print(f"{s.rfilename}: {s.size} bytes ({s.size/(1024*1024):.2f} MB)" if s.size else s.rfilename)
