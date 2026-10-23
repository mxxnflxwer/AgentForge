import json
import sys
import os
sys.path.insert(0, os.path.abspath("."))
from app.core.database import SessionLocal
from app.models.document import Document, DocumentChunk
from app.services.vector_store import get_vector_store
from app.core.config import settings
import chromadb

db = SessionLocal()
print("=== POSTGRES DOCUMENTS ===")
docs = db.query(Document).all()
for d in docs:
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == d.id).all()
    print(f"Doc ID: {d.id}, Name: {d.original_filename}, Status: {d.processing_status}, Stored chunk_count: {d.chunk_count}, Actual chunks in DB: {len(chunks)}, Owner ID: {d.owner_id}")
    for c in chunks[:2]:
        print(f"   Chunk {c.chunk_index} (ID: {c.id}, Section: {c.section_title}): {c.content[:60]}...")

import sqlite3
conn = sqlite3.connect("storage/chromadb/chroma.sqlite3")
cur = conn.cursor()
print("=== SQLITE COLLECTIONS ===")
for row in cur.execute("SELECT id, name, dimension FROM collections"):
    print("Collection:", row)
for row in cur.execute("SELECT collection_id, count(*) FROM embeddings GROUP BY collection_id"):
    print("Embeddings count:", row)
for row in cur.execute("SELECT key, string_value, int_value, float_value FROM embedding_metadata LIMIT 10"):
    print("Sample metadata:", row)
conn.close()

for col in cols:
    print(f"\nCollection: {col.name}, Count: {col.count()}, Metadata: {col.metadata}")
    if col.count() > 0:
        sample = col.peek(limit=2)
        print("   Sample IDs:", sample.get("ids"))
        print("   Sample Metadatas:", json.dumps(sample.get("metadatas"), indent=4))
        embeddings = sample.get("embeddings")
        if embeddings is not None and len(embeddings) > 0:
            print("   Embedding dim:", len(embeddings[0]))

db.close()
