import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.core.database import SessionLocal
from app.models.document import DocumentChunk

db = SessionLocal()
c = db.query(DocumentChunk).filter(DocumentChunk.document_id == '61019e5d-e369-4c1c-b860-7be9be4c9c34', DocumentChunk.section_title == 'Plan').first()
print("Plan chunk content:\n", repr(c.content))
db.close()
