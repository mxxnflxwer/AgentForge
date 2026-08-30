import io
from docx import Document as DocxDocument
from pypdf import PdfWriter
import pytest

from app.services.document.cleaner import TextCleaner
from app.services.document.chunker import DocumentChunker
from app.services.document.docx_processor import DocxProcessor
from app.services.document.pdf_processor import PDFProcessor
from app.services.document.section_detector import SectionDetector
from app.services.document.txt_processor import TxtProcessor
from app.services.storage.local import LocalStorageBackend


def test_text_cleaner():
    raw = "Patient  presents   with treat-\nment resistant hypertension.\n\n\n\n\nBlood pressure: 150/95.\u00a0\u00a0"
    cleaned = TextCleaner.clean(raw)
    assert "treat- ment" not in cleaned
    assert "treatment" in cleaned
    assert "  " not in cleaned
    assert "\n\n\n" not in cleaned
    assert "150/95" in cleaned


def test_txt_processor_utf8_and_latin1():
    processor = TxtProcessor()
    
    # UTF-8 text
    utf8_bytes = "CHIEF COMPLAINT: Patient has severe migraine.".encode("utf-8")
    doc = processor.extract(utf8_bytes)
    assert "migraine" in doc.raw_text
    assert doc.page_count == 1
    assert doc.character_count > 0

    # Latin-1 text with special chars
    latin1_bytes = "DIAGNOSIS: Café au lait macules observed.".encode("latin-1")
    doc_latin = processor.extract(latin1_bytes)
    assert "Café" in doc_latin.raw_text or "Cafe" in doc_latin.raw_text


def test_docx_processor():
    processor = DocxProcessor()
    
    # Create in-memory docx
    docx_file = io.BytesIO()
    doc = DocxDocument()
    doc.add_heading("CLINICAL SUMMARY", level=1)
    doc.add_paragraph("Patient evaluated in cardiology clinic.")
    
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Test Name"
    table.cell(0, 1).text = "Result"
    table.cell(1, 0).text = "ECG"
    table.cell(1, 1).text = "Normal Sinus Rhythm"
    doc.save(docx_file)
    docx_bytes = docx_file.getvalue()

    extracted = processor.extract(docx_bytes)
    assert "CLINICAL SUMMARY" in extracted.raw_text
    assert "cardiology clinic" in extracted.raw_text
    assert "Normal Sinus Rhythm" in extracted.raw_text


def test_pdf_processor_with_text():
    processor = PDFProcessor()

    # Create a small valid PDF in memory
    pdf_writer = PdfWriter()
    pdf_writer.add_blank_page(width=200, height=200)
    pdf_stream = io.BytesIO()
    pdf_writer.write(pdf_stream)
    pdf_bytes = pdf_stream.getvalue()

    extracted = processor.extract(pdf_bytes)
    assert extracted.page_count == 1


def test_section_detector():
    detector = SectionDetector()
    sample_text = (
        "CHIEF COMPLAINT:\n"
        "Chest pain radiating to left arm.\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "The patient is a 55-year-old male presenting with acute sub-sternal pressure.\n\n"
        "ASSESSMENT & PLAN:\n"
        "1. Order serial troponins and 12-lead ECG."
    )
    sections = detector.detect_sections(sample_text)
    assert len(sections) >= 3
    titles = [s.title for s in sections]
    assert any("CHIEF COMPLAINT" in t.upper() for t in titles)
    assert any("HISTORY OF PRESENT ILLNESS" in t.upper() for t in titles)
    assert any("ASSESSMENT & PLAN" in t.upper() or "PLAN" in t.upper() for t in titles)


def test_document_chunker():
    detector = SectionDetector()
    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20)
    
    sample_text = (
        "CHIEF COMPLAINT:\n"
        "Shortness of breath on exertion for the past 3 weeks.\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "Patient reports worsening dyspnea with minimal activity. No fever, cough, or chills. "
        "Symptoms have progressively impaired daily living activities including walking up stairs.\n\n"
        "MEDICATIONS:\n"
        "Lisinopril 10mg daily, Metoprolol 25mg daily."
    )
    
    sections = detector.detect_sections(sample_text)
    chunks = chunker.chunk_sections(sections)
    
    assert len(chunks) >= 3
    for idx, c in enumerate(chunks):
        assert c.chunk_index == idx
        assert len(c.content) > 0
        assert c.character_count == len(c.content)
        assert c.token_count > 0


def test_local_storage_backend(tmp_path):
    storage = LocalStorageBackend(base_dir=str(tmp_path))
    test_data = b"AgentForge unit test payload"
    filename = "test_doc.bin"

    stored_path = storage.save(test_data, filename)
    assert storage.exists(filename)
    assert storage.read(filename) == test_data

    storage.delete(filename)
    assert not storage.exists(filename)
