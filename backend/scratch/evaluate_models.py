import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import math
import uuid
import numpy as np
from typing import Dict, List, Any

from app.models.document import DocumentChunk
from app.services.embedding_service import LocalChromaEmbeddingService, SentenceTransformerEmbeddingService
from app.services.vector_store import VectorStore
from app.services.retrieval_service import search_similar_chunks
from app.services.reranker import LocalSemanticReranker

# Sample structured clinical documents
CARDIOLOGY_DOC_CHUNKS = [
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=0,
        content="Specialty: Cardiology\nReport Type: Consultation History and Physical Examination\nPatient: John Doe, 58M",
        section_title="Document Overview",
        token_count=18,
        character_count=110,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=1,
        content="Chief Complaint\nPatient presents with intermittent chest tightness, substernal pressure, and shortness of breath on exertion for the past 3 weeks.",
        section_title="Chief Complaint",
        token_count=24,
        character_count=154,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=2,
        content="History of Present Illness\nA 58-year-old male with a history of hyperlipidemia reports exertional substernal chest discomfort radiating to the left arm, relieved by rest within 5 minutes.",
        section_title="History of Present Illness",
        token_count=28,
        character_count=178,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=3,
        content="Physical Examination\nBlood pressure 142/88 mmHg, heart rate 76 bpm regular, SpO2 98% on room air. Cardiovascular: S1/S2 present, no murmurs, rubs, or gallops. Lungs: Clear to auscultation bilaterally.",
        section_title="Physical Examination",
        token_count=32,
        character_count=202,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=4,
        content="Diagnostic Findings\nECG reveals normal sinus rhythm with 1mm ST-segment depression in lateral leads V5-V6 during treadmill stress testing. High-sensitivity troponin is negative. Lipid panel shows LDL 162 mg/dL.",
        section_title="Diagnostic Findings",
        token_count=32,
        character_count=215,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=5,
        content="Assessment\n1. Exertional angina pectoris consistent with underlying coronary artery disease (CAD).\n2. Uncontrolled hyperlipidemia.",
        section_title="Assessment",
        token_count=18,
        character_count=138,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_cardio_1",
        chunk_index=6,
        content="Plan\n1. Initiate Atorvastatin 40mg PO daily.\n2. Prescribe Aspirin 81mg daily and sublingual Nitroglycerin 0.4mg PRN chest pain.\n3. Schedule coronary CT angiography next week.\n4. Recommend cardiac rehabilitation and Mediterranean diet.",
        section_title="Plan",
        token_count=38,
        character_count=248,
    ),
]

ENDOCRINOLOGY_DOC_CHUNKS = [
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=0,
        content="Specialty: Endocrinology\nReport Type: Outpatient Endocrine Evaluation\nPatient: Jane Smith, 42F",
        section_title="Document Overview",
        token_count=16,
        character_count=102,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=1,
        content="Chief Complaint\nPatient presents with progressive fatigue, unexplained weight gain of 12 lbs, cold intolerance, and anterior neck fullness over 4 months.",
        section_title="Chief Complaint",
        token_count=23,
        character_count=153,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=2,
        content="History of Present Illness\nA 42-year-old female presents with sluggishness, severe dry skin, constipation, and difficulty staying warm in heated environments. Maternal aunt had Hashimoto thyroiditis.",
        section_title="History of Present Illness",
        token_count=27,
        character_count=185,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=3,
        content="Physical Examination\nVitals: BP 118/74, HR 58 bpm (bradycardia), Temp 97.4 F. Neck: Symmetrical, diffusely enlarged, non-tender thyroid gland (thyromegaly) without palpable discrete nodules or bruit.",
        section_title="Physical Examination",
        token_count=30,
        character_count=198,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=4,
        content="Diagnostic Findings\nThyroid panel reveals elevated TSH at 14.8 mIU/L (normal 0.4-4.0) and low Free T4 at 0.6 ng/dL. Anti-TPO antibodies strongly positive at >500 IU/mL. Thyroid ultrasound shows heterogeneous hypoechoic parenchyma.",
        section_title="Diagnostic Findings",
        token_count=36,
        character_count=238,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=5,
        content="Assessment\nPrimary hypothyroidism secondary to chronic autoimmune thyroiditis (Hashimoto disease).",
        section_title="Assessment",
        token_count=11,
        character_count=98,
    ),
    DocumentChunk(
        id=str(uuid.uuid4()),
        document_id="doc_endo_1",
        chunk_index=6,
        content="Plan\n1. Start Levothyroxine 75 mcg orally every morning on an empty stomach 30-60 min before breakfast.\n2. Recheck serum TSH and Free T4 in 6-8 weeks for dose titration.\n3. Educate on symptoms of hyperthyroidism/over-replacement.",
        section_title="Plan",
        token_count=37,
        character_count=238,
    ),
]

# Benchmark evaluation dataset: (query, document_id, expected_relevant_section_or_none)
EVAL_DATASET = [
    # Cardiology queries
    ("What is the diagnosis?", "doc_cardio_1", "Assessment"),
    ("What are the symptoms?", "doc_cardio_1", "Chief Complaint"),
    ("What are the diagnostic findings?", "doc_cardio_1", "Diagnostic Findings"),
    ("What treatment is recommended?", "doc_cardio_1", "Plan"),
    ("What medications were prescribed?", "doc_cardio_1", "Plan"),
    ("What does the assessment say?", "doc_cardio_1", "Assessment"),
    # Endocrinology queries
    ("What is the diagnosis?", "doc_endo_1", "Assessment"),
    ("What are the symptoms?", "doc_endo_1", "Chief Complaint"),
    ("What are the thyroid findings?", "doc_endo_1", "Diagnostic Findings"),
    ("What is the assessment?", "doc_endo_1", "Assessment"),
    ("What treatment is recommended?", "doc_endo_1", "Plan"),
    # Irrelevant queries (expect 0 results)
    ("What is Google?", "doc_cardio_1", None),
    ("What is the weather?", "doc_cardio_1", None),
    ("Explain quantum computing", "doc_endo_1", None),
    ("Who is Elon Musk?", "doc_endo_1", None),
    ("abcdefxyz123", "doc_cardio_1", None),
]

def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.0

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 1.0 if not any(r) else 0.0
    return dcg_at_k(r, k) / dcg_max

def evaluate_retrieval_system(name: str, embedding_svc, chunks_cardio, chunks_endo, persist_dir: str):
    store = VectorStore(
        persist_directory=persist_dir,
        collection_name=f"eval_{name.lower().replace('-', '_')}_{uuid.uuid4().hex[:6]}",
        embedding_service=embedding_svc,
    )
    user_id = f"eval_user_{uuid.uuid4().hex[:6]}"
    store.upsert_document_chunks(user_id=user_id, document_id="doc_cardio_1", chunks=chunks_cardio)
    store.upsert_document_chunks(user_id=user_id, document_id="doc_endo_1", chunks=chunks_endo)
    
    reranker = LocalSemanticReranker()
    
    p1_list, p3_list, p5_list = [], [], []
    recall5_list = []
    mrr_list = []
    ndcg5_list = []
    irrelevant_rejections = 0
    total_irrelevant = 0
    
    for query, doc_id, expected_section in EVAL_DATASET:
        # Run search with candidate pool expansion + reranking
        candidate_k = 20
        candidates = store.query_similar_chunks(
            user_id=user_id,
            query_text=query,
            document_id=doc_id,
            top_k=candidate_k,
        )
        reranked = reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=5,
            relevance_threshold=0.25,
        )
        
        if expected_section is None:
            total_irrelevant += 1
            if len(reranked) == 0:
                irrelevant_rejections += 1
            continue
        
        # Binary relevance for top results
        retrieved_sections = [c.get("section_title") for c in reranked]
        binary_rel = [1 if s == expected_section else 0 for s in retrieved_sections]
        
        # Pad binary_rel to 5 for DCG
        padded_rel = binary_rel + [0] * (5 - len(binary_rel))
        
        # P@1, P@3, P@5
        p1 = binary_rel[0] if len(binary_rel) >= 1 else 0
        p3 = sum(binary_rel[:3]) / 3.0 if len(binary_rel) >= 1 else 0
        p5 = sum(binary_rel[:5]) / 5.0 if len(binary_rel) >= 1 else 0
        
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        
        # Recall@5 (1 ground truth chunk per test query)
        recall5 = 1.0 if any(binary_rel[:5]) else 0.0
        recall5_list.append(recall5)
        
        # MRR
        first_rel_rank = next((idx + 1 for idx, rel in enumerate(binary_rel) if rel == 1), 0)
        mrr = (1.0 / first_rel_rank) if first_rel_rank > 0 else 0.0
        mrr_list.append(mrr)
        
        # NDCG@5
        ndcg = ndcg_at_k(padded_rel, 5)
        ndcg5_list.append(ndcg)
        
    return {
        "model": name,
        "p1": np.mean(p1_list),
        "p3": np.mean(p3_list),
        "p5": np.mean(p5_list),
        "recall5": np.mean(recall5_list),
        "mrr": np.mean(mrr_list),
        "ndcg5": np.mean(ndcg5_list),
        "irrelevant_rejection_rate": (irrelevant_rejections / total_irrelevant) if total_irrelevant else 1.0,
    }

if __name__ == "__main__":
    print("Benchmarking MiniLM vs BGE-M3 on deterministic clinical query set...")
    
    # 1. MiniLM
    print("\nEvaluating all-MiniLM-L6-v2...")
    minilm_svc = LocalChromaEmbeddingService(model_name="all-MiniLM-L6-v2")
    minilm_metrics = evaluate_retrieval_system(
        "MiniLM", minilm_svc, CARDIOLOGY_DOC_CHUNKS, ENDOCRINOLOGY_DOC_CHUNKS, "storage/chromadb_eval_minilm"
    )
    
    # 2. BGE-M3
    print("\nEvaluating BAAI/bge-m3...")
    bgem3_svc = SentenceTransformerEmbeddingService(model_name="BAAI/bge-m3")
    bgem3_metrics = evaluate_retrieval_system(
        "BGE-M3", bgem3_svc, CARDIOLOGY_DOC_CHUNKS, ENDOCRINOLOGY_DOC_CHUNKS, "storage/chromadb_eval_bgem3"
    )
    
    print("\n" + "="*50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*50)
    print(f"| Metric               | MiniLM  | BGE-M3  |")
    print(f"| -------------------- | ------: | ------: |")
    print(f"| Precision@1          | {minilm_metrics['p1']:.4f}  | {bgem3_metrics['p1']:.4f}  |")
    print(f"| Precision@3          | {minilm_metrics['p3']:.4f}  | {bgem3_metrics['p3']:.4f}  |")
    print(f"| Precision@5          | {minilm_metrics['p5']:.4f}  | {bgem3_metrics['p5']:.4f}  |")
    print(f"| Recall@5             | {minilm_metrics['recall5']:.4f}  | {bgem3_metrics['recall5']:.4f}  |")
    print(f"| MRR                  | {minilm_metrics['mrr']:.4f}  | {bgem3_metrics['mrr']:.4f}  |")
    print(f"| NDCG@5               | {minilm_metrics['ndcg5']:.4f}  | {bgem3_metrics['ndcg5']:.4f}  |")
    print(f"| Irrelevant rejection | {minilm_metrics['irrelevant_rejection_rate']*100:.1f}%  | {bgem3_metrics['irrelevant_rejection_rate']*100:.1f}%  |")
    print("="*50)
