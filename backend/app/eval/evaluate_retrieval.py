import json
import logging
import math
from typing import Any, Dict, List, Set
from app.core.database import SessionLocal
from app.models.user import User
from app.services.retrieval_service import search_similar_chunks
from app.services.vector_store import get_vector_store

logging.basicConfig(level=logging.WARNING)

BENCHMARK_DATA = [
    # --- In-Domain Queries (14 queries) ---
    {
        "id": "Q01",
        "query": "What is the disease / diagnosis for the cardiology patient?",
        "is_in_domain": True,
        "target_sections": {"Assessment"},
        "target_keywords": {"stable angina", "coronary artery disease"},
    },
    {
        "id": "Q02",
        "query": "What medications are prescribed in the treatment plan?",
        "is_in_domain": True,
        "target_sections": {"Plan"},
        "target_keywords": {"atorvastatin", "aspirin"},
    },
    {
        "id": "Q03",
        "query": "What were the patient's chief complaints on presentation?",
        "is_in_domain": True,
        "target_sections": {"Chief Complaint"},
        "target_keywords": {"chest discomfort", "shortness of breath"},
    },
    {
        "id": "Q04",
        "query": "What are the physical examination vital signs?",
        "is_in_domain": True,
        "target_sections": {"Physical Examination"},
        "target_keywords": {"138/86", "78 bpm", "oxygen saturation"},
    },
    {
        "id": "Q05",
        "query": "What did the ECG and stress test findings show?",
        "is_in_domain": True,
        "target_sections": {"Diagnostic Findings"},
        "target_keywords": {"ecg", "stress test", "troponin", "depression"},
    },
    {
        "id": "Q06",
        "query": "Is there a family history of coronary artery disease?",
        "is_in_domain": True,
        "target_sections": {"History of Present Illness"},
        "target_keywords": {"father", "coronary artery disease", "age 60"},
    },
    {
        "id": "Q07",
        "query": "What lifestyle modifications were recommended?",
        "is_in_domain": True,
        "target_sections": {"Plan"},
        "target_keywords": {"dietary changes", "exercise program"},
    },
    {
        "id": "Q08",
        "query": "What is the patient's LDL and lipid panel result?",
        "is_in_domain": True,
        "target_sections": {"Diagnostic Findings"},
        "target_keywords": {"ldl 162", "hdl 38", "lipid"},
    },
    {
        "id": "Q09",
        "query": "What is the assessment for the endocrinology patient?",
        "is_in_domain": True,
        "target_sections": {"Assessment"},
        "target_keywords": {"hypothyroidism", "hashimoto"},
    },
    {
        "id": "Q10",
        "query": "What medication was prescribed for the thyroid condition?",
        "is_in_domain": True,
        "target_sections": {"Plan"},
        "target_keywords": {"levothyroxine", "thyroid", "mcg"},
    },
    {
        "id": "Q11",
        "query": "What symptoms did the endocrinology patient report?",
        "is_in_domain": True,
        "target_sections": {"Chief Complaint", "History of Present Illness"},
        "target_keywords": {"fatigue", "cold intolerance", "weight gain"},
    },
    {
        "id": "Q12",
        "query": "What were the TSH and free T4 laboratory findings?",
        "is_in_domain": True,
        "target_sections": {"Diagnostic Findings"},
        "target_keywords": {"tsh", "free t4", "thyroid"},
    },
    {
        "id": "Q13",
        "query": "What follow-up timeline is recommended for cardiology?",
        "is_in_domain": True,
        "target_sections": {"Plan"},
        "target_keywords": {"2 weeks", "follow-up", "imaging"},
    },
    {
        "id": "Q14",
        "query": "What was the patient's oxygen saturation on room air?",
        "is_in_domain": True,
        "target_sections": {"Physical Examination"},
        "target_keywords": {"98%", "room air", "oxygen saturation"},
    },
    # --- Out-of-Domain Unrelated Queries (6 queries - Expected: Rejection) ---
    {
        "id": "Q15",
        "query": "How do you bake a triple chocolate fudge cake?",
        "is_in_domain": False,
        "target_sections": set(),
        "target_keywords": set(),
    },
    {
        "id": "Q16",
        "query": "Who won the 1998 FIFA World Cup final in Paris?",
        "is_in_domain": False,
        "target_sections": set(),
        "target_keywords": set(),
    },
    {
        "id": "Q17",
        "query": "What is Schrödinger's wave equation in quantum physics?",
        "is_in_domain": False,
        "target_sections": set(),
        "target_keywords": set(),
    },
    {
        "id": "Q18",
        "query": "What are the best tourist attractions and hotels in Tokyo?",
        "is_in_domain": False,
        "target_sections": set(),
        "target_keywords": set(),
    },
    {
        "id": "Q19",
        "query": "How do you change the engine oil on a 2018 Honda Civic?",
        "is_in_domain": False,
        "target_sections": set(),
        "target_keywords": set(),
    },
    {
        "id": "Q20",
        "query": "What are the rules of Texas hold'em poker?",
        "is_in_domain": False,
        "target_sections": set(),
        "target_keywords": set(),
    },
]


def is_chunk_relevant(chunk: Any, target_sections: Set[str], target_keywords: Set[str]) -> bool:
    section = chunk.section_title or ""
    content_lower = chunk.content.lower()

    section_hit = any(ts.lower() in section.lower() for ts in target_sections)
    keyword_hit = any(kw.lower() in content_lower for kw in target_keywords)

    return section_hit or keyword_hit


def compute_dcg(relevances: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def compute_idcg(num_relevant: int, k: int) -> float:
    idcg = 0.0
    for i in range(min(num_relevant, k)):
        idcg += 1.0 / math.log2(i + 2)
    return idcg


def run_evaluation(top_k: int = 3, threshold: float = 0.25):
    db = SessionLocal()
    user = db.query(User).filter(User.id == "863ffca5-e2c2-4a53-bc58-441debdd5177").first()
    if not user:
        user = db.query(User).first()

    user_id = user.id
    vector_store = get_vector_store()

    in_domain_results = []
    unrelated_results = []

    reciprocal_ranks = []
    precisions_at_k = []
    recalls_at_k = []
    ndcgs_at_k = []

    true_rejections = 0
    total_unrelated = 0

    print("=" * 80)
    print(f"AGENTFORGE RAG EVALUATION BENCHMARK (Top-K={top_k}, Relevance Threshold={threshold})")
    print("=" * 80)

    for item in BENCHMARK_DATA:
        q_id = item["id"]
        query = item["query"]
        is_in_domain = item["is_in_domain"]

        results = search_similar_chunks(
            user_id=user_id,
            query=query,
            top_k=top_k,
            relevance_threshold=threshold,
            db=db,
            vector_store=vector_store,
        )

        if not is_in_domain:
            total_unrelated += 1
            is_rejected = len(results) == 0
            if is_rejected:
                true_rejections += 1
            unrelated_results.append({
                "id": q_id,
                "query": query,
                "rejected": is_rejected,
                "num_results": len(results),
            })
            status_str = "REJECTED (Correct)" if is_rejected else f"FAILED (Retrieved {len(results)})"
            print(f"[{q_id}] Unrelated: '{query[:45]}...' -> {status_str}")
        else:
            rel_binary = []
            first_rank = None

            for rank_idx, chunk in enumerate(results):
                rel = 1 if is_chunk_relevant(chunk, item["target_sections"], item["target_keywords"]) else 0
                rel_binary.append(rel)
                if rel == 1 and first_rank is None:
                    first_rank = rank_idx + 1

            # Precision@K
            p_k = sum(rel_binary) / top_k if top_k > 0 else 0.0
            precisions_at_k.append(p_k)

            # Recall@K (assuming target relevant chunk count is at least 1)
            r_k = 1.0 if sum(rel_binary) >= 1 else 0.0
            recalls_at_k.append(r_k)

            # MRR
            rr = (1.0 / first_rank) if first_rank else 0.0
            reciprocal_ranks.append(rr)

            # NDCG@K
            dcg = compute_dcg(rel_binary, top_k)
            idcg = compute_idcg(max(1, sum(rel_binary)), top_k)
            ndcg = (dcg / idcg) if idcg > 0 else 0.0
            ndcgs_at_k.append(ndcg)

            top_chunk_title = results[0].section_title if results else "None"
            top_chunk_score = results[0].relevance_score if results else 0.0

            in_domain_results.append({
                "id": q_id,
                "query": query,
                "top_section": top_chunk_title,
                "relevance_score": top_chunk_score,
                "precision": p_k,
                "recall": r_k,
                "mrr": rr,
                "ndcg": ndcg,
            })

            print(f"[{q_id}] In-Domain: '{query[:40]}...' -> Top: [{top_chunk_title}] Score: {top_chunk_score:.3f} | P@{top_k}: {p_k:.2f} | MRR: {rr:.2f} | NDCG@{top_k}: {ndcg:.2f}")

    # Summary Metrics
    mean_precision = sum(precisions_at_k) / len(precisions_at_k) if precisions_at_k else 0.0
    mean_recall = sum(recalls_at_k) / len(recalls_at_k) if recalls_at_k else 0.0
    mean_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    mean_ndcg = sum(ndcgs_at_k) / len(ndcgs_at_k) if ndcgs_at_k else 0.0
    rejection_accuracy = (true_rejections / total_unrelated) * 100 if total_unrelated > 0 else 100.0

    print("\n" + "=" * 80)
    print("EVALUATION METRIC SUMMARY")
    print("=" * 80)
    print(f"Total Test Questions:      {len(BENCHMARK_DATA)}")
    print(f"In-Domain Queries:         {len(precisions_at_k)}")
    print(f"Out-of-Domain Queries:     {total_unrelated}")
    print(f"Precision@{top_k}:              {mean_precision:.4f} ({mean_precision * 100:.1f}%)")
    print(f"Recall@{top_k}:                 {mean_recall:.4f} ({mean_recall * 100:.1f}%)")
    print(f"Mean Reciprocal Rank (MRR):{mean_mrr:.4f}")
    print(f"NDCG@{top_k}:                   {mean_ndcg:.4f}")
    print(f"Rejection Accuracy:        {rejection_accuracy:.1f}% ({true_rejections}/{total_unrelated})")
    print("=" * 80)

    db.close()
    return {
        "precision_at_k": mean_precision,
        "recall_at_k": mean_recall,
        "mrr": mean_mrr,
        "ndcg_at_k": mean_ndcg,
        "rejection_accuracy": rejection_accuracy,
        "in_domain_results": in_domain_results,
        "unrelated_results": unrelated_results,
    }


if __name__ == "__main__":
    run_evaluation(top_k=3, threshold=0.25)
