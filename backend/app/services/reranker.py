import abc
import logging
import math
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentforge.services.reranker")

STOP_WORDS = {
    "what", "is", "the", "a", "an", "in", "on", "of", "for", "to", "and", "or", "with",
    "are", "was", "were", "at", "by", "who", "how", "when", "where", "which", "do", "does",
    "did", "have", "has", "had", "be", "been", "being", "this", "that", "these", "those",
    "tell", "me", "about", "show", "give", "list", "any", "some", "can", "could", "would",
}

# Domain Intent Mappings
INTENT_SECTION_MAPPINGS = {
    "diagnosis": {
        "keywords": {"disease", "diagnosis", "condition", "syndrome", "illness", "disorder", "finding", "findings", "assessment", "impression", "diagnosed", "consistent with", "etiology", "pathology", "cause", "thyroid"},
        "sections": {"assessment", "impression", "diagnosis", "clinical assessment", "diagnostic findings"},
        "content_terms": {"consistent with", "diagnos", "angina", "coronary", "artery disease", "infarction", "syndrome", "hashimoto", "hypothyroid", "hyperthyroid", "thyroiditis", "goiter", "carcinoma", "adenoma"},
    },
    "treatment": {
        "keywords": {"treatment", "plan", "medication", "medications", "therapy", "drug", "drugs", "prescription", "prescribe", "dose", "dosage", "management", "stat", "start", "recommend", "recommended", "aspirin", "statin", "atorvastatin", "lifestyle", "levothyroxine", "methimazole", "surgery", "inhaler"},
        "sections": {"plan", "treatment plan", "medications", "current medications", "recommendations"},
        "content_terms": {"start", "prescrib", "recommend", "mg", "daily", "atorvastatin", "aspirin", "lifestyle", "levothyroxine", "therapy", "follow-up"},
    },
    "symptoms": {
        "keywords": {"symptom", "symptoms", "complaint", "pain", "discomfort", "tightness", "dyspnea", "shortness", "breath", "cough", "fever", "headache", "fatigue", "exertion", "chest", "weight", "palpitations", "sweating", "heat", "cold", "tremor"},
        "sections": {"chief complaint", "history of present illness", "review of systems"},
        "content_terms": {"presents with", "chest tightness", "discomfort", "shortness of breath", "dyspnea", "exertion", "fatigue", "palpitations", "weight"},
    },
    "examination": {
        "keywords": {"examination", "physical", "exam", "vitals", "vital signs", "blood pressure", "heart rate", "pulse", "respiratory", "auscultation", "edema", "murmurs", "lungs", "palpation", "thyromegaly", "bruit"},
        "sections": {"physical examination", "vital signs", "physical exam", "examination"},
        "content_terms": {"blood pressure", "heart rate", "bpm", "oxygen saturation", "auscultation", "clear", "regular", "palpation", "thyroid"},
    },
    "diagnostics": {
        "keywords": {"diagnostic", "findings", "ecg", "ekg", "troponin", "stress test", "imaging", "ct", "angiography", "lipid", "ldl", "hdl", "panel", "labs", "laboratory", "blood test", "thyroid", "tsh", "t3", "t4", "ultrasound", "biopsy", "fna", "antibody", "tpo"},
        "sections": {"diagnostic findings", "laboratory data", "diagnostic studies", "labs", "imaging", "endocrinology", "thyroid findings"},
        "content_terms": {"ecg", "sinus rhythm", "troponin", "stress test", "st depression", "ldl", "hdl", "mg/dl", "tsh", "t3", "t4", "ultrasound", "biopsy", "nodule", "hypoechoic", "vascularity", "mcu/ml"},
    },
    "history": {
        "keywords": {"history", "family", "father", "mother", "past", "prior", "surgical", "medical history", "smoker", "smoking", "alcohol"},
        "sections": {"history of present illness", "family history", "past medical history", "social history"},
        "content_terms": {"family history", "father", "prior", "history of", "age 60"},
    },
}


class BaseReranker(abc.ABC):
    """
    Abstract interface for hybrid query-candidate reranking.
    """

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        relevance_threshold: float = 0.25,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks according to relevance to the query."""
        pass


class LocalSemanticReranker(BaseReranker):
    """
    High-precision hybrid reranker combining:
    1. Dense Vector Similarity (ChromaDB cosine similarity).
    2. BM25 / Exact Lexical Keyword Overlap & Term Frequency.
    3. Section & Clinical Intent Alignment.
    4. Out-of-domain relevance threshold filtering (rejects unrelated queries).
    """

    def __init__(self, default_threshold: float = 0.25):
        self.default_threshold = default_threshold

    def compute_lexical_score(self, query_tokens: List[str], text: str, section: str) -> float:
        """Compute term frequency and overlap score."""
        if not query_tokens:
            return 0.0

        content_words = re.findall(r"\b\w+\b", text.lower())
        total_content_words = max(1, len(content_words))

        matched_tokens = 0
        term_frequency_sum = 0.0

        for token in query_tokens:
            count = content_words.count(token)
            if token in section.lower():
                count += 3  # strong boost for section header match

            if count > 0:
                matched_tokens += 1
                tf = count / total_content_words
                term_frequency_sum += math.log1p(tf * 100)

        coverage_ratio = matched_tokens / len(query_tokens)
        bm25_proxy = min(1.0, term_frequency_sum * 0.4)
        return 0.6 * coverage_ratio + 0.4 * bm25_proxy

    def compute_intent_score(self, query_tokens: List[str], content_lower: str, section_lower: str) -> float:
        """Evaluate alignment between query domain intent and chunk section/content."""
        if not query_tokens:
            return 0.0

        matched_intents = 0
        intent_score = 0.0

        for intent_name, intent_data in INTENT_SECTION_MAPPINGS.items():
            query_has_intent = any(k in query_tokens for k in intent_data["keywords"])
            if query_has_intent:
                matched_intents += 1
                # Section match bonus
                if any(sec in section_lower for sec in intent_data["sections"]):
                    intent_score += 0.60
                # Content match bonus
                if any(term in content_lower for term in intent_data["content_terms"]):
                    intent_score += 0.35

        if matched_intents == 0:
            return 0.0

        return min(1.0, intent_score / matched_intents)

    def compute_chunk_score(self, query: str, chunk: Dict[str, Any]) -> float:
        content = chunk.get("content") or ""
        section = chunk.get("section_title") or ""
        dense_sim = float(chunk.get("similarity_score", 0.0))

        content_lower = content.lower()
        section_lower = section.lower()

        tokens = [w for w in re.findall(r"\b\w+\b", query.lower()) if w not in STOP_WORDS]
        if not tokens:
            return dense_sim

        # 1. Lexical & TF-IDF score
        lexical_score = self.compute_lexical_score(tokens, content_lower, section_lower)

        # 2. Intent alignment score
        intent_score = self.compute_intent_score(tokens, content_lower, section_lower)

        # Low coverage penalty: if fewer than 35% of query tokens match and no domain intent matches
        matched_tokens = sum(1 for t in tokens if t in content_lower or t in section_lower)
        coverage_ratio = matched_tokens / len(tokens) if tokens else 0.0

        if coverage_ratio < 0.35 and intent_score == 0.0:
            lexical_score *= (coverage_ratio * 0.5)

        # 3. Dense similarity score (discount if lexical and intent are low)
        if lexical_score < 0.10 and intent_score == 0.0:
            # Out-of-domain / unrelated query penalty
            combined = dense_sim * 0.15 + lexical_score * 0.20
        else:
            combined = (
                0.40 * dense_sim +
                0.25 * lexical_score +
                0.35 * intent_score
            )

        return round(min(1.0, max(0.0, combined)), 4)


    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        relevance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        threshold = relevance_threshold if relevance_threshold is not None else self.default_threshold

        scored_chunks = []
        for chunk in chunks:
            chunk_copy = dict(chunk)
            rerank_score = self.compute_chunk_score(query, chunk_copy)
            chunk_copy["relevance_score"] = rerank_score
            chunk_copy["similarity_score"] = float(chunk_copy.get("similarity_score", 0.0))
            chunk_copy["distance"] = float(chunk_copy.get("distance", 0.0))

            # Filter by relevance threshold
            if rerank_score >= threshold:
                scored_chunks.append(chunk_copy)

        # Sort descending by relevance_score, falling back to similarity_score
        scored_chunks.sort(key=lambda x: (x["relevance_score"], x["similarity_score"]), reverse=True)
        return scored_chunks[:top_k]


_reranker_instance: Optional[BaseReranker] = None


def get_reranker() -> BaseReranker:
    """Dependency injection factory for query reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LocalSemanticReranker()
    return _reranker_instance
