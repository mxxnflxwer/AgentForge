import abc
import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings

logger = logging.getLogger("agentforge.services.reranker")

# Standard query stopwords
STOP_WORDS: Set[str] = {
    "what", "is", "the", "a", "an", "in", "on", "of", "for", "to", "and", "or", "with",
    "are", "was", "were", "at", "by", "who", "how", "when", "where", "which", "do", "does",
    "did", "have", "has", "had", "be", "been", "being", "this", "that", "these", "those",
    "tell", "me", "about", "show", "give", "list", "any", "some", "can", "could", "would",
    "they", "their", "them", "he", "she", "it", "its", "there", "patient", "patients",
}

# Conversational greetings / non-informational queries to reject immediately
GREETING_PATTERNS: Set[str] = {
    "hi", "hello", "hey", "howdy", "greetings", "good morning", "good afternoon",
    "good evening", "good day", "how are you", "what's up", "whats up", "yo",
    "test", "testing", "ok", "okay", "thanks", "thank you", "bye", "goodbye",
    "who are you", "what can you do", "help", "help me",
}

# General knowledge / out-of-domain patterns that are unsupported in clinical docs
OUT_OF_DOMAIN_PATTERNS: Set[str] = {
    "google", "weather", "quantum computing", "elon musk", "president", "capital of",
    "movie", "song", "bitcoin", "crypto", "stock market",
}

# Morphological, spelling, and synonym normalization map
SYNONYM_MAP: Dict[str, str] = {
    "speciality": "specialty",
    "specialities": "specialty",
    "specialties": "specialty",
    "oedema": "edema",
    "medications": "medication",
    "medicines": "medication",
    "medicine": "medication",
    "drugs": "medication",
    "drug": "medication",
    "prescriptions": "prescription",
    "prescribed": "prescribe",
    "symptoms": "symptom",
    "complaints": "complaint",
    "diagnoses": "diagnosis",
    "diagnosed": "diagnosis",
    "diagnostic": "diagnosis",
    "diseases": "disease",
    "conditions": "condition",
    "disorders": "disorder",
    "treatments": "treatment",
    "treated": "treatment",
    "therapies": "therapy",
    "findings": "finding",
    "exams": "examination",
    "exam": "examination",
    "ecgs": "ecg",
    "ekgs": "ecg",
    "ekg": "ecg",
    "vitals": "vital",
    "bp": "blood pressure",
    "hr": "heart rate",
    "rates": "rate",
    "levels": "level",
}

# Specific required entity keyword mappings to prevent false-positive matching
# If query contains key, candidate chunk MUST match at least one keyword or section
REQUIRED_ENTITY_KEYWORDS: Dict[str, Set[str]] = {
    "glucose": {"glucose", "blood glucose", "blood sugar", "sugar", "hba1c", "a1c", "glycemic", "diabetes", "diabetic", "mg/dl"},
    "phone": {"phone", "telephone", "cell", "mobile", "contact number", "call"},
    "address": {"address", "residence", "street", "zip", "postal", "city", "state", "residing"},
    "dob": {"dob", "date of birth", "birth date", "born"},
    "hdl": {"hdl", "high-density", "high density lipoprotein"},
    "ldl": {"ldl", "low-density", "low density lipoprotein"},
}

# Domain Intent Mappings
INTENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "diagnosis": {
        "keywords": {"diagnosis", "disease", "condition", "syndrome", "disorder", "illness", "assessment", "impression", "etiology", "pathology"},
        "primary_sections": {"assessment", "impression", "diagnosis", "clinical impression", "clinical assessment"},
        "secondary_sections": set(),
        "content_patterns": [
            r"consistent with", r"stable angina", r"coronary artery disease", r"hypothyroid",
            r"hyperthyroid", r"hashimoto", r"thyroiditis", r"infarction", r"bronchitis",
            r"carcinoma", r"adenoma", r"diagnos",
        ],
    },
    "symptoms": {
        "keywords": {"symptom", "complaint", "pain", "discomfort", "tightness", "shortness", "breath", "dyspnea", "fatigue", "exertion", "cough", "fever", "palpitations", "headache", "cold intolerance", "weight"},
        "primary_sections": {"chief complaint", "history of present illness", "review of systems", "symptoms"},
        "secondary_sections": set(),
        "content_patterns": [
            r"presents with", r"complains of", r"chest discomfort", r"chest tightness",
            r"shortness of breath", r"dyspnea", r"fatigue", r"palpitations", r"cold intolerance",
            r"intermittent", r"exertion",
        ],
    },
    "medications": {
        "keywords": {"medication", "medicine", "drug", "prescription", "prescribe", "dose", "dosage", "tablet", "pill", "mg", "po", "daily", "statin", "atorvastatin", "aspirin", "levothyroxine", "methimazole", "inhaler", "amoxicillin", "antibiotic"},
        "primary_sections": {"plan", "medications", "current medications", "prescriptions", "treatment plan"},
        "secondary_sections": {"assessment"},
        "content_patterns": [
            r"\bstart\b", r"\bprescrib", r"\brecommend\b", r"\bdaily\b", r"\bmg\b", r"\bpo\b",
            r"atorvastatin", r"aspirin", r"levothyroxine", r"amoxicillin", r"nitroglycerin",
        ],
    },
    "treatment": {
        "keywords": {"treatment", "plan", "therapy", "management", "intervention", "recommendation", "recommended", "surgery", "angiography", "follow-up", "lifestyle"},
        "primary_sections": {"plan", "treatment", "treatment plan", "management", "recommendations"},
        "secondary_sections": set(),
        "content_patterns": [
            r"\bplan\b", r"\brecommend", r"\bstart\b", r"lifestyle", r"follow-up", r"angiography",
            r"stratification", r"angioplasty",
        ],
    },
    "demographics": {
        "keywords": {"age", "old", "year-old", "yo", "dob", "birth", "sex", "gender", "male", "female", "demographics"},
        "primary_sections": {"history of present illness", "document overview", "demographics"},
        "secondary_sections": set(),
        "content_patterns": [
            r"\b\d{1,3}\s*-\s*year\s*-\s*old\b", r"\b\d{1,3}\s*yo\b", r"\b\d{1,3}\s*years?\s*old\b",
            r"\bmale\b", r"\bfemale\b", r"\bpatient presents\b",
        ],
    },
    "specialty": {
        "keywords": {"specialty", "department", "service", "consultation", "overview", "clinic", "physician", "doctor"},
        "primary_sections": {"document overview", "overview", "header"},
        "secondary_sections": set(),
        "content_patterns": [
            r"specialty\s*:", r"speciality\s*:", r"cardiology", r"endocrinology", r"pulmonology",
            r"neurology", r"oncology", r"dermatology", r"consultation",
        ],
    },
    "examination": {
        "keywords": {"edema", "examination", "physical", "exam", "auscultation", "vitals", "vital", "blood pressure", "heart rate", "pulse", "lungs", "murmurs", "clear to auscultation", "regular rate"},
        "primary_sections": {"physical examination", "physical exam", "vital signs", "examination"},
        "secondary_sections": set(),
        "content_patterns": [
            r"blood pressure", r"heart rate", r"bpm", r"edema", r"auscultation", r"lungs clear",
            r"no peripheral edema", r"regular rate", r"\d{2,3}/\d{2,3}",
        ],
    },
    "findings": {
        "keywords": {"ecg", "finding", "test", "tests", "labs", "laboratory", "troponin", "tsh", "t4", "t3", "imaging", "ct", "ultrasound", "stress test", "angiography", "biopsy", "panel", "ldl", "hdl", "lipid"},
        "primary_sections": {"diagnostic findings", "laboratory data", "diagnostic studies", "labs", "imaging", "endocrinology", "diagnostic", "investigations"},
        "secondary_sections": set(),
        "content_patterns": [
            r"\becg\b", r"sinus rhythm", r"troponin", r"\btsh\b", r"st depression", r"ultrasound",
            r"biopsy", r"mg/dl", r"miu/l", r"free t4", r"ldl", r"hdl",
        ],
    },
}


def normalize_token(token: str) -> str:
    """Normalize token via lowercasing and synonym/lemmatization mapping."""
    t = token.lower().strip()
    return SYNONYM_MAP.get(t, t)


def is_greeting_or_nonsense(query: str) -> bool:
    """Detect if query is purely conversational, greeting, or meaningless."""
    clean_q = re.sub(r"[^\w\s]", "", query.lower()).strip()
    if not clean_q:
        return True
    if clean_q in GREETING_PATTERNS:
        return True
    tokens = clean_q.split()
    if len(tokens) == 1 and tokens[0] in GREETING_PATTERNS:
        return True
    return False


def is_out_of_domain(query: str) -> bool:
    """Detect general out-of-domain knowledge queries that are not in medical records."""
    clean_q = query.lower().strip()
    for ood in OUT_OF_DOMAIN_PATTERNS:
        if ood in clean_q:
            return True
    return False


def classify_query_intent(query: str) -> str:
    """
    Lightweight rule-based query intent classifier.
    Detects:
    - 'summary': Requests for summarizing the document or complete report.
    - 'diagnosis': Questions about patient diagnosis, disease, or clinical impression.
    - 'findings': Questions about investigations, labs, ECG, imaging, or physical exam findings.
    - 'explanation': Requests to explain or clarify why/how.
    - 'unsupported_query': Conversational greetings or out-of-domain queries.
    - 'specific_fact': Direct factual queries (age, blood pressure, heart rate, medications, etc.).
    """
    q_lower = query.lower().strip()

    if is_greeting_or_nonsense(q_lower) or is_out_of_domain(q_lower):
        return "unsupported_query"

    # Summary Intent
    if any(k in q_lower for k in [
        "summarize", "summary", "overview of the report", "overview of the complete report",
        "summarise", "give me a summary", "complete report", "full report", "synthesize", "document summary"
    ]):
        return "summary"

    # Explanation Intent
    if re.search(r"\b(explain|why|how come|describe why|explain simply|elaborate)\b", q_lower):
        return "explanation"

    # Diagnosis Intent
    if any(k in q_lower for k in [
        "diagnosis", "diagnoses", "diagnose", "disease", "condition", "syndrome",
        "disorder", "illness", "assessment", "clinical impression", "what is wrong with"
    ]):
        return "diagnosis"

    # Findings Intent
    if any(k in q_lower for k in [
        "findings", "finding", "important findings", "diagnostic findings", "ecg", "ekg",
        "troponin", "tsh", "labs", "laboratory", "investigation", "investigations",
        "imaging", "stress test", "angiography", "ultrasound", "biopsy", "test results"
    ]):
        return "findings"

    # Default to specific_fact for factual clinical inquiries
    return "specific_fact"


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
        relevance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate chunks according to relevance to the query."""
        pass


class LocalSemanticReranker(BaseReranker):
    """
    High-precision hybrid reranker combining:
    1. Dense Vector Similarity (ChromaDB cosine similarity from BGE-M3 / all-MiniLM).
    2. BM25 / Exact Lexical Keyword Overlap with Normalized Stemming.
    3. Section & Clinical Intent Alignment.
    4. Exact Factual Pattern & Compound Entity Verification.
    5. Configurable relevance threshold gating (rejects unmentioned facts & out-of-domain queries).
    """

    def __init__(self, default_threshold: Optional[float] = None):
        self.default_threshold = (
            default_threshold
            if default_threshold is not None
            else getattr(settings, "MIN_RELEVANCE_SCORE", 0.35)
        )

    def extract_and_normalize_query_tokens(self, query: str) -> List[str]:
        """Extract words, remove stopwords, and apply synonym normalization."""
        raw_tokens = re.findall(r"\b\w+\b", query.lower())
        tokens = [
            normalize_token(w)
            for w in raw_tokens
            if w not in STOP_WORDS and len(w) > 1
        ]
        return tokens

    def check_required_entities(self, query_lower: str, content_lower: str, section_lower: str) -> bool:
        """
        Entity gating: if the query targets a specific entity (e.g. glucose, phone, address, hdl, ldl),
        the chunk MUST contain evidence of that entity or belong to the direct entity section.
        Returns False if the chunk fails the required entity check (meaning it should NOT match).
        """
        for trigger, required_keywords in REQUIRED_ENTITY_KEYWORDS.items():
            if trigger in query_lower:
                # Check if chunk contains at least one of the required keywords
                has_keyword = any(kw in content_lower for kw in required_keywords)
                has_section = any(kw in section_lower for kw in required_keywords)
                if not (has_keyword or has_section):
                    return False
        return True

    def compute_lexical_score(self, query_tokens: List[str], text: str, section: str) -> Tuple[float, float]:
        """
        Compute BM25-proxy term frequency, section boost, and token coverage ratio.
        Returns: (lexical_score, coverage_ratio)
        """
        if not query_tokens:
            return 0.0, 0.0

        content_words = [normalize_token(w) for w in re.findall(r"\b\w+\b", text.lower())]
        section_words = [normalize_token(w) for w in re.findall(r"\b\w+\b", section.lower())]
        total_content_words = max(1, len(content_words))

        matched_tokens = 0
        term_frequency_sum = 0.0

        for token in query_tokens:
            count = content_words.count(token)
            if token in section_words:
                count += 4  # Strong boost for section header match

            if count > 0:
                matched_tokens += 1
                tf = count / total_content_words
                term_frequency_sum += math.log1p(tf * 100)

        coverage_ratio = matched_tokens / len(query_tokens)
        bm25_proxy = min(1.0, term_frequency_sum * 0.45)
        lexical_score = 0.55 * coverage_ratio + 0.45 * bm25_proxy
        return min(1.0, lexical_score), coverage_ratio

    def compute_intent_score(self, query_tokens: List[str], content_lower: str, section_lower: str) -> float:
        """Evaluate alignment between query domain intent and chunk section/content."""
        if not query_tokens:
            return 0.0

        matched_intents = 0
        total_intent_score = 0.0

        for intent_name, intent_data in INTENT_DEFINITIONS.items():
            query_has_intent = any(k in query_tokens for k in intent_data["keywords"])
            if query_has_intent:
                matched_intents += 1
                intent_val = 0.0

                # 1. Primary Section match bonus
                if any(sec in section_lower for sec in intent_data["primary_sections"]):
                    intent_val += 0.65
                elif any(sec in section_lower for sec in intent_data.get("secondary_sections", set())):
                    intent_val += 0.30

                # 2. Content pattern match bonus
                for pat in intent_data["content_patterns"]:
                    if re.search(pat, content_lower):
                        intent_val += 0.35
                        break

                total_intent_score += min(1.0, intent_val)

        if matched_intents == 0:
            return 0.0

        return min(1.0, total_intent_score / matched_intents)

    def compute_exact_fact_score(self, query_lower: str, query_tokens: List[str], content_lower: str, section_lower: str) -> float:
        """Check for specific factual queries like age, blood pressure, heart rate, LDL, HDL, ECG."""
        score = 0.0

        # Specialty query
        if "specialty" in query_tokens:
            if "specialty:" in content_lower or "speciality:" in content_lower:
                score += 0.70
            if "document overview" in section_lower or "overview" in section_lower:
                score += 0.30

        # Age query
        if "age" in query_tokens or "old" in query_tokens:
            if re.search(r"\b\d{1,3}\s*-\s*year\s*-\s*old\b", content_lower) or re.search(r"\b\d{1,3}\s*yo\b", content_lower) or re.search(r"\b\d{1,3}\s*years?\s*old\b", content_lower):
                score += 0.70
            if "history of present illness" in section_lower:
                score += 0.30

        # Blood pressure query
        if "blood pressure" in query_lower or "bp" in query_tokens:
            if re.search(r"blood pressure\s*\d{2,3}/\d{2,3}", content_lower) or re.search(r"\b\d{2,3}/\d{2,3}\b", content_lower):
                score += 0.70
            if "physical examination" in section_lower or "vital signs" in section_lower:
                score += 0.30

        # Heart rate query
        if "heart rate" in query_lower or "pulse" in query_tokens or "bpm" in query_tokens:
            if re.search(r"heart rate\s*\d{2,3}", content_lower) or "bpm" in content_lower or "pulse" in content_lower:
                score += 0.70
            if "physical examination" in section_lower or "vital signs" in section_lower:
                score += 0.30

        # LDL level query
        if "ldl" in query_lower:
            if "ldl" in content_lower:
                score += 0.75
            if "diagnostic findings" in section_lower or "labs" in section_lower:
                score += 0.25

        # HDL level query
        if "hdl" in query_lower:
            if "hdl" in content_lower:
                score += 0.75
            if "diagnostic findings" in section_lower or "labs" in section_lower:
                score += 0.25

        # Edema query
        if "edema" in query_tokens:
            if "edema" in content_lower:
                score += 0.70
            if "physical examination" in section_lower:
                score += 0.30

        # Diagnosis query
        if "diagnosis" in query_tokens or "disease" in query_tokens:
            if "assessment" in section_lower or "impression" in section_lower:
                score += 0.60
            if re.search(r"(stable angina|coronary artery disease|hypothyroidism|hashimoto)", content_lower):
                score += 0.40

        # ECG query
        if "ecg" in query_tokens or "ekg" in query_tokens:
            if "ecg" in content_lower or "ekg" in content_lower or "sinus rhythm" in content_lower:
                score += 0.60
            if "diagnostic findings" in section_lower:
                score += 0.40

        return min(1.0, score)

    def compute_chunk_score(self, query: str, chunk: Dict[str, Any]) -> float:
        """
        Compute multi-signal relevance score:
        - Dense Vector Similarity
        - Lexical BM25 Score
        - Section Intent Alignment
        - Exact Factual Match
        - Strict Entity Gating Suppression
        """
        content = chunk.get("content") or ""
        section = chunk.get("section_title") or ""
        dense_sim = float(chunk.get("similarity_score", 0.0))

        content_lower = content.lower()
        section_lower = section.lower()
        query_lower = query.lower().strip()

        # Step 1: Strict Entity Gating Check
        # If query asks for specific entity (e.g. glucose, phone, address) and chunk lacks it:
        if not self.check_required_entities(query_lower, content_lower, section_lower):
            # Suppress completely to avoid returning false-positive chunks
            return 0.0

        tokens = self.extract_and_normalize_query_tokens(query)
        if not tokens:
            return 0.0

        lexical_score, coverage_ratio = self.compute_lexical_score(tokens, content_lower, section_lower)
        intent_score = self.compute_intent_score(tokens, content_lower, section_lower)
        exact_fact_score = self.compute_exact_fact_score(query_lower, tokens, content_lower, section_lower)

        # Out-of-domain / zero-evidence suppression:
        if lexical_score < 0.08 and intent_score == 0.0 and exact_fact_score == 0.0:
            combined = dense_sim * 0.10
        elif coverage_ratio < 0.30 and intent_score == 0.0 and exact_fact_score == 0.0:
            combined = dense_sim * 0.15 + lexical_score * 0.15
        else:
            combined = (
                0.35 * dense_sim +
                0.25 * lexical_score +
                0.25 * intent_score +
                0.15 * exact_fact_score
            )

        return round(min(1.0, max(0.0, combined)), 4)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
        relevance_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        if not chunks or not query or not query.strip():
            return []

        # Immediately reject conversational / greeting / out-of-domain queries
        if is_greeting_or_nonsense(query) or is_out_of_domain(query):
            logger.info(f"Query '{query}' classified as conversational/out-of-domain; returning 0 results.")
            return []

        threshold = (
            relevance_threshold
            if relevance_threshold is not None
            else self.default_threshold
        )

        scored_chunks = []
        seen_content_signatures = set()

        for chunk in chunks:
            chunk_copy = dict(chunk)
            rerank_score = self.compute_chunk_score(query, chunk_copy)
            chunk_copy["relevance_score"] = rerank_score
            chunk_copy["similarity_score"] = float(chunk_copy.get("similarity_score", 0.0))
            chunk_copy["distance"] = float(chunk_copy.get("distance", 0.0))

            # Filter by relevance threshold
            if rerank_score >= threshold:
                content_sig = (chunk_copy.get("document_id"), chunk_copy.get("chunk_index"))
                if content_sig not in seen_content_signatures:
                    seen_content_signatures.add(content_sig)
                    scored_chunks.append(chunk_copy)

        # Sort descending by relevance_score, tie-breaking with dense similarity_score
        scored_chunks.sort(key=lambda x: (x["relevance_score"], x["similarity_score"]), reverse=True)
        return scored_chunks[:top_k]


_reranker_instance: Optional[BaseReranker] = None


def get_reranker() -> BaseReranker:
    """Dependency injection factory for query reranker."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = LocalSemanticReranker()
    return _reranker_instance
