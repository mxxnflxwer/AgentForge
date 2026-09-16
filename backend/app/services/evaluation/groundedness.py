import enum
import logging
import re
from typing import List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger("agentforge.services.evaluation.groundedness")


class ClaimStatus(str, enum.Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ClaimDetail(BaseModel):
    """Detailed evaluation of an individual factual claim."""
    claim: str = Field(..., description="Extracted factual assertion")
    status: ClaimStatus = Field(..., description="'supported', 'unsupported', or 'insufficient_evidence'")
    confidence: float = Field(1.0, description="Verification confidence score [0.0 - 1.0]")
    matched_context_snippet: Optional[str] = Field(None, description="Context snippet supporting the claim if found")
    reason: Optional[str] = Field(None, description="Explanation for claim verification status")


class GroundednessResult(BaseModel):
    """Groundedness evaluation metrics and claim-level breakdown."""
    groundedness: float = Field(..., description="Ratio of supported claims: supported / total")
    supported_claims: int = Field(..., description="Number of supported factual claims")
    unsupported_claims: int = Field(..., description="Number of unsupported or hallucinated claims")
    total_claims: int = Field(..., description="Total evaluated factual claims")
    claim_details: List[ClaimDetail] = Field(default_factory=list, description="Claim-level verification details")


# Conversational and boilerplate phrases to ignore during factual claim extraction
BOILERPLATE_PATTERNS = [
    r"^based on the provided document.*?[,:]\s*",
    r"^according to the (uploaded|provided)?\s*(document|report).*?[,:]\s*",
    r"^the requested information was not found.*",
    r"^information (was|is) not found.*",
    r"^this analysis is for research.*",
    r"^please note that.*",
    r"^in summary[,:]\s*",
]

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "from",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "should",
    "would", "patient", "report", "document", "provided", "uploaded", "clinical",
}


def clean_text(text: str) -> str:
    """Normalize text for evaluation."""
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_factual_claims(text: str) -> List[str]:
    """
    Extract discrete factual assertions from generated text.
    Filters out conversational filler and disclaimers.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Handle standard safe "not found" responses
    if "not found in the uploaded document" in cleaned.lower() or "information was not found" in cleaned.lower():
        return []

    # Split on sentence boundaries and bullet points
    raw_segments = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\n)\s*[-*•\d+.]\s*", cleaned)
    claims: List[str] = []

    for seg in raw_segments:
        s = seg.strip()
        if not s or len(s) < 5:
            continue

        # Strip common boilerplate prefixes
        for pat in BOILERPLATE_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()

        # Ignore remaining boilerplate or disclaimer sentences
        if not s or len(s) < 5:
            continue
        if "research and demonstration purposes" in s.lower() or "substitute for professional" in s.lower():
            continue

        # Split complex compound sentences with multiple distinct medical facts (e.g. "BP is 130/80 and HR is 72")
        sub_clauses = re.split(r";\s*|\.\s+", s)
        for clause in sub_clauses:
            cl = clause.strip()
            if len(cl) >= 6:
                claims.append(cl)

    return claims


def extract_content_keywords(text: str) -> Set[str]:
    """Extract contentful keywords (excluding common stopwords)."""
    words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
    return {w for w in words if w not in STOP_WORDS and len(w) > 1}


def evaluate_claim_support(claim: str, context: str) -> ClaimDetail:
    """
    Evaluate whether a single factual claim is supported by the retrieved context.
    """
    context_lower = context.lower()
    claim_lower = claim.lower()

    # Exact substring match
    if claim_lower in context_lower:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=1.0,
            matched_context_snippet=claim,
            reason="Exact factual match found in context.",
        )

    claim_keywords = extract_content_keywords(claim)
    if not claim_keywords:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=0.8,
            reason="Non-specific statement supported by general context.",
        )

    # Check numerical and entity matches specifically (e.g. blood pressure, dosages, lab values)
    numbers_in_claim = set(re.findall(r"\b\d+(?:[./]\d+)?\b", claim_lower))
    if numbers_in_claim:
        numbers_in_context = set(re.findall(r"\b\d+(?:[./]\d+)?\b", context_lower))
        missing_numbers = numbers_in_claim - numbers_in_context
        if missing_numbers:
            return ClaimDetail(
                claim=claim,
                status=ClaimStatus.UNSUPPORTED,
                confidence=0.9,
                reason=f"Numerical values {missing_numbers} not found in retrieved context.",
            )

    # Context window scanning for keyword density
    paragraphs = [p.strip().lower() for p in context.split("\n") if p.strip()]
    best_overlap_ratio = 0.0
    best_matching_snippet = None

    for para in paragraphs:
        if not para:
            continue
        para_keywords = extract_content_keywords(para)
        overlap = claim_keywords & para_keywords
        ratio = len(overlap) / float(len(claim_keywords))

        if ratio > best_overlap_ratio:
            best_overlap_ratio = ratio
            best_matching_snippet = para[:150]

    # Classification thresholds:
    # >= 60% keyword overlap -> SUPPORTED
    # 35% - 59% -> INSUFFICIENT_EVIDENCE (treated as unsupported in conservative evaluation)
    # < 35% -> UNSUPPORTED
    if best_overlap_ratio >= 0.60:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=round(best_overlap_ratio, 2),
            matched_context_snippet=best_matching_snippet,
            reason=f"Strong contextual support ({int(best_overlap_ratio * 100)}% content overlap).",
        )
    elif best_overlap_ratio >= 0.35:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.INSUFFICIENT_EVIDENCE,
            confidence=round(best_overlap_ratio, 2),
            matched_context_snippet=best_matching_snippet,
            reason=f"Partial mention but insufficient evidence ({int(best_overlap_ratio * 100)}% overlap).",
        )
    else:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.UNSUPPORTED,
            confidence=round(1.0 - best_overlap_ratio, 2),
            reason=f"Claim absent from retrieved context (only {int(best_overlap_ratio * 100)}% overlap).",
        )


def calculate_groundedness(
    generated_answer: str,
    retrieved_context: str,
) -> GroundednessResult:
    """
    Calculate Groundedness score based on whether factual claims in the generated
    answer are supported by the retrieved document context.

    Formula:
        Groundedness = supported factual claims / total factual claims
    """
    cleaned_context = clean_text(retrieved_context)
    cleaned_answer = clean_text(generated_answer)

    # Handle safe "not found" response
    if "not found in the uploaded document" in cleaned_answer.lower():
        return GroundednessResult(
            groundedness=1.0,
            supported_claims=0,
            unsupported_claims=0,
            total_claims=0,
            claim_details=[
                ClaimDetail(
                    claim="Information not found in document",
                    status=ClaimStatus.SUPPORTED,
                    confidence=1.0,
                    reason="Accurate recognition of missing document context.",
                )
            ],
        )

    claims = extract_factual_claims(generated_answer)
    if not claims:
        # If no distinct factual claims were made but text is non-empty
        score = 1.0 if cleaned_answer else 0.0
        return GroundednessResult(
            groundedness=score,
            supported_claims=0,
            unsupported_claims=0,
            total_claims=0,
            claim_details=[],
        )

    claim_details: List[ClaimDetail] = []
    supported_count = 0
    unsupported_count = 0

    for claim in claims:
        detail = evaluate_claim_support(claim, retrieved_context)
        claim_details.append(detail)
        if detail.status == ClaimStatus.SUPPORTED:
            supported_count += 1
        else:
            unsupported_count += 1

    total = len(claims)
    groundedness_score = round(supported_count / float(total), 4) if total > 0 else 1.0

    return GroundednessResult(
        groundedness=groundedness_score,
        supported_claims=supported_count,
        unsupported_claims=unsupported_count,
        total_claims=total,
        claim_details=claim_details,
    )
