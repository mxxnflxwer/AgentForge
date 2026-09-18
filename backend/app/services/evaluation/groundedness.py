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
    supported_claims_list: List[str] = Field(default_factory=list, description="List of supported factual claims")
    unsupported_claims_list: List[str] = Field(default_factory=list, description="List of unsupported factual claims")
    total_claims: int = Field(..., description="Total evaluated factual claims")
    claim_details: List[ClaimDetail] = Field(default_factory=list, description="Claim-level verification details")


# Conversational and boilerplate phrases to ignore during factual claim extraction
BOILERPLATE_PATTERNS = [
    r"^(?:based on the (?:provided|uploaded)?\s*(?:document|context|report|clinical record)?[,:]?\s*)+",
    r"^(?:according to the (?:provided|uploaded)?\s*(?:document|context|report|clinical record)?[,:]?\s*)+",
    r"^(?:the\s+)?(?:clinical\s+)?(?:diagnosis|assessment|plan|summary|treatment|impression|findings?|recommendations?|physical examination|overview|chief complaint|vitals)\s*(?:is|are)?\s*:\s*",
    r"^(?:summary|assessment|plan|physical examination|findings|vitals|impression|diagnosis|treatment|medications?)\s*:\s*",
    r"^in summary[,:]\s*",
    r"^please note that[,:]?\s*",
    r"^the requested information was not found.*",
    r"^information (was|is) not found.*",
    r"^this analysis is for research.*",
]

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "at", "by", "from",
    "for", "with", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "of", "in", "on", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "can", "could", "should",
    "would", "patient", "report", "document", "provided", "uploaded", "clinical",
}


CATEGORY_HEADER_PATTERN = (
    r"^(?:medications?|diagnostic\s+(?:step|findings?)|lifestyle(?:\s+and\s+treatment\s+measures)?|"
    r"follow-?up|assessment|plan|summary|physical\s+examination|vital\s+signs?|vitals|impression|"
    r"diagnosis|treatment(?:\s+plan)?|recommendations?|overview|history(?:\s+of\s+present\s+illness)?|"
    r"chief\s+complaint)\s*:?$"
)


def clean_text(text: str) -> str:
    """Normalize text and Unicode whitespace for evaluation."""
    return re.sub(r"[\s\u202f\xa0]+", " ", (text or "").strip())


def extract_factual_claims(text: str) -> List[str]:
    """
    Extract discrete factual assertions from generated text.
    Filters out Markdown headings, section labels, numbering, conversational filler, and disclaimers.
    """
    if not text or not text.strip():
        return []

    # 1. Clean markdown headers and formatting
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Remove bullet prefix from line if checking header
        l_no_bullet = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", l).strip()
        l_no_md = re.sub(r"\*\*([^*]+)\*\*", r"\1", l_no_bullet).strip()
        l_no_md = re.sub(r"\*([^*]+)\*", r"\1", l_no_md).strip()
        # Skip pure section or category header lines
        if re.match(CATEGORY_HEADER_PATTERN, l_no_md, re.IGNORECASE):
            continue
        # Skip pure markdown headers like '# Assessment' or '### Plan:'
        if re.match(r"^#{1,6}\s+", l):
            l = re.sub(r"^#{1,6}\s+", "", l).strip()
            if re.match(CATEGORY_HEADER_PATTERN, l, re.IGNORECASE):
                continue
        # Strip bold/italic markdown
        l = re.sub(r"\*\*([^*]+)\*\*", r"\1", l)
        l = re.sub(r"\*([^*]+)\*", r"\1", l)
        cleaned_lines.append(l)

    cleaned_text = " ".join(cleaned_lines)
    cleaned_text = re.sub(r"[\u202f\xa0]", " ", cleaned_text)

    # 2. Check safe not-found response
    if "not found in the uploaded document" in cleaned_text.lower() or "information was not found" in cleaned_text.lower():
        return []

    # 3. Split by bullet/number markers and sentence boundaries
    raw_segments = re.split(r"(?:^|\s+)(?:[-*•]|\d+[.)])\s+", cleaned_text)

    sub_segments = []
    for seg in raw_segments:
        if not seg.strip():
            continue
        # Split by sentence terminators
        sentences = re.split(r"(?<=[.!?])\s+", seg)
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            # Split coordinate clauses joined by semicolon or coordinate conjunction + subject
            clauses = re.split(
                r";\s*|\s+(?:and|but)\s+(?=(?:the\s+patient|they|she|he|patient|patient's)\s+)",
                sent,
                flags=re.IGNORECASE,
            )
            for c in clauses:
                if c and c.strip():
                    sub_segments.append(c.strip())

    claims: List[str] = []
    for s in sub_segments:
        s = s.strip()
        if not s:
            continue

        # Strip boilerplate and carrier prefixes repeatedly
        changed = True
        while changed:
            changed = False
            for pat in BOILERPLATE_PATTERNS:
                new_s = re.sub(pat, "", s, flags=re.IGNORECASE).strip()
                if new_s != s:
                    s = new_s
                    changed = True

        # Strip remaining list numbers or bullets
        s = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", s).strip()

        # Reject labels ending with colon or disclaimers or category headers or too short
        if not s or len(s) < 6:
            continue
        if s.endswith(":") or re.match(CATEGORY_HEADER_PATTERN, s, re.IGNORECASE):
            continue
        if re.match(r"^(?:the\s+)?(?:clinical\s+)?(?:diagnosis|assessment|plan|summary|treatment)\s+is\s*:?$", s, re.IGNORECASE):
            continue
        if "research and demonstration purposes" in s.lower() or "substitute for professional" in s.lower():
            continue

        claims.append(s)

    return claims


import difflib

def token_matches_set(token: str, token_set: Set[str]) -> bool:
    """
    Check if a token matches any token in the target set via exact match,
    common morphological suffix stripping, shared root prefix, or high character similarity.
    Handles spelling variants (e.g., 'speciality' vs 'specialty') and inflections ('diagnosed' vs 'diagnosis').
    """
    if token in token_set:
        return True

    # Suffix stripping normalization
    t_clean = re.sub(r"(?:s|es|ed|ing|ly|al|ic|ion|ity|y|ive)$", "", token)
    for other in token_set:
        o_clean = re.sub(r"(?:s|es|ed|ing|ly|al|ic|ion|ity|y|ive)$", "", other)
        if t_clean and o_clean and t_clean == o_clean:
            return True

        # Common root prefix for words with length >= 4
        if len(token) >= 4 and len(other) >= 4:
            prefix_len = min(min(len(token), len(other)), 5)
            if token[:prefix_len] == other[:prefix_len]:
                return True

        # Character similarity for spelling variants (e.g., speciality / specialty)
        if len(token) >= 5 and len(other) >= 5:
            sim = difflib.SequenceMatcher(None, token, other).ratio()
            if sim >= 0.80:
                return True

    return False


DISCOURSE_WORDS = {
    "recorded", "examination", "physical", "shown", "found", "noted", "indicated",
    "listed", "mentioned", "section", "chart", "report", "documented", "stated",
    "states", "presents", "presented", "described", "observed", "result", "results",
    "evaluation", "assessment", "plan", "history", "review", "overview", "clinical",
    "given", "based", "according", "value", "values", "measured", "level", "levels",
    "current", "present", "reveals", "revealed", "noting", "per", "regarding", "note",
    "patient's", "patient", "document", "provided", "uploaded", "taken", "prescribed",
    "ordered", "administered", "started", "recommended", "advised", "directed",
    "instructed", "use", "used", "include", "includes", "including", "calls",
}

ALL_IGNORE_WORDS = STOP_WORDS | DISCOURSE_WORDS


def extract_informative_keywords(text: str) -> Set[str]:
    """Extract contentful keywords excluding stopwords and discourse carrier words."""
    # Normalize Unicode hyphens and whitespace
    norm_text = re.sub(r"[\u2011\u2012\u2013\u2014]", "-", (text or "").lower())
    norm_text = re.sub(r"[\u202f\xa0\u200b]", " ", norm_text)
    # Normalize measurement units e.g. 40mg -> 40 mg, 118/76mmHg -> 118/76 mmHg
    normalized = re.sub(r"(\d+)\s*([a-zA-Z]+)", r"\1 \2", norm_text)
    words = re.findall(r"[a-zA-Z0-9_/.-]+", normalized)
    clean_words = set()
    for w in words:
        w_strip = w.strip(".,;:!?()[]\"'")
        if w_strip and w_strip not in ALL_IGNORE_WORDS and len(w_strip) > 1:
            clean_words.add(w_strip)
    return clean_words


def extract_numerical_tokens(text: str) -> Set[str]:
    """Extract numerical and measurement tokens (including ratios/fractions like 118/76)."""
    matches = re.findall(r"\b\d+(?:[./]\d+)?\b", (text or "").lower())
    nums = set()
    for m in matches:
        nums.add(m)
        if "/" in m:
            for sub in m.split("/"):
                if sub:
                    nums.add(sub)
    return nums


def evaluate_claim_support(claim: str, context: str) -> ClaimDetail:
    """
    Evaluate whether a single factual claim is semantically supported by the retrieved context.
    1. Checks safe out-of-domain and exact substring match.
    2. Verifies numerical alignment (detects numerical hallucinations e.g. 160/100 vs 118/76).
    3. Performs discourse-filtered entity & morphological overlap across context paragraphs.
    """
    context_lower = (context or "").lower()
    claim_lower = (claim or "").strip().lower()

    # 1. Safe "not found" response handling
    if "not found in the uploaded document" in claim_lower or "information was not found" in claim_lower:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=1.0,
            matched_context_snippet="",
            reason="Safe recognition of missing document context.",
        )

    # 2. Exact substring match
    if claim_lower in context_lower:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=1.0,
            matched_context_snippet=claim,
            reason="Exact factual match found in context.",
        )

    claim_entities = extract_informative_keywords(claim)
    if not claim_entities:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=0.8,
            reason="Non-specific statement supported by general context.",
        )

    # 3. Check numerical and measurement consistency
    claim_nums = extract_numerical_tokens(claim)
    if claim_nums:
        context_nums = extract_numerical_tokens(context)
        missing_nums = claim_nums - context_nums
        really_missing = set()
        for num in missing_nums:
            if "/" in num:
                parts = num.split("/")
                if not all(p in context_nums for p in parts):
                    really_missing.add(num)
            else:
                really_missing.add(num)
        if really_missing:
            return ClaimDetail(
                claim=claim,
                status=ClaimStatus.UNSUPPORTED,
                confidence=0.95,
                reason=f"Numerical values {really_missing} not found in retrieved context.",
            )

    # 4. Paragraph-level entity & morphological overlap
    paragraphs = [p.strip().lower() for p in context.split("\n") if p.strip()]
    best_claim_ratio = 0.0
    best_para_coverage = 0.0
    best_matching_snippet = None

    for para in paragraphs:
        if not para:
            continue
        para_entities = extract_informative_keywords(para)
        matched_claim = {k for k in claim_entities if token_matches_set(k, para_entities)}
        claim_ratio = len(matched_claim) / float(len(claim_entities)) if claim_entities else 1.0

        matched_para = {k for k in para_entities if token_matches_set(k, claim_entities)}
        para_cov = len(matched_para) / float(len(para_entities)) if para_entities else 0.0

        if claim_ratio > best_claim_ratio or (claim_ratio == best_claim_ratio and para_cov > best_para_coverage):
            best_claim_ratio = claim_ratio
            best_para_coverage = para_cov
            best_matching_snippet = para[:150]

    # Classification thresholds:
    # >= 50% claim overlap OR >= 60% context coverage (for short target facts) -> SUPPORTED
    # 25% - 49% -> INSUFFICIENT_EVIDENCE
    # < 25% -> UNSUPPORTED
    if best_claim_ratio >= 0.50 or best_para_coverage >= 0.60:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.SUPPORTED,
            confidence=round(max(best_claim_ratio, best_para_coverage), 2),
            matched_context_snippet=best_matching_snippet,
            reason=f"Strong contextual support ({int(best_claim_ratio * 100)}% entity overlap).",
        )
    elif best_claim_ratio >= 0.25:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.INSUFFICIENT_EVIDENCE,
            confidence=round(best_claim_ratio, 2),
            matched_context_snippet=best_matching_snippet,
            reason=f"Partial mention but insufficient evidence ({int(best_claim_ratio * 100)}% overlap).",
        )
    else:
        return ClaimDetail(
            claim=claim,
            status=ClaimStatus.UNSUPPORTED,
            confidence=round(1.0 - best_claim_ratio, 2),
            reason=f"Claim absent from retrieved context (only {int(best_claim_ratio * 100)}% overlap).",
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
    if "not found in the uploaded document" in cleaned_answer.lower() or "information was not found" in cleaned_answer.lower():
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
        # If no distinct factual claims were extracted
        score = 1.0 if cleaned_answer and ("error" not in cleaned_answer.lower() and "no answer" not in cleaned_answer.lower()) else 0.0
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

    supp_list = [c.claim for c in claim_details if c.status == ClaimStatus.SUPPORTED]
    unsupp_list = [c.claim for c in claim_details if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.INSUFFICIENT_EVIDENCE)]

    return GroundednessResult(
        groundedness=groundedness_score,
        supported_claims=supported_count,
        unsupported_claims=unsupported_count,
        supported_claims_list=supp_list,
        unsupported_claims_list=unsupp_list,
        total_claims=total,
        claim_details=claim_details,
    )
