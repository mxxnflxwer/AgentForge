import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from app.services.evaluation.groundedness import (
    ClaimDetail,
    ClaimStatus,
    GroundednessResult,
    calculate_groundedness,
)

logger = logging.getLogger("agentforge.services.evaluation.hallucination")


class HallucinationResult(BaseModel):
    """Hallucination evaluation metrics and catalog of unsupported claims."""
    hallucination_rate: float = Field(..., description="Ratio of unsupported claims: unsupported / total")
    hallucinated_claims: int = Field(..., description="Number of unsupported or hallucinated claims")
    total_claims: int = Field(..., description="Total evaluated claims")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unsupported claim strings")
    claim_details: List[ClaimDetail] = Field(default_factory=list, description="Detailed claim verification status")


def calculate_hallucination_rate(
    generated_answer: str,
    retrieved_context: str,
    groundedness_result: Optional[GroundednessResult] = None,
) -> HallucinationResult:
    """
    Calculate Hallucination Rate based on the proportion of factual claims
    in the generated answer that are unsupported or fabricated relative to the retrieved context.

    Formula:
        Hallucination Rate = unsupported factual claims / total factual claims

    Preserves the list of unsupported claims for the future AgentEvo Optimization Report.
    """
    g_res = groundedness_result or calculate_groundedness(
        generated_answer=generated_answer,
        retrieved_context=retrieved_context,
    )

    if g_res.total_claims == 0:
        return HallucinationResult(
            hallucination_rate=0.0,
            hallucinated_claims=0,
            total_claims=0,
            unsupported_claims=[],
            claim_details=g_res.claim_details,
        )

    unsupported_claims_list = [
        c.claim for c in g_res.claim_details
        if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.INSUFFICIENT_EVIDENCE)
    ]

    rate = round(len(unsupported_claims_list) / float(g_res.total_claims), 4)

    return HallucinationResult(
        hallucination_rate=rate,
        hallucinated_claims=len(unsupported_claims_list),
        total_claims=g_res.total_claims,
        unsupported_claims=unsupported_claims_list,
        claim_details=g_res.claim_details,
    )
