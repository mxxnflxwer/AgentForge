import logging
from typing import List, Optional

from app.services.agent_evo.workflow import AgentWorkflow, get_baseline_workflow

logger = logging.getLogger("agentforge.services.agent_evo.candidate_generator")


class CandidateGenerator:
    """
    Generates structured, executable candidate workflows mutated along valid dimensions.
    Initial implementation produces 3–5 diverse candidates targeting distinct performance/cost trade-offs.
    """

    @classmethod
    def generate_initial_candidates(
        cls,
        baseline: Optional[AgentWorkflow] = None,
        count: int = 5,
    ) -> List[AgentWorkflow]:
        """
        Generate candidate workflows from the baseline workflow.
        """
        base = baseline or get_baseline_workflow()
        candidates: List[AgentWorkflow] = []

        # Candidate 1: High-Precision Retrieval (Fewer chunks, higher relevance threshold)
        c1 = base.clone(
            new_id="cand_high_precision",
            new_name="High-Precision Retrieval",
            new_description="Restricts retrieval to top_k=3 with stricter relevance threshold 0.45 to reduce noise and input tokens.",
        )
        c1.retrieval.top_k = 3
        c1.retrieval.similarity_threshold = 0.45
        candidates.append(c1)

        # Candidate 2: Broad-Coverage Retrieval (More chunks, lower relevance threshold)
        c2 = base.clone(
            new_id="cand_broad_coverage",
            new_name="Broad-Coverage Retrieval",
            new_description="Expands retrieval to top_k=8 with lower threshold 0.25 to capture multi-section context.",
        )
        c2.retrieval.top_k = 8
        c2.retrieval.similarity_threshold = 0.25
        candidates.append(c2)

        # Candidate 3: Concise Clinical Prompt (Lower output token cost)
        c3 = base.clone(
            new_id="cand_concise_prompt",
            new_name="Concise Prompt Optimization",
            new_description="Uses concise medical prompt template and top_k=4 to lower input/output token cost while maintaining grounding.",
        )
        c3.retrieval.top_k = 4
        c3.prompt.template_name = "concise_medical"
        candidates.append(c3)

        # Candidate 4: Alternative Open Reasoning Model (GPT-OSS 120B)
        c4 = base.clone(
            new_id="cand_gpt_oss_reasoning",
            new_name="GPT-OSS 120B Reasoning",
            new_description="Evaluates open-weights 120B reasoning model hosted on Hugging Face with baseline retrieval.",
        )
        c4.model.provider = "Hugging Face"
        c4.model.model_id = "openai/gpt-oss-120b"
        candidates.append(c4)

        # Candidate 5: Alternative High-Capacity Model (Qwen 3.6 27B)
        c5 = base.clone(
            new_id="cand_qwen_capacity",
            new_name="Qwen 3.6 27B Synthesis",
            new_description="Evaluates Qwen 3.6 27B model via OpenRouter for dense clinical synthesis.",
        )
        c5.model.provider = "OpenRouter"
        c5.model.model_id = "qwen/qwen3.6-27b"
        candidates.append(c5)

        # Return requested count
        selected = candidates[: max(1, min(count, len(candidates)))]
        logger.info(f"Generated {len(selected)} candidate workflows from baseline.")
        return selected


def generate_candidates(
    baseline: Optional[AgentWorkflow] = None,
    count: int = 5,
) -> List[AgentWorkflow]:
    """Convenience helper to generate candidate workflows."""
    return CandidateGenerator.generate_initial_candidates(baseline=baseline, count=count)
