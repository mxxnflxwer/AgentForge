import copy
import logging
import random
from typing import Any, Callable, Dict, List, Optional, Set

from app.services.agent_evo.validator import validate_workflow
from app.services.agent_evo.workflow import (
    AgentWorkflow,
    MutationMetadata,
    MutationType,
    get_baseline_workflow,
)

logger = logging.getLogger("agentforge.services.agent_evo.candidate_generator")


def normalize_workflow_config(wf_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract a canonical configuration dictionary from a workflow for strict duplicate detection.
    Excludes identifiers, names, descriptions, and metadata to focus purely on execution semantics.
    """
    return {
        "retrieval": {
            "chunk_size": wf_dict.get("retrieval", {}).get("chunk_size", 1000),
            "chunk_overlap": wf_dict.get("retrieval", {}).get("chunk_overlap", 200),
            "top_k": wf_dict.get("retrieval", {}).get("top_k", 5),
            "similarity_threshold": round(float(wf_dict.get("retrieval", {}).get("similarity_threshold", 0.35)), 3),
            "reranking": bool(wf_dict.get("retrieval", {}).get("reranking", True)),
        },
        "model": {
            "provider": str(wf_dict.get("model", {}).get("provider", "Google")),
            "model_id": str(wf_dict.get("model", {}).get("model_id", "gemini-3.5-flash-lite")),
            "temperature": round(float(wf_dict.get("model", {}).get("temperature", 0.1)), 2),
            "max_tokens": int(wf_dict.get("model", {}).get("max_tokens", 1024)),
        },
        "prompt": {
            "template_name": str(wf_dict.get("prompt", {}).get("template_name", "default_medical")),
            "custom_instructions": (wf_dict.get("prompt", {}).get("custom_instructions") or "").strip() or None,
        },
        "execution": {
            "timeout_seconds": float(wf_dict.get("execution", {}).get("timeout_seconds", 30.0)),
            "context_compression": bool(wf_dict.get("execution", {}).get("context_compression", False)),
        },
    }


class CandidateGenerator:
    """
    AgentEvo Evolutionary Exploration and Refinement Candidate Generator.
    Expands and refines the workflow search space across 6 key mutation categories:
    1. PROMPT (templates, instruction style, grounding constraints)
    2. RETRIEVAL (top_k, similarity_threshold, chunk_size, chunk_overlap, reranking)
    3. MODEL (model_id, provider, temperature, max_tokens)
    4. OPERATOR (context compression, extractive filter, execution constraints)
    5. MIXED (composite mutations across multiple dimensions)
    6. REFINEMENT (safe, deterministic simplification and cost reduction from an existing parent)
    """

    # =========================================================================
    # Phase 7.2 Exploration Mutations
    # =========================================================================

    @classmethod
    def _mutate_prompt_concise(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_prompt_concise_{rand.randint(100, 999)}",
            new_name="Concise Grounded Prompt",
            new_description="Replaced default prompt with concise medical template to minimize token usage while maintaining grounding.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.PROMPT,
                parent_workflow_id=base.workflow_id,
                mutation_description="Replaced default prompt with concise medical template to minimize token usage while maintaining grounding.",
                generation_metadata={
                    "mutated_field": "prompt.template_name",
                    "previous_value": base.prompt.template_name,
                    "new_value": "concise_medical",
                },
            ),
        )
        cand.prompt.template_name = "concise_medical"
        return cand

    @classmethod
    def _mutate_prompt_detailed_clinical(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_prompt_detailed_{rand.randint(100, 999)}",
            new_name="Detailed Clinical Synthesis Prompt",
            new_description="Applied detailed clinical prompt template requiring structured medical synthesis and explicit evidence citations.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.PROMPT,
                parent_workflow_id=base.workflow_id,
                mutation_description="Applied detailed clinical prompt template requiring structured medical synthesis and explicit evidence citations.",
                generation_metadata={
                    "mutated_field": "prompt.template_name",
                    "previous_value": base.prompt.template_name,
                    "new_value": "detailed_clinical",
                },
            ),
        )
        cand.prompt.template_name = "detailed_clinical"
        return cand

    @classmethod
    def _mutate_prompt_strict_grounding(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        custom_inst = (
            "Strict Grounding Instruction: Answer strictly using facts stated in the retrieved context. "
            "Do not infer, speculate, or extrapolate clinical findings beyond what is explicitly recorded."
        )
        cand = base.clone(
            new_id=f"cand_prompt_strict_{rand.randint(100, 999)}",
            new_name="Strict Grounding Constraint",
            new_description="Injected rigorous grounding constraints into instructions to minimize hallucination risk.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.PROMPT,
                parent_workflow_id=base.workflow_id,
                mutation_description="Injected rigorous grounding constraints into instructions to minimize hallucination risk.",
                generation_metadata={
                    "mutated_field": "prompt.custom_instructions",
                    "instruction_length_chars": len(custom_inst),
                },
            ),
        )
        cand.prompt.custom_instructions = custom_inst
        return cand

    @classmethod
    def _mutate_retrieval_high_precision(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_retrieval_precision_{rand.randint(100, 999)}",
            new_name="High-Precision Retrieval",
            new_description="Restricted retrieval to top_k=3 with stricter relevance threshold 0.45 to filter low-confidence chunks.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.RETRIEVAL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Restricted retrieval to top_k=3 with stricter relevance threshold 0.45 to filter low-confidence chunks.",
                generation_metadata={
                    "top_k": 3,
                    "similarity_threshold": 0.45,
                },
            ),
        )
        cand.retrieval.top_k = 3
        cand.retrieval.similarity_threshold = 0.45
        return cand

    @classmethod
    def _mutate_retrieval_broad_coverage(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_retrieval_broad_{rand.randint(100, 999)}",
            new_name="Broad-Coverage Retrieval",
            new_description="Expanded retrieval to top_k=8 with lower threshold 0.25 to maximize multi-section context capture.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.RETRIEVAL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Expanded retrieval to top_k=8 with lower threshold 0.25 to maximize multi-section context capture.",
                generation_metadata={
                    "top_k": 8,
                    "similarity_threshold": 0.25,
                },
            ),
        )
        cand.retrieval.top_k = 8
        cand.retrieval.similarity_threshold = 0.25
        return cand

    @classmethod
    def _mutate_retrieval_compact_chunking(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_retrieval_chunking_{rand.randint(100, 999)}",
            new_name="Fine-Grained Chunking Retrieval",
            new_description="Configured compact chunk size (500 tokens) with 100-token overlap and top_k=6. (Applies during document indexing; queries search active index).",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.RETRIEVAL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Configured compact chunk size (500 tokens) with 100-token overlap and top_k=6. (Applies during document indexing; queries search active index).",
                generation_metadata={
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "top_k": 6,
                    "indexing_parameter": True,
                },
            ),
        )
        cand.retrieval.chunk_size = 500
        cand.retrieval.chunk_overlap = 100
        cand.retrieval.top_k = 6
        return cand

    @classmethod
    def _mutate_model_qwen(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_model_qwen_{rand.randint(100, 999)}",
            new_name="Qwen 3.6 27B Synthesis",
            new_description="Evaluates high-capacity Qwen 3.6 27B model hosted via OpenRouter for dense medical synthesis.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.MODEL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Evaluates high-capacity Qwen 3.6 27B model hosted via OpenRouter for dense medical synthesis.",
                generation_metadata={
                    "provider": "OpenRouter",
                    "model_id": "qwen/qwen3.6-27b",
                    "temperature": 0.1,
                },
            ),
        )
        cand.model.provider = "OpenRouter"
        cand.model.model_id = "qwen/qwen3.6-27b"
        cand.model.temperature = 0.1
        return cand

    @classmethod
    def _mutate_model_gemma(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_model_gemma_{rand.randint(100, 999)}",
            new_name="Gemma 4 26B A4B Synthesis",
            new_description="Evaluates Google Gemma 4 26B A4B open-weights model hosted via OpenRouter free tier.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.MODEL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Evaluates Google Gemma 4 26B A4B open-weights model hosted via OpenRouter free tier.",
                generation_metadata={
                    "provider": "OpenRouter",
                    "model_id": "google/gemma-4-26b-a4b-it:free",
                    "temperature": 0.1,
                },
            ),
        )
        cand.model.provider = "OpenRouter"
        cand.model.model_id = "google/gemma-4-26b-a4b-it:free"
        cand.model.temperature = 0.1
        return cand

    @classmethod
    def _mutate_model_gpt_oss(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """Legacy mutation method pointing to Gemma 4 26B A4B for backward compatibility."""
        return cls._mutate_model_gemma(base, rand)

    @classmethod
    def _mutate_model_gemini_low_temp(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_model_deterministic_{rand.randint(100, 999)}",
            new_name="Deterministic Gemini 3.5 Flash-Lite",
            new_description="Adjusted Gemini 3.5 Flash-Lite to 0.0 temperature and 512 max tokens for deterministic clinical answers.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.MODEL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Adjusted Gemini 3.5 Flash-Lite to 0.0 temperature and 512 max tokens for deterministic clinical answers.",
                generation_metadata={
                    "provider": "Google",
                    "model_id": "gemini-3.5-flash-lite",
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            ),
        )
        cand.model.temperature = 0.0
        cand.model.max_tokens = 512
        return cand

    @classmethod
    def _mutate_operator_context_compression(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_operator_compress_{rand.randint(100, 999)}",
            new_name="Context Compression Operator",
            new_description="Activated context compression operator to filter redundant retrieved chunk sentences before prompt injection.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.OPERATOR,
                parent_workflow_id=base.workflow_id,
                mutation_description="Activated context compression operator to filter redundant retrieved chunk sentences before prompt injection.",
                generation_metadata={
                    "operator": "context_compression",
                    "context_compression_enabled": True,
                    "top_k": 6,
                },
            ),
        )
        cand.execution.context_compression = True
        cand.retrieval.top_k = 6
        return cand

    @classmethod
    def _mutate_operator_strict_filter(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_operator_filter_{rand.randint(100, 999)}",
            new_name="Extractive Filter & Rerank Operator",
            new_description="Combined context compression operator with strict threshold (0.40) and active reranking for high signal-to-noise ratio.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.OPERATOR,
                parent_workflow_id=base.workflow_id,
                mutation_description="Combined context compression operator with strict threshold (0.40) and active reranking for high signal-to-noise ratio.",
                generation_metadata={
                    "operator": "context_compression + reranking",
                    "context_compression": True,
                    "similarity_threshold": 0.40,
                    "reranking": True,
                },
            ),
        )
        cand.execution.context_compression = True
        cand.retrieval.similarity_threshold = 0.40
        cand.retrieval.reranking = True
        return cand

    @classmethod
    def _mutate_mixed_cost_optimized(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_mixed_cost_{rand.randint(100, 999)}",
            new_name="Cost-Optimized Composite",
            new_description="Composite mutation: concise medical prompt + top_k=3 + context compression to maximize query efficiency.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.MIXED,
                parent_workflow_id=base.workflow_id,
                mutation_description="Composite mutation: concise medical prompt + top_k=3 + context compression to maximize query efficiency.",
                generation_metadata={
                    "categories_mutated": ["prompt", "retrieval", "operator"],
                    "template_name": "concise_medical",
                    "top_k": 3,
                    "context_compression": True,
                },
            ),
        )
        cand.prompt.template_name = "concise_medical"
        cand.retrieval.top_k = 3
        cand.execution.context_compression = True
        return cand

    @classmethod
    def _mutate_mixed_high_recall_synthesis(cls, base: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        cand = base.clone(
            new_id=f"cand_mixed_synthesis_{rand.randint(100, 999)}",
            new_name="High-Recall Synthesis Composite",
            new_description="Composite mutation: Qwen 3.6 27B + detailed clinical prompt + top_k=7 for deep clinical grounding.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.MIXED,
                parent_workflow_id=base.workflow_id,
                mutation_description="Composite mutation: Qwen 3.6 27B + detailed clinical prompt + top_k=7 for deep clinical grounding.",
                generation_metadata={
                    "categories_mutated": ["model", "prompt", "retrieval"],
                    "model_id": "qwen/qwen3.6-27b",
                    "template_name": "detailed_clinical",
                    "top_k": 7,
                },
            ),
        )
        cand.model.provider = "OpenRouter"
        cand.model.model_id = "qwen/qwen3.6-27b"
        cand.prompt.template_name = "detailed_clinical"
        cand.retrieval.top_k = 7
        return cand

    # =========================================================================
    # Phase 7.3 Refinement Operators (Simplification & Cost Optimization)
    # =========================================================================

    @classmethod
    def _refine_retrieval_reduction(cls, parent: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """
        Refinement Operator A: Retrieval Reduction.
        Reduces excessive top_k (e.g. 8->4, 5->3, 4->3, 3->2) to remove redundancy while preserving grounding.
        """
        curr_k = parent.retrieval.top_k
        if curr_k <= 2:
            return None  # Cannot reduce below safe operational minimum

        new_k = max(2, curr_k - 2 if curr_k > 4 else curr_k - 1)
        if new_k == curr_k:
            return None

        rationale = f"Reduced top_k from {curr_k} to {new_k} to remove retrieval redundancy while preserving grounding."
        cand = parent.clone(
            new_id=f"cand_refine_retrieval_{rand.randint(100, 999)}",
            new_name="Refined: Retrieval Reduction",
            new_description=rationale,
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.REFINEMENT,
                parent_workflow_id=parent.workflow_id,
                mutation_description=rationale,
                generation_metadata={
                    "refinement_operation": "retrieval_reduction",
                    "rationale": rationale,
                    "fields_changed": ["retrieval.top_k"],
                    "before_configuration": {"top_k": curr_k},
                    "after_configuration": {"top_k": new_k},
                },
            ),
        )
        cand.retrieval.top_k = new_k
        return cand

    @classmethod
    def _refine_chunk_reduction(cls, parent: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """
        Refinement Operator B: Chunk Size Reduction.
        Reduces unnecessarily large chunk sizes while preserving overlap validity constraints.
        """
        curr_size = parent.retrieval.chunk_size
        if curr_size <= 500:
            return None  # Already at compact granular minimum

        if curr_size >= 1500:
            new_size = 1000
            new_overlap = min(parent.retrieval.chunk_overlap, 200)
        elif curr_size >= 1000:
            new_size = 750
            new_overlap = min(parent.retrieval.chunk_overlap, 150)
        else:
            new_size = 500
            new_overlap = min(parent.retrieval.chunk_overlap, 100)

        # Validate chunk_overlap < chunk_size constraint
        if new_overlap >= new_size:
            new_overlap = max(50, new_size // 5)

        rationale = f"Configured target chunk_size={new_size} tokens (overlap: {new_overlap}). (Note: Applies during document indexing; query runtime searches current vector index)."
        cand = parent.clone(
            new_id=f"cand_refine_chunk_{rand.randint(100, 999)}",
            new_name="Refined: Granular Chunk Size",
            new_description=rationale,
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.REFINEMENT,
                parent_workflow_id=parent.workflow_id,
                mutation_description=rationale,
                generation_metadata={
                    "refinement_operation": "chunk_reduction",
                    "rationale": rationale,
                    "fields_changed": ["retrieval.chunk_size", "retrieval.chunk_overlap"],
                    "before_configuration": {"chunk_size": curr_size, "chunk_overlap": parent.retrieval.chunk_overlap},
                    "after_configuration": {"chunk_size": new_size, "chunk_overlap": new_overlap},
                },
            ),
        )
        cand.retrieval.chunk_size = new_size
        cand.retrieval.chunk_overlap = new_overlap
        return cand

    @classmethod
    def _refine_context_compression(cls, parent: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """
        Refinement Operator C: Context Compression.
        Activates runtime context compression operator if currently disabled.
        """
        if parent.execution.context_compression is True:
            return None  # Already active

        rationale = "Enabled context compression to reduce redundant retrieved context before LLM generation."
        cand = parent.clone(
            new_id=f"cand_refine_compress_{rand.randint(100, 999)}",
            new_name="Refined: Context Compression",
            new_description=rationale,
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.REFINEMENT,
                parent_workflow_id=parent.workflow_id,
                mutation_description=rationale,
                generation_metadata={
                    "refinement_operation": "context_compression",
                    "rationale": rationale,
                    "fields_changed": ["execution.context_compression"],
                    "before_configuration": {"context_compression": False},
                    "after_configuration": {"context_compression": True},
                },
            ),
        )
        cand.execution.context_compression = True
        return cand

    @classmethod
    def _refine_prompt_simplification(cls, parent: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """
        Refinement Operator D: Prompt Simplification.
        Replaces verbose prompt templates with concise grounded medical prompt while strictly preserving safety.
        """
        curr_tmpl = parent.prompt.template_name
        curr_custom = parent.prompt.custom_instructions

        if curr_tmpl == "concise_medical" and not curr_custom:
            return None  # Already fully simplified

        rationale = "Replaced verbose medical prompt with concise grounded prompt to reduce input token cost while preserving grounding and safety requirements."
        cand = parent.clone(
            new_id=f"cand_refine_prompt_{rand.randint(100, 999)}",
            new_name="Refined: Prompt Simplification",
            new_description=rationale,
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.REFINEMENT,
                parent_workflow_id=parent.workflow_id,
                mutation_description=rationale,
                generation_metadata={
                    "refinement_operation": "prompt_simplification",
                    "rationale": rationale,
                    "fields_changed": ["prompt.template_name", "prompt.custom_instructions"],
                    "before_configuration": {"template_name": curr_tmpl, "custom_instructions": curr_custom},
                    "after_configuration": {"template_name": "concise_medical", "custom_instructions": None},
                },
            ),
        )
        cand.prompt.template_name = "concise_medical"
        cand.prompt.custom_instructions = None
        return cand

    @classmethod
    def _refine_output_token_reduction(cls, parent: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """
        Refinement Operator E: Output Token Reduction.
        Safely scales down excessive max_tokens bounds (1024 -> 512 -> 384) based on clinical brevity.
        """
        curr_tokens = parent.model.max_tokens
        if curr_tokens <= 256:
            return None  # At safe minimum boundary

        if curr_tokens >= 1024:
            new_tokens = 512
        elif curr_tokens >= 512:
            new_tokens = 384
        else:
            new_tokens = 256

        rationale = f"Reduced max_tokens from {curr_tokens} to {new_tokens} because historical evaluations consistently produced substantially shorter responses."
        cand = parent.clone(
            new_id=f"cand_refine_tokens_{rand.randint(100, 999)}",
            new_name="Refined: Output Token Limit",
            new_description=rationale,
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.REFINEMENT,
                parent_workflow_id=parent.workflow_id,
                mutation_description=rationale,
                generation_metadata={
                    "refinement_operation": "output_reduction",
                    "rationale": rationale,
                    "fields_changed": ["model.max_tokens"],
                    "before_configuration": {"max_tokens": curr_tokens},
                    "after_configuration": {"max_tokens": new_tokens},
                },
            ),
        )
        cand.model.max_tokens = new_tokens
        return cand

    @classmethod
    def _refine_redundant_config_cleanup(cls, parent: AgentWorkflow, rand: random.Random) -> Optional[AgentWorkflow]:
        """
        Refinement Operator F: Redundant Configuration Cleanup.
        Normalizes redundant timeout and reranking configurations for low top_k.
        """
        has_change = False
        fields = []
        before = {}
        after = {}

        cand = parent.clone(
            new_id=f"cand_refine_cleanup_{rand.randint(100, 999)}",
            new_name="Refined: Config Normalization",
        )

        # 1. Normalize excessive timeout
        if cand.execution.timeout_seconds > 30.0:
            before["timeout_seconds"] = cand.execution.timeout_seconds
            cand.execution.timeout_seconds = 30.0
            after["timeout_seconds"] = 30.0
            fields.append("execution.timeout_seconds")
            has_change = True

        # 2. Normalize reranking for tiny top_k (<=2)
        if cand.retrieval.top_k <= 2 and cand.retrieval.reranking is True:
            before["reranking"] = True
            cand.retrieval.reranking = False
            after["reranking"] = False
            fields.append("retrieval.reranking")
            has_change = True

        if not has_change:
            return None

        rationale = "Cleaned up redundant workflow configuration without affecting execution semantics."
        cand.description = rationale
        cand.mutation_metadata = MutationMetadata(
            mutation_type=MutationType.REFINEMENT,
            parent_workflow_id=parent.workflow_id,
            mutation_description=rationale,
            generation_metadata={
                "refinement_operation": "redundant_cleanup",
                "rationale": rationale,
                "fields_changed": fields,
                "before_configuration": before,
                "after_configuration": after,
            },
        )
        return cand

    # --- Mutation Registry ---
    MUTATION_REGISTRY: Dict[MutationType, List[Callable[[AgentWorkflow, random.Random], Optional[AgentWorkflow]]]] = {}

    @classmethod
    def _get_registry(cls) -> Dict[MutationType, List[Callable[[AgentWorkflow, random.Random], Optional[AgentWorkflow]]]]:
        if not cls.MUTATION_REGISTRY:
            cls.MUTATION_REGISTRY = {
                MutationType.PROMPT: [
                    cls._mutate_prompt_concise,
                    cls._mutate_prompt_detailed_clinical,
                    cls._mutate_prompt_strict_grounding,
                ],
                MutationType.RETRIEVAL: [
                    cls._mutate_retrieval_high_precision,
                    cls._mutate_retrieval_broad_coverage,
                    cls._mutate_retrieval_compact_chunking,
                ],
                MutationType.MODEL: [
                    cls._mutate_model_qwen,
                    cls._mutate_model_gemma,
                    cls._mutate_model_gemini_low_temp,
                ],
                MutationType.OPERATOR: [
                    cls._mutate_operator_context_compression,
                    cls._mutate_operator_strict_filter,
                ],
                MutationType.MIXED: [
                    cls._mutate_mixed_cost_optimized,
                    cls._mutate_mixed_high_recall_synthesis,
                ],
                MutationType.REFINEMENT: [
                    cls._refine_retrieval_reduction,
                    cls._refine_context_compression,
                    cls._refine_prompt_simplification,
                    cls._refine_output_token_reduction,
                    cls._refine_chunk_reduction,
                    cls._refine_redundant_config_cleanup,
                ],
            }
        return cls.MUTATION_REGISTRY

    @classmethod
    def generate_initial_candidates(
        cls,
        baseline: Optional[AgentWorkflow] = None,
        count: int = 5,
        seed: Optional[int] = None,
        allowed_types: Optional[List[MutationType]] = None,
        existing_versions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AgentWorkflow]:
        """
        Generate diverse, valid, independently executable candidate workflows.
        Enforces deep-copy parent isolation, validator safety, and duplicate configuration prevention.
        """
        base = baseline or get_baseline_workflow()
        rand = random.Random(seed) if seed is not None else random.Random()
        registry = cls._get_registry()

        # Parse allowed types
        raw_types = allowed_types or [
            MutationType.PROMPT,
            MutationType.RETRIEVAL,
            MutationType.MODEL,
            MutationType.OPERATOR,
            MutationType.MIXED,
        ]
        types_to_use: List[MutationType] = []
        for t in raw_types:
            if isinstance(t, str):
                try:
                    types_to_use.append(MutationType(t.lower().strip()))
                except ValueError:
                    continue
            elif isinstance(t, MutationType):
                types_to_use.append(t)

        if not types_to_use:
            types_to_use = [MutationType.RETRIEVAL]

        # Populate known configurations for duplicate prevention
        known_configs: List[Dict[str, Any]] = [normalize_workflow_config(base.to_dict())]
        if existing_versions:
            for v in existing_versions:
                v_cfg = v.get("configuration", v)
                if isinstance(v_cfg, dict):
                    known_configs.append(normalize_workflow_config(v_cfg))

        generated_candidates: List[AgentWorkflow] = []
        seen_ids: Set[str] = {base.workflow_id}

        # Check if refinement-only run
        is_refinement_only = len(types_to_use) == 1 and types_to_use[0] == MutationType.REFINEMENT

        if is_refinement_only:
            # Execute all refinement operators systematically against parent
            refine_ops = registry[MutationType.REFINEMENT]
            for op in refine_ops:
                if len(generated_candidates) >= count:
                    break
                candidate = op(base, rand)
                if not candidate:
                    continue

                cand_cfg = normalize_workflow_config(candidate.to_dict())
                # Duplicate prevention: Check against parent and previous candidates
                if any(cand_cfg == k for k in known_configs):
                    logger.info(f"Refinement candidate '{candidate.workflow_id}' skipped as duplicate configuration.")
                    continue

                # Ensure unique ID
                suffix = f"_{len(generated_candidates) + 1}"
                if candidate.workflow_id in seen_ids:
                    candidate.workflow_id = f"{candidate.workflow_id}{suffix}"
                seen_ids.add(candidate.workflow_id)

                # Validation safety
                val_res = validate_workflow(candidate)
                if not val_res.is_valid:
                    logger.warning(f"Refinement candidate '{candidate.workflow_id}' failed validation ({val_res.errors}); rejected.")
                    continue

                known_configs.append(cand_cfg)
                generated_candidates.append(candidate)

            logger.info(
                f"Generated {len(generated_candidates)} refinement candidates from parent '{base.workflow_id}'."
            )
            return generated_candidates

        # Multi-category exploration generation
        category_order: List[MutationType] = []
        for i in range(count):
            category_order.append(types_to_use[i % len(types_to_use)])

        for idx, m_type in enumerate(category_order):
            mutators = registry.get(m_type, registry[MutationType.RETRIEVAL])
            mutator_fn = mutators[idx % len(mutators)] if seed is not None else rand.choice(mutators)

            candidate = mutator_fn(base, rand)
            if not candidate:
                candidate = cls._mutate_retrieval_high_precision(base, rand)
                if not candidate:
                    continue

            cand_cfg = normalize_workflow_config(candidate.to_dict())
            if any(cand_cfg == k for k in known_configs):
                logger.info(f"Exploration candidate '{candidate.workflow_id}' matches existing config; generating unique variation.")
                candidate = cls._mutate_prompt_detailed_clinical(base, rand)
                if not candidate:
                    continue

            suffix = f"_{idx + 1}"
            if candidate.workflow_id in seen_ids:
                candidate.workflow_id = f"{candidate.workflow_id}{suffix}"
            seen_ids.add(candidate.workflow_id)

            val_res = validate_workflow(candidate)
            if not val_res.is_valid:
                logger.warning(
                    f"Generated candidate '{candidate.workflow_id}' failed validation ({val_res.errors}); fallback to safe mutation."
                )
                candidate = cls._mutate_retrieval_high_precision(base, rand)
                if candidate:
                    candidate.workflow_id = f"cand_fallback_{idx + 1}"
                else:
                    continue

            known_configs.append(cand_cfg)
            generated_candidates.append(candidate)

        logger.info(
            f"Generated {len(generated_candidates)} diverse exploration/refinement candidates across {[c.mutation_metadata.mutation_type.value for c in generated_candidates if c.mutation_metadata]}."
        )
        return generated_candidates


def generate_candidates(
    baseline: Optional[AgentWorkflow] = None,
    count: int = 5,
    seed: Optional[int] = None,
    allowed_types: Optional[List[MutationType]] = None,
    existing_versions: Optional[List[Dict[str, Any]]] = None,
) -> List[AgentWorkflow]:
    """Convenience helper to generate exploration and refinement candidate workflows."""
    return CandidateGenerator.generate_initial_candidates(
        baseline=baseline,
        count=count,
        seed=seed,
        allowed_types=allowed_types,
        existing_versions=existing_versions,
    )
