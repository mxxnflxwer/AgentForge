import copy
import logging
import random
from typing import Any, Callable, Dict, List, Optional

from app.services.agent_evo.validator import validate_workflow
from app.services.agent_evo.workflow import (
    AgentWorkflow,
    MutationMetadata,
    MutationType,
    get_baseline_workflow,
)

logger = logging.getLogger("agentforge.services.agent_evo.candidate_generator")


class CandidateGenerator:
    """
    AgentEvo Evolutionary Exploration Candidate Generator.
    Expands the workflow search space by generating diverse, independently evaluable
    candidate workflows through controlled mutations across 5 key categories:
    1. PROMPT (templates, instruction style, grounding constraints)
    2. RETRIEVAL (top_k, similarity_threshold, chunk_size, chunk_overlap, reranking)
    3. MODEL (model_id, provider, temperature, max_tokens)
    4. OPERATOR (context compression, extractive filter, execution constraints)
    5. MIXED (composite mutations across multiple dimensions)
    """

    # --- Prompt Mutations ---

    @classmethod
    def _mutate_prompt_concise(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_prompt_detailed_clinical(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_prompt_strict_grounding(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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

    # --- Retrieval Mutations ---

    @classmethod
    def _mutate_retrieval_high_precision(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_retrieval_broad_coverage(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_retrieval_compact_chunking(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
        cand = base.clone(
            new_id=f"cand_retrieval_chunking_{rand.randint(100, 999)}",
            new_name="Fine-Grained Chunking Retrieval",
            new_description="Configured compact chunk size (500 tokens) with 100-token overlap and top_k=6 for granular context retrieval.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.RETRIEVAL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Configured compact chunk size (500 tokens) with 100-token overlap and top_k=6 for granular context retrieval.",
                generation_metadata={
                    "chunk_size": 500,
                    "chunk_overlap": 100,
                    "top_k": 6,
                },
            ),
        )
        cand.retrieval.chunk_size = 500
        cand.retrieval.chunk_overlap = 100
        cand.retrieval.top_k = 6
        return cand

    # --- Model Mutations ---

    @classmethod
    def _mutate_model_qwen(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_model_gpt_oss(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
        cand = base.clone(
            new_id=f"cand_model_gpt_oss_{rand.randint(100, 999)}",
            new_name="GPT-OSS 120B Reasoning",
            new_description="Evaluates open-weights 120B reasoning model hosted on Hugging Face with zero temperature.",
            mutation_metadata=MutationMetadata(
                mutation_type=MutationType.MODEL,
                parent_workflow_id=base.workflow_id,
                mutation_description="Evaluates open-weights 120B reasoning model hosted on Hugging Face with zero temperature.",
                generation_metadata={
                    "provider": "Hugging Face",
                    "model_id": "openai/gpt-oss-120b",
                    "temperature": 0.0,
                },
            ),
        )
        cand.model.provider = "Hugging Face"
        cand.model.model_id = "openai/gpt-oss-120b"
        cand.model.temperature = 0.0
        return cand

    @classmethod
    def _mutate_model_gemini_low_temp(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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

    # --- Operator / Logic Mutations ---

    @classmethod
    def _mutate_operator_context_compression(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_operator_strict_filter(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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

    # --- Mixed / Composite Mutations ---

    @classmethod
    def _mutate_mixed_cost_optimized(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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
    def _mutate_mixed_high_recall_synthesis(cls, base: AgentWorkflow, rand: random.Random) -> AgentWorkflow:
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

    # --- Mutation Registry ---
    MUTATION_REGISTRY: Dict[MutationType, List[Callable[[AgentWorkflow, random.Random], AgentWorkflow]]] = {}

    @classmethod
    def _get_registry(cls) -> Dict[MutationType, List[Callable[[AgentWorkflow, random.Random], AgentWorkflow]]]:
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
                    cls._mutate_model_gpt_oss,
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
            }
        return cls.MUTATION_REGISTRY

    @classmethod
    def generate_initial_candidates(
        cls,
        baseline: Optional[AgentWorkflow] = None,
        count: int = 5,
        seed: Optional[int] = None,
        allowed_types: Optional[List[MutationType]] = None,
    ) -> List[AgentWorkflow]:
        """
        Generate diverse, valid, independently executable candidate workflows.
        Guarantees mutation category diversity across Prompt, Retrieval, Model, Operator, and Mixed.
        """
        base = baseline or get_baseline_workflow()
        rand = random.Random(seed) if seed is not None else random.Random()
        registry = cls._get_registry()

        types_to_use = allowed_types or [
            MutationType.PROMPT,
            MutationType.RETRIEVAL,
            MutationType.MODEL,
            MutationType.OPERATOR,
            MutationType.MIXED,
        ]

        # Prioritize 1 candidate per distinct category first, then cycle
        category_order: List[MutationType] = []
        for i in range(count):
            category_order.append(types_to_use[i % len(types_to_use)])

        generated_candidates: List[AgentWorkflow] = []
        seen_ids = set()

        for idx, m_type in enumerate(category_order):
            mutators = registry.get(m_type, registry[MutationType.RETRIEVAL])
            # Select mutator deterministically based on index/seed
            mutator_fn = mutators[idx % len(mutators)] if seed is not None else rand.choice(mutators)

            # Generate candidate
            candidate = mutator_fn(base, rand)

            # Ensure candidate ID uniqueness
            suffix = f"_{idx + 1}"
            if candidate.workflow_id in seen_ids:
                candidate.workflow_id = f"{candidate.workflow_id}{suffix}"
            seen_ids.add(candidate.workflow_id)

            # Validate mutation safety
            val_res = validate_workflow(candidate)
            if not val_res.is_valid:
                logger.warning(
                    f"Generated candidate '{candidate.workflow_id}' failed validation ({val_res.errors}); falling back to safe mutation."
                )
                candidate = cls._mutate_retrieval_high_precision(base, rand)
                candidate.workflow_id = f"cand_fallback_{idx + 1}"

            generated_candidates.append(candidate)

        logger.info(
            f"Generated {len(generated_candidates)} diverse exploration candidates across {[c.mutation_metadata.mutation_type.value for c in generated_candidates if c.mutation_metadata]}."
        )
        return generated_candidates


def generate_candidates(
    baseline: Optional[AgentWorkflow] = None,
    count: int = 5,
    seed: Optional[int] = None,
    allowed_types: Optional[List[MutationType]] = None,
) -> List[AgentWorkflow]:
    """Convenience helper to generate exploration candidate workflows."""
    return CandidateGenerator.generate_initial_candidates(
        baseline=baseline,
        count=count,
        seed=seed,
        allowed_types=allowed_types,
    )
