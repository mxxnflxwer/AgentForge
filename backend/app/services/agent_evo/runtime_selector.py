import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.models.workflow_version import WorkflowVersion
from app.services.agent_evo.workflow import AgentWorkflow, get_baseline_workflow
from app.services.reranker import classify_query_intent

logger = logging.getLogger("agentforge.services.agent_evo.runtime_selector")


class WorkflowSelectionResult(BaseModel):
    """
    Structured outcome of automatic runtime workflow selection.
    Represents the best-matching eligible approved workflow for an incoming query.
    Describes the selection as the best-matching eligible workflow, not as a guaranteed global optimum.
    """
    version_id: str = Field(..., description="ID of the selected workflow version or 'baseline'")
    version_number: Optional[int] = Field(None, description="Sequential version number, or 0 for baseline")
    workflow_name: str = Field(..., description="Name of the selected workflow")
    task_category: str = Field(..., description="Classified task category for the incoming query")
    configuration: Dict[str, Any] = Field(..., description="Exact immutable workflow configuration dictionary")
    selection_reason: str = Field(..., description="Transparent, human-readable rationale for workflow selection")
    is_fallback: bool = Field(False, description="True if no specialized approved profile matched and fallback was used")
    status: str = Field("ACTIVE", description="Lifecycle status: ACTIVE, APPROVED, or BASELINE_FALLBACK")
    model_id: str = Field("gemini-3.5-flash-lite", description="Target model identifier")
    provider: str = Field("Google", description="Target provider identifier")
    evaluation_snapshot: Optional[Dict[str, Any]] = Field(None, description="Persisted evaluation metrics snapshot")


class RuntimeWorkflowSelector:
    """
    Automatic runtime selector choosing eligible, approved workflows based on incoming query task type.

    Invariants:
    1. NEVER executes an unapproved candidate workflow.
    2. Does NOT run full evolutionary optimization at query runtime.
    3. Categorizes queries reliably using established rule-based intent classification.
    4. Matches against approved workflow evaluation profiles using Groundedness and Cost.
    5. Falls back safely to active production workflow or configured production baseline.
    6. Selection is explicitly described as best-matching eligible, never guaranteed globally optimal.
    """

    RELIABLE_TASK_CATEGORIES = {
        "diagnosis",
        "symptoms",
        "medications",
        "treatment",
        "findings",
        "examination",
        "summary",
        "explanation",
        "specific_fact",
        "unsupported_query",
    }

    @classmethod
    def classify_query_task(cls, query: str) -> str:
        """
        Classify incoming query into a supported clinical task category.
        Uses rule-based domain keyword and regex pattern classification.
        Supported reliable categories:
        - 'summary': Document or report summaries.
        - 'diagnosis': Disease, condition, etiology, clinical impression.
        - 'medications': Drugs, prescriptions, dosages, pharmacotherapy.
        - 'symptoms': Patient complaints, symptoms, pain, dyspnea.
        - 'treatment': Treatment plans, therapies, management, lifestyle.
        - 'findings': Labs, diagnostics, ECG, imaging, physical exam findings.
        - 'explanation': Requests explaining reasons or mechanisms.
        - 'specific_fact': Direct factual queries (age, vitals, etc.).
        - 'unsupported_query': Out-of-domain or greetings.
        """
        if not query or not query.strip():
            return "unsupported_query"

        q_lower = query.lower().strip()
        from app.services.reranker import is_greeting_or_nonsense, is_out_of_domain
        if is_greeting_or_nonsense(q_lower) or is_out_of_domain(q_lower):
            return "unsupported_query"

        # Summary Intent
        if any(k in q_lower for k in [
            "summarize", "summary", "overview of the report", "overview of the complete report",
            "summarise", "give me a summary", "complete report", "full report", "synthesize", "document summary"
        ]):
            return "summary"

        # Explanation Intent
        import re
        if re.search(r"\b(explain|why|how come|describe why|explain simply|elaborate)\b", q_lower):
            return "explanation"

        # Diagnosis Intent
        if any(k in q_lower for k in [
            "diagnosis", "diagnoses", "diagnose", "disease", "condition", "syndrome",
            "disorder", "illness", "assessment", "clinical impression", "what is wrong with"
        ]):
            return "diagnosis"

        # Medications Intent
        if any(k in q_lower for k in [
            "medication", "medications", "medicine", "drug", "drugs", "prescription", "prescriptions",
            "prescribe", "dose", "dosage", "tablet", "pill", "statin", "atorvastatin", "aspirin",
            "levothyroxine", "metformin", "inhaler", "antibiotic", "pharmacotherapy"
        ]):
            return "medications"

        # Symptoms Intent
        if any(k in q_lower for k in [
            "symptom", "symptoms", "complaint", "complaints", "chest discomfort", "chest tightness",
            "shortness of breath", "dyspnea", "fatigue", "palpitations", "cough", "fever"
        ]):
            return "symptoms"

        # Treatment Intent
        if any(k in q_lower for k in [
            "treatment", "therapy", "management", "intervention", "recommendation", "recommendations",
            "surgery", "lifestyle", "follow-up plan"
        ]):
            return "treatment"

        # Findings Intent
        if any(k in q_lower for k in [
            "findings", "finding", "important findings", "diagnostic findings", "ecg", "ekg",
            "troponin", "tsh", "labs", "laboratory", "investigation", "investigations",
            "imaging", "stress test", "angiography", "ultrasound", "biopsy", "test results"
        ]):
            return "findings"

        return "specific_fact"

    @classmethod
    def select_workflow_for_query(
        cls,
        query: str,
        db: Optional[Session] = None,
        candidate_version_id_override: Optional[str] = None,
    ) -> WorkflowSelectionResult:
        """
        Selects an eligible approved workflow based on incoming query task category and evaluated profiles.

        Routing hierarchy:
        1. If candidate_version_id_override is provided, verify it is an APPROVED or ACTIVE version.
           Rejects unapproved candidates with ValueError / safe fallback.
        2. Classify query into task category.
        3. Retrieve all approved and active workflow versions from the database.
        4. Match evaluated workflow profiles against the query task category.
        5. Rank matching profiles strictly by Groundedness (max) and Cost / Query (min).
        6. If no specialized approved profile matches, fall back to currently ACTIVE workflow.
        7. If no active workflow exists, fall back to configured Production Baseline.
        """
        task_category = cls.classify_query_task(query)

        # Baseline fallback helper
        def _get_baseline_fallback(reason: str) -> WorkflowSelectionResult:
            baseline = get_baseline_workflow()
            logger.info(
                f"Query classification: '{task_category}' | Selected workflow version: 'baseline' | "
                f"Selection reason: '{reason}'"
            )
            return WorkflowSelectionResult(
                version_id="baseline",
                version_number=0,
                workflow_name="Production Baseline (Default)",
                task_category=task_category,
                configuration=baseline.to_dict(),
                selection_reason=reason,
                is_fallback=True,
                status="BASELINE_FALLBACK",
                model_id=baseline.model.model_id,
                provider=baseline.model.provider,
                evaluation_snapshot=None,
            )

        if not db:
            return _get_baseline_fallback(
                "No database session provided. Safely defaulted to configured production baseline. "
                "Selection represents the safe fallback, not a guaranteed global optimum."
            )

        # Step 1: Handle explicit version request guardrail
        if candidate_version_id_override:
            if candidate_version_id_override in ("baseline", "Production Baseline"):
                return _get_baseline_fallback(
                    "Production baseline explicitly requested. "
                    "Selection represents the safe baseline, not a guaranteed global optimum."
                )

            # Query DB for this version
            requested_ver = db.query(WorkflowVersion).filter(
                (WorkflowVersion.version_id == candidate_version_id_override)
                | (WorkflowVersion.id == candidate_version_id_override)
            ).first()

            if not requested_ver:
                logger.warning(
                    f"Rejection: Version '{candidate_version_id_override}' not found in registry. "
                    f"Unapproved candidates cannot be executed at runtime."
                )
                return _get_baseline_fallback(
                    f"Workflow version '{candidate_version_id_override}' was not found in the approved registry. "
                    f"Unapproved candidates are never executed in production; safely defaulted to baseline."
                )

            status_normalized = (requested_ver.status or "").upper()
            if status_normalized not in ("APPROVED", "ACTIVE"):
                logger.warning(
                    f"Rejection: Workflow version '{requested_ver.version_id}' has status '{requested_ver.status}'. "
                    f"Only APPROVED or ACTIVE workflows may be executed."
                )
                return _get_baseline_fallback(
                    f"Workflow version '{requested_ver.version_id}' is not in APPROVED or ACTIVE state (status: '{requested_ver.status}'). "
                    f"Safely fell back to production baseline."
                )

            # Legitimate approved version requested
            cfg = requested_ver.configuration or {}
            reason = (
                f"Explicitly requested approved workflow '{requested_ver.version_id}' (v{requested_ver.version_number}). "
                f"Configuration verified from immutable registry. "
                f"Selection represents the requested approved workflow, not a guaranteed global optimum."
            )
            logger.info(
                f"Query classification: '{task_category}' | Selected workflow version: '{requested_ver.version_id}' | "
                f"Selection reason: '{reason}'"
            )
            return WorkflowSelectionResult(
                version_id=requested_ver.version_id,
                version_number=requested_ver.version_number,
                workflow_name=requested_ver.name,
                task_category=task_category,
                configuration=cfg,
                selection_reason=reason,
                is_fallback=False,
                status=requested_ver.status.upper(),
                model_id=requested_ver.model_id,
                provider=requested_ver.provider,
                evaluation_snapshot=requested_ver.evaluation_snapshot,
            )

        # Step 2: Fetch all eligible approved & active versions
        eligible_versions = db.query(WorkflowVersion).filter(
            WorkflowVersion.status.in_(["ACTIVE", "active", "APPROVED", "approved"])
        ).all()

        active_version = next(
            (v for v in eligible_versions if (v.status or "").upper() == "ACTIVE"),
            None,
        )

        if not eligible_versions:
            return _get_baseline_fallback(
                f"No approved or active workflows found in database for task category '{task_category}'. "
                f"Safely defaulted to configured production baseline. "
                f"Selection represents the safe fallback, not a guaranteed global optimum."
            )

        # Step 3: Match evaluated profiles against query task category
        matched_candidates: List[Tuple[WorkflowVersion, float, float]] = []

        for v in eligible_versions:
            cfg = v.configuration or {}
            eval_snap = v.evaluation_snapshot or {}

            # Determine task categories associated with this version's evaluation profile
            version_categories = set()
            if isinstance(cfg.get("task_categories"), list):
                version_categories.update(cfg.get("task_categories"))
            if isinstance(cfg.get("target_tasks"), list):
                version_categories.update(cfg.get("target_tasks"))
            if isinstance(eval_snap.get("evaluated_task_categories"), list):
                version_categories.update(eval_snap.get("evaluated_task_categories"))

            # Inspect per-query evaluations if available
            raw_details = eval_snap.get("raw_details") or {}
            per_query = raw_details.get("per_query_evaluations") or []
            for pq in per_query:
                q_text = pq.get("query")
                if q_text:
                    version_categories.add(cls.classify_query_task(q_text))

            # If task_category matches
            if task_category in version_categories:
                groundedness = float(eval_snap.get("groundedness") or 0.0)
                cost = float(eval_snap.get("cost") or 0.0)
                matched_candidates.append((v, groundedness, cost))

        # Step 4: If specialized matching versions exist, choose Pareto best (Groundedness max, Cost min)
        if matched_candidates:
            # Sort by groundedness descending, then cost ascending
            matched_candidates.sort(key=lambda item: (-item[1], item[2]))
            best_ver, best_g, best_c = matched_candidates[0]

            reason = (
                f"Selected approved workflow '{best_ver.version_id}' (v{best_ver.version_number}) as best-matching "
                f"eligible workflow for task category '{task_category}' (evaluated profile: groundedness={best_g:.4f}, "
                f"cost=${best_c:.6f}/query). Selection represents the best-matching eligible workflow from the approved registry, "
                f"not a guaranteed global optimum."
            )
            logger.info(
                f"Query classification: '{task_category}' | Selected workflow version: '{best_ver.version_id}' | "
                f"Selection reason: '{reason}'"
            )
            return WorkflowSelectionResult(
                version_id=best_ver.version_id,
                version_number=best_ver.version_number,
                workflow_name=best_ver.name,
                task_category=task_category,
                configuration=best_ver.configuration or {},
                selection_reason=reason,
                is_fallback=False,
                status=best_ver.status.upper(),
                model_id=best_ver.model_id,
                provider=best_ver.provider,
                evaluation_snapshot=best_ver.evaluation_snapshot,
            )

        # Step 5: Fallback to Active Production Workflow if no specialized category match
        if active_version:
            reason = (
                f"No specialized approved workflow matched task category '{task_category}'. "
                f"Safely selected active production workflow '{active_version.version_id}' (v{active_version.version_number}). "
                f"Selection represents the best-matching eligible workflow from the approved registry, "
                f"not a guaranteed global optimum."
            )
            logger.info(
                f"Query classification: '{task_category}' | Selected workflow version: '{active_version.version_id}' | "
                f"Selection reason: '{reason}'"
            )
            return WorkflowSelectionResult(
                version_id=active_version.version_id,
                version_number=active_version.version_number,
                workflow_name=active_version.name,
                task_category=task_category,
                configuration=active_version.configuration or {},
                selection_reason=reason,
                is_fallback=True,
                status="ACTIVE",
                model_id=active_version.model_id,
                provider=active_version.provider,
                evaluation_snapshot=active_version.evaluation_snapshot,
            )

        # Step 6: Fallback to baseline
        return _get_baseline_fallback(
            f"No specialized approved workflow matched task category '{task_category}', and no active workflow is set. "
            f"Safely defaulted to configured production baseline. "
            f"Selection represents the safe fallback, not a guaranteed global optimum."
        )
