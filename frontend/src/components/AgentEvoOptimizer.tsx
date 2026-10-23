import React, { useState, useEffect, useRef } from 'react'

export type MutationType = 'prompt' | 'retrieval' | 'model' | 'operator' | 'mixed' | 'refinement' | 'crossover'

export interface MutationMetadata {
  mutation_type: MutationType
  parent_workflow_id: string
  mutation_description: string
  generation_metadata?: Record<string, any>
}

export interface RetrievalConfig {
  top_k: number
  similarity_threshold: number
  chunk_size: number
  chunk_overlap: number
  enable_rerank: boolean
}

export interface ModelConfig {
  model_id: string
  provider: string
  temperature: number
  max_tokens: number
}

export interface PromptConfig {
  template_name: string
  include_disclaimer: boolean
  custom_instructions?: string | null
}

export interface ExecutionConfig {
  timeout_seconds: number
  max_retries: number
  context_compression?: boolean
}

export interface AgentWorkflow {
  workflow_id: string
  name: string
  description?: string
  retrieval: RetrievalConfig
  model: ModelConfig
  prompt: PromptConfig
  execution: ExecutionConfig
  mutation_metadata?: MutationMetadata | null
}

export interface CandidateMetrics {
  workflow_id: string
  workflow_name: string
  performance: number
  cost: number
  groundedness?: number | null
  hallucination_rate?: number | null
  accuracy?: number | null
  total_tokens: number
  input_tokens: number
  output_tokens: number
  llm_latency_ms: number
  execution_time_ms: number
  status: string
  error_message?: string | null
  evaluation_count?: number
  avg_groundedness?: number | null
  avg_hallucination?: number | null
  avg_cost_per_query?: number | null
  avg_tokens_per_query?: number | null
  avg_latency_ms?: number | null
  raw_details?: Record<string, any>
}

export interface EvaluatedCandidate {
  candidate_id: string
  workflow: AgentWorkflow
  metrics: CandidateMetrics
  is_pareto_optimal: boolean
  dominated_by: string[]
  dominates_candidates: string[]
  mutation_metadata?: MutationMetadata | null
}

export interface ParetoTradeOff {
  candidate_id: string
  name: string
  performance_delta_pct: number
  cost_delta_pct: number
  latency_delta_pct: number
  trade_off_summary: string
  workflow_differences: Record<string, any>
}

export interface OptimizationReport {
  run_id: string
  timestamp: string
  query: string
  baseline: EvaluatedCandidate
  total_candidates_evaluated: number
  successful_candidates: number
  failed_candidates: number
  pareto_frontier_count: number
  pareto_frontier: EvaluatedCandidate[]
  all_candidates: EvaluatedCandidate[]
  trade_off_analyses: ParetoTradeOff[]
  executive_summary: string
  human_approval_required: boolean
  approved_workflow_id?: string | null
  is_workload?: boolean
  workload_queries?: string[]
  total_workflow_evaluations?: number
  strategy?: string
  scheduler_decision?: {
    stage?: string
    stage_name?: string
    strategy?: string
    reason: string
    rationale?: string
    stagnation_counter?: number
    pareto_archive_size?: number
    parameters?: Record<string, any>
  } | null
  scope_claim?: string
}

export interface WorkflowVersion {
  id: string
  version_id: string
  version_number: number
  name: string
  description?: string | null
  source_run_id: string
  source_candidate_id: string
  model_id: string
  provider: string
  configuration: Record<string, any>
  evaluation_snapshot: Record<string, any>
  status: string
  approved_by: string
  approved_at: string
  activated_at?: string | null
  created_at: string
}

export interface ActiveWorkflowInfo {
  version_id: string
  active_version_id?: string
  version_number?: number | null
  name: string
  status: string
  activated_at?: string | null
  configuration: Record<string, any>
  evaluation_snapshot?: Record<string, any> | null
  model_id: string
  provider: string
  is_active_set?: boolean
}

export interface WorkflowExecution {
  execution_id: string
  workflow_version_id: string
  query: string
  model_id?: string
  model?: string
  provider: string
  input_tokens?: number | null
  output_tokens?: number | null
  total_tokens?: number | null
  cost?: number | null
  llm_latency_ms?: number | null
  total_execution_time_ms: number
  response: string
  retrieved_doc_ids?: string[] | null
  sources?: Array<{
    chunk_id: string
    document_id: string
    chunk_index: number
    section_title?: string
    similarity_score?: number
    relevance_score?: number
  }>
  groundedness?: number | null
  execution_trace?: Record<string, any> | null
  status?: string
  created_at: string
}

export interface WorkflowFeedback {
  feedback_id: string
  execution_id: string
  workflow_version_id: string
  user_id: string
  rating: number
  feedback_text?: string | null
  created_at: string
}

export interface WorkflowAuditEvent {
  event_id: string
  event_type: string
  workflow_version_id?: string | null
  previous_version_id?: string | null
  triggered_by?: string
  user_email?: string
  details?: Record<string, any> | null
  created_at: string
}

export interface BenchmarkItemResult {
  version_id: string
  version?: string
  version_number?: number | null
  workflow_name: string
  workflow?: string
  parent_workflow_id?: string | null
  parent_workflow?: string | null
  source_candidate_id?: string | null
  source_run_id?: string | null
  model_id: string
  model?: string
  provider: string
  groundedness: number
  hallucination_rate: number
  hallucination?: number
  cost: number
  cost_per_query?: number
  total_tokens: number
  input_tokens: number
  output_tokens: number
  llm_latency_ms: number
  latency_ms?: number
  execution_time_ms: number
  retrieval: Record<string, any>
  retrieval_config_summary?: string | null
  retrieval_config?: string | null
  prompt_template: string
  prompt_strategy?: string | null
  context_compression: boolean
  operator_config?: Record<string, any> | null
  is_pareto_optimal: boolean
  pareto_status: string
  dominated_by: string[]
  dominates: string[]
  claim_breakdown?: Array<Record<string, any>> | null
  raw_details?: Record<string, any> | null
  configuration: Record<string, any>
  iterations_count?: number
  runs?: Array<{
    iteration: number
    groundedness: number
    hallucination_rate: number
    cost: number
    total_tokens: number
    input_tokens: number
    output_tokens: number
    llm_latency_ms: number
    execution_time_ms: number
  }> | null
  relative_changes?: {
    groundedness_delta?: number
    groundedness_change?: number
    cost_reduction_pct?: number
    cost_change_pct?: number
    latency_reduction_pct?: number
    latency_change_pct?: number
    token_reduction_pct?: number
    token_change_pct?: number
  } | null
  std_dev?: {
    groundedness?: number
    hallucination?: number
    cost?: number
    cost_per_query?: number
    tokens?: number
    latency_ms?: number
  } | null
}

export interface BenchmarkReport {
  benchmark_id: string
  name: string
  description?: string | null
  status: string
  preset_name?: string | null
  query: string
  document_id?: string | null
  document_ids?: string[] | null
  expected_answer?: string | null
  repetitions?: number | null
  selected_versions: string[]
  results: BenchmarkItemResult[]
  pareto_frontier: BenchmarkItemResult[]
  dominated_versions: BenchmarkItemResult[]
  summary: string
  created_by: string
  created_at: string
}

export interface LineageNode {
  version_id: string
  version_number?: number | null
  name: string
  parent_workflow_id?: string | null
  parent_a_version_id?: string | null
  parent_b_version_id?: string | null
  candidate_type?: string | null
  crossover_strategy?: string | null
  crossover_map?: Record<string, string> | null
  source_run_id?: string | null
  source_candidate_id?: string | null
  model_id: string
  provider: string
  mutation_type?: string | null
  refinement_operation?: string | null
  scheduler_strategy?: string | null
  fields_changed?: string[] | null
  rationale?: string | null
  before_configuration?: Record<string, any> | null
  after_configuration?: Record<string, any> | null
  evaluation_snapshot?: Record<string, any> | null
  approved_by?: string | null
  approved_at?: string | null
  status: string
  configuration?: Record<string, any> | null
}

export interface LineageEdge {
  source: string
  target: string
  label?: string | null
}

export interface LineageData {
  total_versions: number
  nodes: LineageNode[]
  edges: LineageEdge[]
}

export interface VersionInspectData {
  version_id: string
  version_number?: number | null
  name: string
  description?: string | null
  parent_workflow_id?: string | null
  parent_a_version_id?: string | null
  parent_b_version_id?: string | null
  candidate_type?: string | null
  crossover_strategy?: string | null
  crossover_map?: Record<string, string> | null
  parent_version?: {
    version_id: string
    version_number: number
    name: string
    model_id: string
    provider: string
  } | null
  parent_a_version?: {
    version_id: string
    version_number: number
    name: string
    model_id: string
    provider: string
  } | null
  parent_b_version?: {
    version_id: string
    version_number: number
    name: string
    model_id: string
    provider: string
  } | null
  source_run_id?: string | null
  source_candidate_id?: string | null
  model_id: string
  provider: string
  mutation_type?: string | null
  refinement_operation?: string | null
  fields_changed?: string[] | null
  rationale?: string | null
  before_configuration?: Record<string, any> | null
  after_configuration?: Record<string, any> | null
  configuration: Record<string, any>
  evaluation_snapshot: Record<string, any>
  status: string
  approved_by: string
  approved_at: string
  created_at: string
}

export interface CrossoverInheritanceItem {
  component: string
  source_parent: 'parent_a' | 'parent_b'
  source_version_id: string
  value: any
}

export interface CrossoverPreviewResponse {
  is_valid: boolean
  validation_errors: string[]
  parent_a: {
    version_id: string
    name: string
    configuration: Record<string, any>
  }
  parent_b: {
    version_id: string
    name: string
    configuration: Record<string, any>
  }
  crossover_strategy: string
  crossover_map: Record<string, string>
  inheritance_breakdown: CrossoverInheritanceItem[]
  resulting_configuration: Record<string, any>
  post_crossover_mutation: string
}

interface DocumentItem {
  id: string
  filename?: string
  original_filename?: string
}

interface AgentEvoOptimizerProps {
  apiBase: string
  documents: DocumentItem[]
  selectedDocId: string
  setSelectedDocId: (id: string) => void
  isLoggedIn: boolean
}

type PresetType = 'clinical' | 'medications' | 'ood' | 'custom'

const PRESET_QUERIES: Record<PresetType, { label: string; query: string }> = {
  clinical: {
    label: 'Clinical Diagnosis',
    query: 'What is the primary clinical diagnosis and assessment for this patient?',
  },
  medications: {
    label: 'Medications & Plan',
    query: 'What medications were prescribed and what is the detailed treatment plan?',
  },
  ood: {
    label: 'Out-of-Domain',
    query: 'What is the clinic telephone number and office hours?',
  },
  custom: {
    label: 'Custom',
    query: '',
  },
}

type WorkloadPresetType = 'clinical_comprehensive' | 'diagnostic_triage' | 'medication_safety' | 'custom_workload'

const WORKLOAD_PRESETS: Record<WorkloadPresetType, { label: string; description: string; queries: string[] }> = {
  clinical_comprehensive: {
    label: 'Clinical Assessment Suite (5 Queries)',
    description: 'Comprehensive 5-query panel spanning diagnosis, medications, labs, treatment plan, and cardiac history.',
    queries: [
      'What is the primary clinical diagnosis and assessment for this patient?',
      'What medications were prescribed for this patient?',
      'What are the abnormal laboratory findings?',
      'What is the patient\'s treatment plan?',
      'What is the relevant cardiac history?',
    ],
  },
  diagnostic_triage: {
    label: 'Diagnostic Triage Panel (3 Queries)',
    description: 'Focused 3-query evaluation covering primary diagnosis, lab abnormalities, and immediate clinical plan.',
    queries: [
      'What is the primary clinical diagnosis and assessment for this patient?',
      'What are the abnormal laboratory findings?',
      'What is the patient\'s treatment plan?',
    ],
  },
  medication_safety: {
    label: 'Pharmacotherapy & Safety (3 Queries)',
    description: 'Safety-oriented 3-query evaluation on prescribed drugs, contraindications, and monitoring.',
    queries: [
      'What medications were prescribed for this patient?',
      'Are there any contraindications or drug interactions noted?',
      'What follow-up monitoring is recommended?',
    ],
  },
  custom_workload: {
    label: 'Custom Evaluation Workload',
    description: 'Define your own representative multi-query workload for multi-query validation.',
    queries: [],
  },
}

export const AgentEvoOptimizer: React.FC<AgentEvoOptimizerProps> = ({
  apiBase,
  documents,
  selectedDocId,
  setSelectedDocId,
  isLoggedIn,
}) => {
  // Top-level Navigation Sub-Tabs: 6 Purpose-Built Sections
  const [activeMainTab, setActiveMainTabState] = useState<
    'overview' | 'evolution' | 'review' | 'production' | 'executions' | 'benchmarking'
  >('overview')
  const [showLineageInProduction, setShowLineageInProduction] = useState<boolean>(false)
  const [showAdvancedStrategyDetails, setShowAdvancedStrategyDetails] = useState<boolean>(false)

  const setActiveMainTab = (tab: string) => {
    if (tab === 'optimizer') setActiveMainTabState('evolution')
    else if (tab === 'lifecycle') setActiveMainTabState('production')
    else if (tab === 'benchmark') setActiveMainTabState('benchmarking')
    else if (tab === 'lineage') {
      setActiveMainTabState('production')
      setShowLineageInProduction(true)
    } else {
      setActiveMainTabState(tab as any)
    }
  }

  // Evaluation Mode: Single Query vs Multi-Query Workload
  const [evalMode, setEvalMode] = useState<'single' | 'workload'>('single')
  const [activeWorkloadPreset, setActiveWorkloadPreset] = useState<WorkloadPresetType>('clinical_comprehensive')
  const [workloadText, setWorkloadText] = useState<string>(WORKLOAD_PRESETS.clinical_comprehensive.queries.join('\n'))

  // Candidate Generation Strategy Selector inside Optimizer: Adaptive, Mutation, Refinement, Crossover
  const [candidateStrategy, setCandidateStrategy] = useState<'adaptive' | 'mutation' | 'refinement' | 'crossover'>('adaptive')

  // Optimization Setup State
  const [activePreset, setActivePreset] = useState<PresetType>('clinical')
  const [query, setQuery] = useState(PRESET_QUERIES.clinical.query)
  const [generatedCandidateCount, setGeneratedCandidateCount] = useState<number>(4)
  const [selectedMutationType, setSelectedMutationType] = useState<string>('all')
  const [selectedRefinementType, setSelectedRefinementType] = useState<string>('all')
  const [selectedParentWorkflowId, setSelectedParentWorkflowId] = useState<string>('baseline')

  // Crossover Specific Controls
  const [crossoverParentA, setCrossoverParentA] = useState<string>('baseline')
  const [crossoverParentB, setCrossoverParentB] = useState<string>('wf_v1_60e5af')
  const [crossoverMap, setCrossoverMap] = useState<Record<string, 'parent_a' | 'parent_b'>>({
    retrieval: 'parent_a',
    prompt: 'parent_b',
    model: 'parent_a',
    generation: 'parent_a',
    operators: 'parent_b',
  })
  const [postCrossoverMutation, setPostCrossoverMutation] = useState<string>('none')
  const [crossoverPreview, setCrossoverPreview] = useState<CrossoverPreviewResponse | null>(null)
  const [isPreviewLoading, setIsPreviewLoading] = useState<boolean>(false)

  // Execution & Pipeline State
  const [running, setRunning] = useState(false)
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0)
  const [report, setReport] = useState<OptimizationReport | null>(null)
  const [, setRuns] = useState<OptimizationReport[]>([])
  const [workflowVersions, setWorkflowVersions] = useState<WorkflowVersion[]>([])
  const [activeWorkflow, setActiveWorkflow] = useState<ActiveWorkflowInfo | null>(null)
  const [executions, setExecutions] = useState<WorkflowExecution[]>([])
  const [auditEvents, setAuditEvents] = useState<WorkflowAuditEvent[]>([])
  const [expandedExecId, setExpandedExecId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Interactive Modals & Governance State
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null)
  const [inspectModalCandidate, setInspectModalCandidate] = useState<EvaluatedCandidate | null>(null)
  const [inspectModalVersion, setInspectModalVersion] = useState<VersionInspectData | null>(null)
  const [confirmApprovalCandidate, setConfirmApprovalCandidate] = useState<EvaluatedCandidate | null>(null)
  const [approvingId, setApprovingId] = useState<string | null>(null)
  const [isReportCollapsed, setIsReportCollapsed] = useState<boolean>(false)

  // Activation & Rollback Modal State
  const [actionModalVersion, setActionModalVersion] = useState<{ version: WorkflowVersion; type: 'activate' | 'rollback' } | null>(null)
  const [actionReason, setActionReason] = useState<string>('')
  const [isActionSubmitting, setIsActionSubmitting] = useState<boolean>(false)

  // Feedback Modal State
  const [feedbackModalExecution, setFeedbackModalExecution] = useState<WorkflowExecution | null>(null)
  const [feedbackRating, setFeedbackRating] = useState<number>(5)
  const [feedbackText, setFeedbackText] = useState<string>('')
  const [isSubmittingFeedback, setIsSubmittingFeedback] = useState<boolean>(false)

  // Benchmarking State
  const [selectedBenchmarkVersions, setSelectedBenchmarkVersions] = useState<string[]>(['baseline'])
  const [benchPreset, setBenchPreset] = useState<PresetType>('clinical')
  const [benchQuery, setBenchQuery] = useState<string>(PRESET_QUERIES.clinical.query)
  const [benchDocId, setBenchDocId] = useState<string>(selectedDocId)
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false)
  const [benchmarkReport, setBenchmarkReport] = useState<BenchmarkReport | null>(null)
  const [, setBenchmarkHistory] = useState<BenchmarkReport[]>([])
  const [expandedBenchmarkVersion, setExpandedBenchmarkVersion] = useState<string | null>(null)

  // Lineage State
  const [lineageData, setLineageData] = useState<LineageData | null>(null)
  const [, setIsLineageLoading] = useState<boolean>(false)

  // Modal accessibility ref
  const modalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isLoggedIn) {
      fetchRuns()
      fetchVersions()
      fetchActiveWorkflow()
      fetchExecutions()
      fetchAuditEvents()
      fetchBenchmarks()
      fetchLineage()
    }
  }, [isLoggedIn])

  // Sync selected doc with benchmark
  useEffect(() => {
    if (selectedDocId) {
      setBenchDocId(selectedDocId)
    }
  }, [selectedDocId])

  // Handle ESC key for modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (confirmApprovalCandidate) setConfirmApprovalCandidate(null)
        else if (inspectModalVersion) setInspectModalVersion(null)
        else if (inspectModalCandidate) setInspectModalCandidate(null)
        else if (actionModalVersion) setActionModalVersion(null)
        else if (feedbackModalExecution) setFeedbackModalExecution(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [inspectModalCandidate, inspectModalVersion, confirmApprovalCandidate, actionModalVersion, feedbackModalExecution])

  // Select initial Pareto candidate when report loads
  useEffect(() => {
    if (report) {
      if (report.pareto_frontier && report.pareto_frontier.length > 0) {
        setSelectedCandidateId(report.pareto_frontier[0].candidate_id)
      } else if (report.all_candidates && report.all_candidates.length > 0) {
        setSelectedCandidateId(report.all_candidates[0].candidate_id)
      }
    }
  }, [report])

  const fetchRuns = async () => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/runs`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        if (data.runs && Array.isArray(data.runs)) {
          setRuns(data.runs)
          if (data.runs.length > 0 && !report) {
            setReport(data.runs[0])
          }
        }
      }
    } catch (err) {
      console.warn('Could not fetch prior AgentEvo runs', err)
    }
  }

  const fetchVersions = async () => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/versions`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        if (data.versions && Array.isArray(data.versions)) {
          setWorkflowVersions(data.versions)
          const allIds = ['baseline', ...data.versions.map((v: WorkflowVersion) => v.version_id)]
          setSelectedBenchmarkVersions(allIds)
          if (data.versions.length > 0 && crossoverParentB === 'wf_v1_60e5af') {
            const firstVer = data.versions[0].version_id
            if (firstVer !== crossoverParentA) {
              setCrossoverParentB(firstVer)
            }
          }
        }
      }
    } catch (err) {
      console.warn('Could not fetch workflow versions', err)
    }
  }

  const fetchActiveWorkflow = async () => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/active`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        const resolvedId = data.active_version_id || data.version_id || 'baseline'
        setActiveWorkflow({
          ...data,
          version_id: resolvedId,
          active_version_id: resolvedId,
        })
      }
    } catch (err) {
      console.warn('Could not fetch active workflow', err)
    }
  }

  const fetchExecutions = async () => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/executions?limit=30`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        if (data.executions && Array.isArray(data.executions)) {
          setExecutions(data.executions)
        }
      }
    } catch (err) {
      console.warn('Could not fetch executions', err)
    }
  }

  const fetchAuditEvents = async () => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/audit?limit=50`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        const items = data.events || data.audit_events
        if (items && Array.isArray(items)) {
          setAuditEvents(items)
        }
      }
    } catch (err) {
      console.warn('Could not fetch audit events', err)
    }
  }

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/benchmarks`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        if (data.benchmarks && Array.isArray(data.benchmarks)) {
          setBenchmarkHistory(data.benchmarks)
          if (data.benchmarks.length > 0 && !benchmarkReport) {
            setBenchmarkReport(data.benchmarks[0])
          }
        }
      }
    } catch (err) {
      console.warn('Could not fetch benchmark history', err)
    }
  }

  const fetchLineage = async () => {
    setIsLineageLoading(true)
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/lineage`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        setLineageData(data)
      }
    } catch (err) {
      console.warn('Could not fetch lineage graph', err)
    } finally {
      setIsLineageLoading(false)
    }
  }

  const handleInspectVersion = async (versionId: string) => {
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/versions/${versionId}/inspect`, { credentials: 'include' })
      if (res.ok) {
        const data: VersionInspectData = await res.json()
        setInspectModalVersion(data)
      } else {
        setError(`Failed to inspect version ${versionId}`)
      }
    } catch (err: any) {
      setError(err.message || 'Failed to inspect version')
    }
  }

  const handlePresetChange = (preset: PresetType) => {
    setActivePreset(preset)
    if (preset !== 'custom') {
      setQuery(PRESET_QUERIES[preset].query)
    }
  }

  const handleWorkloadPresetChange = (preset: WorkloadPresetType) => {
    setActiveWorkloadPreset(preset)
    if (preset !== 'custom_workload') {
      setWorkloadText(WORKLOAD_PRESETS[preset].queries.join('\n'))
    }
  }

  const handleBenchPresetChange = (preset: PresetType) => {
    setBenchPreset(preset)
    if (preset !== 'custom') {
      setBenchQuery(PRESET_QUERIES[preset].query)
    }
  }

  const handlePreviewCrossover = async () => {
    if (crossoverParentA === crossoverParentB) {
      setError('Parent A and Parent B cannot be the same workflow version. Please choose two distinct parents.')
      return
    }
    setError(null)
    setIsPreviewLoading(true)
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/crossover/preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          parent_a_version_id: crossoverParentA,
          parent_b_version_id: crossoverParentB,
          crossover_map: crossoverMap,
          post_crossover_mutation: postCrossoverMutation,
        }),
      })
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to preview crossover configuration.')
      }
      const data: CrossoverPreviewResponse = await res.json()
      setCrossoverPreview(data)
    } catch (err: any) {
      setError(err.message || 'Crossover preview failed.')
    } finally {
      setIsPreviewLoading(false)
    }
  }

  const handleRunOptimization = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()

    let targetQueries: string[] = []
    if (evalMode === 'workload') {
      targetQueries = workloadText
        .split('\n')
        .map((q) => q.trim())
        .filter((q) => q.length > 0)
      if (targetQueries.length === 0) {
        setError('Please provide at least one query in the Evaluation Workload.')
        return
      }
    } else {
      if (!query.trim()) {
        setError('Please provide a Target Evaluation Query.')
        return
      }
      targetQueries = [query.trim()]
    }

    if (candidateStrategy === 'crossover' && crossoverParentA === crossoverParentB) {
      setError('Parent A and Parent B cannot be the same workflow version for Crossover. Please choose two distinct parents.')
      return
    }

    setError(null)
    setRunning(true)
    setCurrentStageIndex(1) // Stage 1: Candidate Generation

    const t1 = setTimeout(() => setCurrentStageIndex(2), 1200) // Evaluating
    const t2 = setTimeout(() => setCurrentStageIndex(3), 2800) // Pareto Analysis

    try {
      const payload: Record<string, any> = {
        query: targetQueries[0],
        strategy: candidateStrategy,
      }
      if (evalMode === 'workload') {
        payload.workload = targetQueries
      }
      if (selectedDocId) {
        payload.document_id = selectedDocId
      }

      if (candidateStrategy === 'mutation') {
        payload.candidate_count = generatedCandidateCount
        if (selectedParentWorkflowId && selectedParentWorkflowId !== 'baseline') {
          payload.parent_workflow_id = selectedParentWorkflowId
        }
        if (selectedMutationType !== 'all') {
          payload.allowed_mutation_types = [selectedMutationType]
        }
      } else if (candidateStrategy === 'refinement') {
        payload.candidate_count = generatedCandidateCount
        if (selectedParentWorkflowId && selectedParentWorkflowId !== 'baseline') {
          payload.parent_workflow_id = selectedParentWorkflowId
        }
        payload.refinement_strategy = selectedRefinementType
      } else if (candidateStrategy === 'crossover') {
        payload.candidate_count = 1
        payload.parent_workflow_id = crossoverParentA
        payload.parent_b_workflow_id = crossoverParentB
        payload.crossover_map = crossoverMap
        payload.post_crossover_mutation = postCrossoverMutation
      } else if (candidateStrategy === 'adaptive') {
        payload.candidate_count = generatedCandidateCount
        if (selectedParentWorkflowId && selectedParentWorkflowId !== 'baseline') {
          payload.parent_workflow_id = selectedParentWorkflowId
        }
      }

      const res = await fetch(`${apiBase}/api/agent-evo/optimize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Optimization failed with status ${res.status}`)
      }

      setCurrentStageIndex(4) // Report / Human Review
      const newReport: OptimizationReport = await res.json()
      setReport(newReport)
      await fetchRuns()
      await fetchVersions()
      await fetchLineage()
    } catch (err: any) {
      setError(err.message || 'The optimization run could not be completed.')
    } finally {
      clearTimeout(t1)
      clearTimeout(t2)
      setRunning(false)
    }
  }

  const handleRunBenchmark = async () => {
    if (selectedBenchmarkVersions.length === 0) {
      setError('Select at least one workflow version to benchmark.')
      return
    }
    if (!benchQuery.trim()) {
      setError('Provide a benchmark query.')
      return
    }

    setError(null)
    setIsBenchmarking(true)

    try {
      const payload = {
        version_ids: selectedBenchmarkVersions,
        name: `Benchmark: ${PRESET_QUERIES[benchPreset].label} (${selectedBenchmarkVersions.length} versions)`,
        preset_name: benchPreset,
        query: benchQuery.trim(),
        document_id: benchDocId || selectedDocId || undefined,
      }

      const res = await fetch(`${apiBase}/api/agent-evo/benchmark`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Benchmark failed with status ${res.status}`)
      }

      const reportData: BenchmarkReport = await res.json()
      setBenchmarkReport(reportData)
      await fetchBenchmarks()
    } catch (err: any) {
      setError(err.message || 'Benchmark execution failed.')
    } finally {
      setIsBenchmarking(false)
    }
  }

  const handleApproveWorkflow = async (candidateId: string) => {
    if (!report) return
    setApprovingId(candidateId)
    setError(null)
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/runs/${report.run_id}/candidates/${candidateId}/approve`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Approval failed')
      }

      const resData = await res.json()
      if (resData.report) {
        setReport(resData.report)
      }
      setConfirmApprovalCandidate(null)
      await fetchVersions()
      await fetchRuns()
      await fetchLineage()
      await fetchAuditEvents()
    } catch (err: any) {
      setError(err.message || 'Failed to record developer approval')
    } finally {
      setApprovingId(null)
    }
  }

  const handleExecuteAction = async () => {
    if (!actionModalVersion) return
    const { version, type } = actionModalVersion
    setIsActionSubmitting(true)
    setError(null)
    try {
      const endpoint = type === 'activate'
        ? `${apiBase}/api/agent-evo/versions/${version.version_id}/activate`
        : `${apiBase}/api/agent-evo/versions/${version.version_id}/rollback`

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ reason: actionReason.trim() || undefined }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Failed to ${type} workflow`)
      }

      setActionModalVersion(null)
      setActionReason('')
      await fetchVersions()
      await fetchActiveWorkflow()
      await fetchAuditEvents()
    } catch (err: any) {
      setError(err.message || `Failed to ${type} workflow`)
    } finally {
      setIsActionSubmitting(false)
    }
  }

  const handleSubmitFeedback = async () => {
    if (!feedbackModalExecution) return
    setIsSubmittingFeedback(true)
    setError(null)
    try {
      const res = await fetch(`${apiBase}/api/agent-evo/executions/${feedbackModalExecution.execution_id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          rating: feedbackRating,
          feedback_text: feedbackText.trim() || undefined,
        }),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Failed to submit feedback')
      }

      setFeedbackModalExecution(null)
      setFeedbackText('')
      setFeedbackRating(5)
      await fetchExecutions()
    } catch (err: any) {
      setError(err.message || 'Failed to submit feedback')
    } finally {
      setIsSubmittingFeedback(false)
    }
  }

  const handleTriggerReoptimize = (parentVerId?: string) => {
    if (parentVerId) {
      setSelectedParentWorkflowId(parentVerId)
      setCrossoverParentA(parentVerId)
    } else if (activeWorkflow?.version_id) {
      setSelectedParentWorkflowId(activeWorkflow.version_id)
      setCrossoverParentA(activeWorkflow.version_id)
    }
    setActiveMainTab('optimizer')
  }

  const toggleBenchmarkVersion = (vId: string) => {
    setSelectedBenchmarkVersions((prev) =>
      prev.includes(vId) ? prev.filter((id) => id !== vId) : [...prev, vId]
    )
  }

  const selectAllVersionsForBenchmark = () => {
    const all = ['baseline', ...workflowVersions.map((v) => v.version_id)]
    setSelectedBenchmarkVersions(all)
    setActiveMainTab('benchmark')
  }

  const renderMutationBadge = (cand: EvaluatedCandidate) => {
    const meta = cand.mutation_metadata || cand.workflow.mutation_metadata
    if (!meta) {
      return <span className="evo-mut-tag mut-baseline">baseline</span>
    }
    const type = meta.mutation_type || 'custom'
    return (
      <span className={`evo-mut-tag mut-${type} ${type === 'refinement' ? 'evo-mut-refinement' : type === 'crossover' ? 'evo-mut-crossover' : ''}`}>
        {type.toUpperCase()}
      </span>
    )
  }

  const baselineCandidate = report?.baseline || report?.all_candidates?.find((c) => c.candidate_id === 'baseline') || null

  // Active workflow verified benchmark vs snapshot metrics
  const activeVersionId = activeWorkflow?.version_id || 'baseline'
  const activeBenchmarkResult = benchmarkReport?.results?.find(
    (r) => r.version_id === activeVersionId || (activeVersionId === 'baseline' && (r.version_id === 'baseline' || r.version === 'baseline'))
  )
  const isVerifiedBenchmark = !!activeBenchmarkResult
  const activeGroundedness = isVerifiedBenchmark
    ? (activeBenchmarkResult.groundedness !== undefined && activeBenchmarkResult.groundedness !== null ? activeBenchmarkResult.groundedness : null)
    : (activeWorkflow?.evaluation_snapshot?.groundedness ?? activeWorkflow?.evaluation_snapshot?.performance ?? null)

  const activeCost = isVerifiedBenchmark
    ? (activeBenchmarkResult.cost_per_query ?? activeBenchmarkResult.cost ?? null)
    : (activeWorkflow?.evaluation_snapshot?.cost_per_query ?? activeWorkflow?.evaluation_snapshot?.cost ?? null)

  const activeLatency = isVerifiedBenchmark
    ? (activeBenchmarkResult.latency_ms ?? activeBenchmarkResult.llm_latency_ms ?? null)
    : (activeWorkflow?.evaluation_snapshot?.llm_latency_ms ?? activeWorkflow?.evaluation_snapshot?.latency_ms ?? null)

  const selectedCandidate = report?.all_candidates?.find((c) => c.candidate_id === selectedCandidateId) ||
    report?.pareto_frontier?.[0] ||
    report?.all_candidates?.[0] ||
    null

  return (
    <div className="evo-container">
      {/* Top Header */}
      <div className="evo-page-header">
        <div>
          <h1 className="evo-title">AgentEvo Evolutionary Workflow Optimization</h1>
          <p className="evo-subtitle">
            Autonomous multi-candidate evolution, crossover recombination, Phase 6 claim-groundedness evaluation, Pareto optimization, and production lifecycle governance.
          </p>
        </div>
        <div className="evo-header-meta">
          {activeWorkflow ? (
            <span className="evo-status-pill" style={{ background: 'rgba(16, 185, 129, 0.12)', borderColor: 'rgba(16, 185, 129, 0.3)', color: '#34d399' }}>
              <span className="evo-dot" style={{ background: '#10b981' }} /> Active: {activeWorkflow.version_id === 'baseline' ? 'Baseline' : `v${activeWorkflow.version_number ?? 0}`}
            </span>
          ) : (
            <span className="evo-status-pill">
              <span className="evo-dot" /> Autonomous Evolutionary RAG
            </span>
          )}
        </div>
      </div>

      {/* 6-Section Purpose-Built Developer Navigation */}
      <div className="evo-nav-tabs">
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'overview' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('overview')
            fetchActiveWorkflow()
            fetchVersions()
            fetchExecutions()
            fetchAuditEvents()
          }}
        >
          Overview
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'evolution' ? 'active' : ''}`}
          onClick={() => setActiveMainTab('evolution')}
        >
          Evolution Lab
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'review' ? 'active' : ''}`}
          onClick={() => setActiveMainTab('review')}
        >
          Candidate Review
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'production' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('production')
            fetchVersions()
            fetchActiveWorkflow()
            fetchAuditEvents()
            fetchLineage()
          }}
        >
          Production & Versions
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'executions' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('executions')
            fetchExecutions()
            fetchActiveWorkflow()
          }}
        >
          Execution & Evidence
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'benchmarking' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('benchmarking')
            fetchBenchmarks()
          }}
        >
          Benchmarking
        </button>
      </div>

      {error && (
        <div className="evo-error-banner" role="alert">
          <span className="evo-error-icon">⚠️</span>
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} className="evo-error-dismiss" aria-label="Dismiss error">✕</button>
        </div>
      )}

      {/* ========================================================================= */}
      {/* SECTION A: OVERVIEW & RUNTIME DASHBOARD                                   */}
      {/* ========================================================================= */}
      {activeMainTab === 'overview' && (
        <>
          {/* Active Production Workflow Banner */}
          <section className="evo-section evo-active-banner">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                  <span className="evo-status-tag tag-active" style={{ fontSize: '0.82rem', padding: '3px 10px', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
                    ● Active Production Deployment
                  </span>
                  <span className="evo-version-tag" style={{ background: '#10b981' }}>
                    {activeWorkflow?.version_id === 'baseline' ? 'Production Baseline' : `v${activeWorkflow?.version_number ?? 0}`}
                  </span>
                </div>
                <h2 className="evo-section-title" style={{ margin: '4px 0 6px 0', color: '#f8fafc', fontSize: '1.35rem' }}>
                  {activeWorkflow?.name || 'Production Baseline'}
                </h2>
                <div className="evo-cell-sub" style={{ color: '#94a3b8' }}>
                  Version ID: <code>{activeWorkflow?.version_id || 'baseline'}</code> |
                  Model: <code>{activeWorkflow?.model_id || 'gemini-3.5-flash-lite'}</code> ({activeWorkflow?.provider || 'Google Gemini'}) |
                  Top-K: <code>{activeWorkflow?.configuration?.retrieval?.top_k ?? (activeWorkflow?.version_id === 'baseline' ? 5 : '—')}</code> |
                  Sim: <code>{activeWorkflow?.configuration?.retrieval?.similarity_threshold ?? 0.35}</code> |
                  Activated: {activeWorkflow?.activated_at ? new Date(activeWorkflow.activated_at).toLocaleString() : 'System Genesis'}
                </div>
              </div>

              {/* Primary Action Center */}
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                <button
                  type="button"
                  onClick={() => setActiveMainTab('evolution')}
                  className="evo-btn-primary"
                >
                  Optimize Workflow
                </button>
                <button
                  type="button"
                  onClick={() => setActiveMainTab('benchmarking')}
                  className="evo-btn-secondary"
                >
                  Compare Versions
                </button>
                <button
                  type="button"
                  onClick={() => setActiveMainTab('production')}
                  className="evo-btn-secondary"
                >
                  View Production
                </button>
                <button
                  type="button"
                  onClick={() => setActiveMainTab('executions')}
                  className="evo-btn-secondary"
                >
                  Inspect Traces
                </button>
              </div>
            </div>
          </section>

          {/* High-Density KPI Cards (Clear Distinction: Verified Benchmark vs Snapshot Metric) */}
          <div className="evo-summary-grid" style={{ marginTop: 0 }}>
            {/* Groundedness Card */}
            <div className="evo-metric-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span className="evo-metric-label">Active Groundedness</span>
                {isVerifiedBenchmark ? (
                  <span className="evo-metric-source-badge verified">Verified Benchmark</span>
                ) : activeGroundedness !== null ? (
                  <span className="evo-metric-source-badge snapshot">Snapshot Metric</span>
                ) : (
                  <span className="evo-metric-source-badge na">N/A</span>
                )}
              </div>
              <span className="evo-metric-value evo-val-green">
                {activeGroundedness !== null ? `${(activeGroundedness * 100).toFixed(1)}%` : 'N/A'}
              </span>
              <span className="evo-metric-delta">
                {isVerifiedBenchmark
                  ? `Head-to-head verified on "${benchmarkReport?.preset_name || 'Clinical Suite'}"`
                  : activeGroundedness !== null
                  ? 'Recorded during initial approval evaluation snapshot'
                  : 'Run benchmark to record verified measurement'}
              </span>
            </div>

            {/* Cost / Query Card */}
            <div className="evo-metric-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span className="evo-metric-label">Cost / Query</span>
                {isVerifiedBenchmark ? (
                  <span className="evo-metric-source-badge verified">Verified Benchmark</span>
                ) : activeCost !== null ? (
                  <span className="evo-metric-source-badge snapshot">Snapshot Metric</span>
                ) : (
                  <span className="evo-metric-source-badge na">N/A</span>
                )}
              </div>
              <span className="evo-metric-value">
                {activeCost !== null ? `$${activeCost.toFixed(6)}` : 'N/A'}
              </span>
              <span className="evo-metric-delta">
                Active Model: <code>{activeWorkflow?.model_id || 'gemini-3.5-flash-lite'}</code>
              </span>
            </div>

            {/* Latency Card */}
            <div className="evo-metric-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span className="evo-metric-label">LLM Latency</span>
                {isVerifiedBenchmark ? (
                  <span className="evo-metric-source-badge verified">Verified Benchmark</span>
                ) : activeLatency !== null ? (
                  <span className="evo-metric-source-badge snapshot">Snapshot Metric</span>
                ) : (
                  <span className="evo-metric-source-badge na">N/A</span>
                )}
              </div>
              <span className="evo-metric-value evo-val-purple">
                {activeLatency !== null ? `${activeLatency.toFixed(0)} ms` : 'N/A'}
              </span>
              <span className="evo-metric-delta">
                Runtime model response latency
              </span>
            </div>

            {/* Retrieval Slicing Card */}
            <div className="evo-metric-card">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span className="evo-metric-label">Retrieval Slicing</span>
                <span className="evo-metric-source-badge verified">Runtime Enforced</span>
              </div>
              <span className="evo-metric-value">
                top_k = {activeWorkflow?.configuration?.retrieval?.top_k ?? (activeWorkflow?.version_id === 'baseline' ? 5 : 3)}
              </span>
              <span className="evo-metric-delta">
                {activeWorkflow?.configuration?.retrieval?.top_k !== undefined && activeWorkflow.configuration.retrieval.top_k < 5
                  ? `Reduced top-k slice enforced (${activeWorkflow.configuration.retrieval.top_k})`
                  : 'Active runtime retrieval chunk limit'}
              </span>
            </div>
          </div>

          {/* System Telemetry & Governance Status Summary */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
            {/* Optimizer Activity Card */}
            <div className="evo-section" style={{ margin: 0, padding: 18 }}>
              <div className="evo-section-header" style={{ marginBottom: 8 }}>
                <h3 className="evo-section-title" style={{ fontSize: '0.98rem' }}>Optimizer Engine</h3>
                <span className={`evo-status-tag ${running ? 'tag-pareto' : 'tag-baseline'}`}>
                  {running ? 'EVALUATING' : report ? 'READY' : 'STANDBY'}
                </span>
              </div>
              <p style={{ margin: 0, fontSize: '0.84rem', color: '#cbd5e1', lineHeight: 1.5 }}>
                {report
                  ? `Last session: Run ${report.run_id} evaluated ${report.all_candidates?.length || 0} candidate workflows (${report.pareto_frontier?.length || 0} Pareto-optimal).`
                  : 'Optimizer ready. Select evaluation workload or single query in the Evolution Lab to optimize workflows.'}
              </p>
              {report?.scheduler_decision && (
                <div style={{ fontSize: '0.78rem', color: '#94a3b8', marginTop: 8, padding: '6px 8px', background: 'rgba(99, 102, 241, 0.08)', borderRadius: 4, borderLeft: '3px solid #6366f1' }}>
                  Scheduler Stage: <strong style={{ color: '#c7d2fe' }}>{(report.scheduler_decision.stage_name || report.scheduler_decision.strategy || report.scheduler_decision.stage || 'Unavailable').toUpperCase()}</strong> — {report.scheduler_decision.reason || 'Deterministic scheduler decision'}
                </div>
              )}
            </div>

            {/* Version Registry Card */}
            <div className="evo-section" style={{ margin: 0, padding: 18 }}>
              <div className="evo-section-header" style={{ marginBottom: 8 }}>
                <h3 className="evo-section-title" style={{ fontSize: '0.98rem' }}>Version Registry & Lineage</h3>
                <span className="evo-version-tag">{workflowVersions.length} Approved</span>
              </div>
              <p style={{ margin: 0, fontSize: '0.84rem', color: '#cbd5e1', lineHeight: 1.5 }}>
                {workflowVersions.length} immutable approved workflow versions stored in PostgreSQL.
                Full provenance graph with mutation and crossover lineage available.
              </p>
              <div style={{ marginTop: 10 }}>
                <button
                  type="button"
                  onClick={() => {
                    setActiveMainTab('production')
                    setShowLineageInProduction(true)
                  }}
                  className="evo-btn-ghost-sm"
                  style={{ color: '#818cf8', borderColor: 'rgba(99, 102, 241, 0.3)' }}
                >
                  Inspect Lineage Graph ➔
                </button>
              </div>
            </div>

            {/* Governance Audit Trail Card */}
            <div className="evo-section" style={{ margin: 0, padding: 18 }}>
              <div className="evo-section-header" style={{ marginBottom: 8 }}>
                <h3 className="evo-section-title" style={{ fontSize: '0.98rem' }}>Governance Audit Log</h3>
                <span className="evo-status-pill">{auditEvents.length} Events</span>
              </div>
              {auditEvents.length > 0 ? (
                <div>
                  <div style={{ fontSize: '0.84rem', color: '#f1f5f9', fontWeight: 600 }}>
                    Latest: <span className="evo-status-tag tag-pareto" style={{ fontSize: '0.72rem' }}>{auditEvents[0].event_type}</span> on <code>{auditEvents[0].workflow_version_id || 'System'}</code>
                  </div>
                  <div className="evo-cell-sub" style={{ marginTop: 4 }}>
                    By {auditEvents[0].triggered_by || 'system'} • {new Date(auditEvents[0].created_at).toLocaleString()}
                  </div>
                </div>
              ) : (
                <p style={{ margin: 0, fontSize: '0.84rem', color: '#cbd5e1' }}>No audit events logged yet.</p>
              )}
            </div>
          </div>
        </>
      )}

      {/* ========================================================================= */}
      {/* SECTION B: EVOLUTION LAB (Candidate Generation & Pareto Frontier)          */}
      {/* ========================================================================= */}
      {(activeMainTab === 'evolution' || (activeMainTab as any) === 'optimizer') && (
        <>
          <section className="evo-section evo-controls-card">
            <div className="evo-section-header">
              <h2 className="evo-section-title">Optimization Setup & Candidate Generation</h2>
              <span className="evo-section-caption">Configure evaluation preset, target context, and evolutionary candidate generation strategy.</span>
            </div>

            <form onSubmit={handleRunOptimization} className="evo-form">
              {/* Evaluation Mode Toggle: Single Query vs Evaluation Workload */}
              <div className="evo-form-group">
                <label className="evo-label">Evaluation Mode</label>
                <div className="evo-strat-tabs" style={{ marginBottom: 12 }}>
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${evalMode === 'single' ? 'active' : ''}`}
                    onClick={() => setEvalMode('single')}
                    disabled={running}
                    style={{ flex: 1 }}
                  >
                    <div>
                      <strong>Single Query Mode</strong>
                      <div className="evo-cell-sub">Evaluate candidates on a single target query</div>
                    </div>
                  </button>
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${evalMode === 'workload' ? 'active' : ''}`}
                    onClick={() => setEvalMode('workload')}
                    disabled={running}
                    style={{ flex: 1 }}
                  >
                    <div>
                      <strong>Evaluation Workload Mode</strong>
                      <div className="evo-cell-sub">Evaluate candidates across multi-query workload</div>
                    </div>
                  </button>
                </div>
              </div>

              {/* SINGLE QUERY MODE CONTROLS */}
              {evalMode === 'single' ? (
                <>
                  <div className="evo-form-group">
                    <label className="evo-label">Single Query Preset</label>
                    <div className="evo-preset-buttons">
                      {(Object.keys(PRESET_QUERIES) as PresetType[]).map((presetKey) => (
                        <button
                          key={presetKey}
                          type="button"
                          onClick={() => handlePresetChange(presetKey)}
                          className={`evo-preset-btn ${activePreset === presetKey ? 'active' : ''}`}
                          disabled={running}
                        >
                          {PRESET_QUERIES[presetKey].label}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="evo-form-group">
                    <label htmlFor="evo-query-input" className="evo-label">Target Evaluation Query</label>
                    <textarea
                      id="evo-query-input"
                      className="evo-textarea"
                      value={query}
                      onChange={(e) => {
                        setQuery(e.target.value)
                        setActivePreset('custom')
                      }}
                      rows={2}
                      placeholder="Enter single medical query to evaluate candidates against..."
                      disabled={running}
                    />
                    <span className="evo-cell-sub" style={{ marginTop: 4 }}>
                      Single-query evaluation optimizes parameters specifically for this query.
                    </span>
                  </div>
                </>
              ) : (
                /* EVALUATION WORKLOAD MODE CONTROLS */
                <>
                  <div className="evo-form-group">
                    <label className="evo-label">Workload Preset</label>
                    <div className="evo-preset-buttons">
                      {(Object.keys(WORKLOAD_PRESETS) as WorkloadPresetType[]).map((presetKey) => (
                        <button
                          key={presetKey}
                          type="button"
                          onClick={() => handleWorkloadPresetChange(presetKey)}
                          className={`evo-preset-btn ${activeWorkloadPreset === presetKey ? 'active' : ''}`}
                          disabled={running}
                        >
                          {WORKLOAD_PRESETS[presetKey].label}
                        </button>
                      ))}
                    </div>
                    <span className="evo-cell-sub" style={{ marginTop: 4 }}>
                      {WORKLOAD_PRESETS[activeWorkloadPreset].description}
                    </span>
                  </div>

                  <div className="evo-form-group">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <label htmlFor="evo-workload-input" className="evo-label" style={{ margin: 0 }}>
                        Evaluation Workload (One Query per Line)
                      </label>
                      <span className="evo-version-tag" style={{ background: '#3b82f6' }}>
                        {workloadText.split('\n').filter((q) => q.trim().length > 0).length} Representative Queries
                      </span>
                    </div>
                    <textarea
                      id="evo-workload-input"
                      className="evo-textarea"
                      value={workloadText}
                      onChange={(e) => {
                        setWorkloadText(e.target.value)
                        setActiveWorkloadPreset('custom_workload')
                      }}
                      rows={5}
                      placeholder="Enter one evaluation query per line..."
                      disabled={running}
                      style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}
                    />
                    <span className="evo-cell-sub" style={{ marginTop: 4 }}>
                      Every workflow candidate will be evaluated across all {workloadText.split('\n').filter((q) => q.trim().length > 0).length} queries. Aggregate averages are used for Pareto comparison.
                    </span>
                  </div>
                </>
              )}

              {/* Document Selection */}
              <div className="evo-form-group">
                <label htmlFor="evo-doc-select" className="evo-label">Target Document Context</label>
                <select
                  id="evo-doc-select"
                  className="evo-select"
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  disabled={running}
                >
                  <option value="">All Uploaded Documents (Global Knowledge Base)</option>
                  {documents.map((doc) => (
                    <option key={doc.id} value={doc.id}>
                      {doc.filename || doc.original_filename || doc.id}
                    </option>
                  ))}
                </select>
              </div>

              {/* Strategy Selector: Adaptive | Mutation / Exploration | Refinement | Crossover / Recombination */}
              <div className="evo-form-group" style={{ marginTop: 16 }}>
                <label className="evo-label">Candidate Generation Strategy</label>
                <div className="evo-strat-tabs">
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${candidateStrategy === 'adaptive' ? 'active' : ''}`}
                    onClick={() => setCandidateStrategy('adaptive')}
                    disabled={running}
                  >
                    <div>
                      <strong>Adaptive Strategy</strong>
                      <div className="evo-cell-sub">Deterministic Evolutionary Scheduler</div>
                    </div>
                  </button>
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${candidateStrategy === 'mutation' ? 'active' : ''}`}
                    onClick={() => setCandidateStrategy('mutation')}
                    disabled={running}
                  >
                    <div>
                      <strong>Mutation / Exploration</strong>
                      <div className="evo-cell-sub">Stochastic parameter & prompt search</div>
                    </div>
                  </button>
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${candidateStrategy === 'refinement' ? 'active' : ''}`}
                    onClick={() => setCandidateStrategy('refinement')}
                    disabled={running}
                  >
                    <div>
                      <strong>Targeted Refinement</strong>
                      <div className="evo-cell-sub">Deterministic simplification & compression</div>
                    </div>
                  </button>
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${candidateStrategy === 'crossover' ? 'active' : ''}`}
                    onClick={() => setCandidateStrategy('crossover')}
                    disabled={running}
                  >
                    <div>
                      <strong>Crossover / Recombination</strong>
                      <div className="evo-cell-sub">Inherit components from two parent workflows</div>
                    </div>
                  </button>
                </div>
              </div>

              {/* Advanced Strategy Details Accordion Toggle */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12, marginBottom: 8 }}>
                <span className="evo-label" style={{ margin: 0 }}>Advanced Strategy Configuration</span>
                <button
                  type="button"
                  onClick={() => setShowAdvancedStrategyDetails(!showAdvancedStrategyDetails)}
                  className="evo-btn-ghost-sm"
                  style={{ color: '#818cf8', borderColor: 'rgba(99, 102, 241, 0.3)', display: 'inline-flex', alignItems: 'center', gap: 4 }}
                >
                  {showAdvancedStrategyDetails ? 'Hide Advanced Parameters ▴' : 'Configure Advanced Parameters ▾'}
                </button>
              </div>

              {/* Expandable Advanced Strategy Details */}
              {showAdvancedStrategyDetails && (
                <>
              {/* STRATEGY 0: ADAPTIVE SCHEDULER CONFIGURATION */}
              {candidateStrategy === 'adaptive' && (
                <div className="evo-strategy-config-box">
                  <div className="evo-form-row">
                    <div className="evo-form-group evo-col-half">
                      <label htmlFor="evo-adaptive-parent" className="evo-label">Initial Parent Workflow</label>
                      <select
                        id="evo-adaptive-parent"
                        className="evo-select"
                        value={selectedParentWorkflowId}
                        onChange={(e) => setSelectedParentWorkflowId(e.target.value)}
                        disabled={running}
                      >
                        <option value="baseline">Production Baseline (Default Reference)</option>
                        {workflowVersions.map((v) => (
                          <option key={v.version_id} value={v.version_id}>
                            v{v.version_number}: {v.name} ({v.model_id}) [{v.status}]
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="evo-form-group evo-col-half">
                      <label htmlFor="evo-adaptive-count" className="evo-label">Candidates per Iteration ({generatedCandidateCount})</label>
                      <input
                        id="evo-adaptive-count"
                        type="range"
                        min={2}
                        max={6}
                        value={generatedCandidateCount}
                        onChange={(e) => setGeneratedCandidateCount(Number(e.target.value))}
                        className="evo-slider"
                        disabled={running}
                      />
                    </div>
                  </div>

                  <div className="evo-summary-box" style={{ marginTop: 10, background: 'rgba(59, 130, 246, 0.06)', borderLeft: '3px solid #3b82f6' }}>
                    <div style={{ fontWeight: 600, color: '#93c5fd', marginBottom: 4 }}>
                      🤖 AgentEvo Adapter Scheduler (Deterministic Policy)
                    </div>
                    <p style={{ margin: 0, fontSize: '0.84rem', color: '#cbd5e1', lineHeight: 1.5 }}>
                      The adaptive scheduler analyzes optimization history, Pareto frontier progress, groundedness retention, and stagnation to select the optimal evolutionary mechanism (Mutation, Refinement, or Crossover).
                    </p>
                  </div>
                </div>
              )}

              {/* STRATEGY 1: MUTATION / EXPLORATION CONFIGURATION */}
              {candidateStrategy === 'mutation' && (
                <div className="evo-strategy-config-box">
                  <div className="evo-form-row">
                    <div className="evo-form-group evo-col-half">
                      <label htmlFor="evo-parent-select" className="evo-label">Parent Workflow</label>
                      <select
                        id="evo-parent-select"
                        className="evo-select"
                        value={selectedParentWorkflowId}
                        onChange={(e) => setSelectedParentWorkflowId(e.target.value)}
                        disabled={running}
                      >
                        <option value="baseline">Production Baseline (Default Reference)</option>
                        {workflowVersions.map((v) => (
                          <option key={v.version_id} value={v.version_id}>
                            v{v.version_number}: {v.name} ({v.model_id}) [{v.status}]
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="evo-form-group evo-col-half">
                      <label htmlFor="evo-count-input" className="evo-label">Candidate Count ({generatedCandidateCount})</label>
                      <input
                        id="evo-count-input"
                        type="range"
                        min={2}
                        max={8}
                        value={generatedCandidateCount}
                        onChange={(e) => setGeneratedCandidateCount(Number(e.target.value))}
                        className="evo-slider"
                        disabled={running}
                      />
                    </div>
                  </div>

                  <div className="evo-form-group">
                    <label className="evo-label">Mutation Dimension</label>
                    <div className="evo-strategy-pills-wrap">
                      {[
                        { id: 'all', label: 'All Dimensions', desc: 'Broad parameter & prompt exploration' },
                        { id: 'prompt', label: 'Prompt Only', desc: 'Template & disclaimer tuning' },
                        { id: 'retrieval', label: 'Retrieval Only', desc: 'Top-K & similarity variations' },
                        { id: 'model', label: 'Model Only', desc: 'Provider & temperature exploration' },
                        { id: 'operator', label: 'Operators Only', desc: 'Context compression toggle' },
                        { id: 'composite', label: 'Composite Mutation', desc: 'Multi-parameter exploration' },
                      ].map((s) => (
                        <button
                          key={s.id}
                          type="button"
                          onClick={() => setSelectedMutationType(s.id)}
                          className={`evo-strat-btn ${selectedMutationType === s.id ? 'active' : ''}`}
                          disabled={running}
                        >
                          <span className="evo-strat-btn-title">{s.label}</span>
                          <span className="evo-strat-btn-desc">{s.desc}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* STRATEGY 2: REFINEMENT CONFIGURATION */}
              {candidateStrategy === 'refinement' && (
                <div className="evo-strategy-config-box">
                  <div className="evo-form-row">
                    <div className="evo-form-group evo-col-half">
                      <label className="evo-label">Parent Workflow</label>
                      <select
                        className="evo-select"
                        value={selectedParentWorkflowId}
                        onChange={(e) => setSelectedParentWorkflowId(e.target.value)}
                        disabled={running}
                      >
                        <option value="baseline">Production Baseline (Default Reference)</option>
                        {workflowVersions.map((v) => (
                          <option key={v.version_id} value={v.version_id}>
                            v{v.version_number}: {v.name} ({v.model_id}) [{v.status}]
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="evo-form-group evo-col-half">
                      <label className="evo-label">Candidate Count ({generatedCandidateCount})</label>
                      <input
                        type="range"
                        min={2}
                        max={6}
                        value={generatedCandidateCount}
                        onChange={(e) => setGeneratedCandidateCount(Number(e.target.value))}
                        className="evo-slider"
                        disabled={running}
                      />
                    </div>
                  </div>

                  <div className="evo-form-group">
                    <label className="evo-label">Refinement Operation</label>
                    <div className="evo-strategy-pills-wrap">
                      {[
                        { id: 'all', label: 'All Refinements', desc: 'Comprehensive targeted simplification' },
                        { id: 'simplify', label: 'Top-K Simplification', desc: 'Reduce retrieval overhead' },
                        { id: 'context_compress', label: 'Context Compression', desc: 'Enable semantic token compression' },
                        { id: 'rerank_disable', label: 'Disable Rerank', desc: 'Latency and compute reduction' },
                        { id: 'composite_refine', label: 'Composite Refinement', desc: 'Multi-aspect cost & latency reduction' },
                      ].map((s) => (
                        <button
                          key={s.id}
                          type="button"
                          onClick={() => setSelectedRefinementType(s.id)}
                          className={`evo-strat-btn ${selectedRefinementType === s.id ? 'active' : ''}`}
                          disabled={running}
                        >
                          <span className="evo-strat-btn-title">{s.label}</span>
                          <span className="evo-strat-btn-desc">{s.desc}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* STRATEGY 3: CROSSOVER / RECOMBINATION CONFIGURATION */}
              {candidateStrategy === 'crossover' && (() => {
                const parentAOptions = [
                  { value: 'baseline', label: 'Production Baseline (Gemini 3.5 Flash-Lite, k=5)' },
                  ...workflowVersions.map((v) => ({
                    value: v.version_id,
                    label: `v${v.version_number}: ${v.name} (${v.model_id}) [${v.status}]`,
                  })),
                ]
                const parentBOptions = parentAOptions.filter((opt) => opt.value !== crossoverParentA)

                return (
                  <div className="evo-strategy-config-box">
                    <div className="evo-form-row">
                      <div className="evo-form-group evo-col-half">
                        <label className="evo-label" style={{ color: '#60a5fa' }}>Parent A (Primary / Scaffold)</label>
                        <select
                          value={crossoverParentA}
                          onChange={(e) => {
                            const newA = e.target.value
                            setCrossoverParentA(newA)
                            if (crossoverParentB === newA) {
                              const remaining = parentAOptions.filter((opt) => opt.value !== newA)
                              if (remaining.length > 0) {
                                setCrossoverParentB(remaining[0].value)
                              }
                            }
                          }}
                          className="evo-select"
                          disabled={running}
                        >
                          {parentAOptions.map((opt) => (
                            <option key={opt.value} value={opt.value}>
                              {opt.label}
                            </option>
                          ))}
                        </select>
                      </div>

                      <div className="evo-form-group evo-col-half">
                        <label className="evo-label" style={{ color: '#a78bfa' }}>Parent B (Secondary / Donor)</label>
                        <select
                          value={crossoverParentB}
                          onChange={(e) => setCrossoverParentB(e.target.value)}
                          className="evo-select"
                          disabled={running}
                        >
                          {parentBOptions.length === 0 ? (
                            <option value="">No other approved versions available</option>
                          ) : (
                            parentBOptions.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {opt.label}
                              </option>
                            ))
                          )}
                        </select>
                      </div>
                    </div>

                    {crossoverParentA === crossoverParentB && (
                      <div className="evo-error-banner" style={{ marginTop: 8, marginBottom: 12 }}>
                        <span>⚠️ Parent A and Parent B must be distinct workflow versions.</span>
                      </div>
                    )}

                    {/* Component-Level Inheritance Table */}
                    <div className="evo-table-container" style={{ marginTop: 16 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <label className="evo-label" style={{ margin: 0 }}>Component-Level Inheritance Map</label>
                        <span className="evo-cell-sub">Select inheritance source for each workflow module</span>
                      </div>
                      <table className="evo-table">
                        <thead>
                          <tr>
                            <th>Workflow Module</th>
                            <th>Parent A Source</th>
                            <th>Inherit From</th>
                            <th>Parent B Source</th>
                          </tr>
                        </thead>
                        <tbody>
                          {[
                            { key: 'retrieval', label: 'Retrieval Configuration', desc: 'Chunk size, overlap, top-K, similarity threshold, rerank' },
                            { key: 'prompt', label: 'Prompt Template', desc: 'Prompt template, grounding instructions, custom guidance' },
                            { key: 'model', label: 'LLM Model Spec', desc: 'Provider, Model ID' },
                            { key: 'generation', label: 'Generation Parameters', desc: 'Temperature, max token limits' },
                            { key: 'operators', label: 'Operators / Execution', desc: 'Context compression, timeout, max retries' },
                          ].map(({ key, label, desc }) => (
                            <tr key={key} className="evo-table-row">
                              <td>
                                <strong>{label}</strong>
                                <div className="evo-cell-sub">{desc}</div>
                              </td>
                              <td>
                                <span className="evo-version-tag" style={{ background: '#3b82f6' }}>Parent A</span>
                              </td>
                              <td>
                                <select
                                  value={crossoverMap[key] || 'parent_a'}
                                  onChange={(e) => setCrossoverMap((prev) => ({ ...prev, [key]: e.target.value as 'parent_a' | 'parent_b' }))}
                                  className="evo-select evo-select-sm"
                                  disabled={running}
                                  style={{ minWidth: '120px', fontWeight: 600 }}
                                >
                                  <option value="parent_a">Parent A</option>
                                  <option value="parent_b">Parent B</option>
                                </select>
                              </td>
                              <td>
                                <span className="evo-version-tag" style={{ background: '#8b5cf6' }}>Parent B</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Post-Crossover Mutation Selection */}
                    <div className="evo-form-row" style={{ marginTop: 16 }}>
                      <div className="evo-form-group evo-col-half">
                        <label className="evo-label">Optional Post-Crossover Mutation</label>
                        <select
                          value={postCrossoverMutation}
                          onChange={(e) => setPostCrossoverMutation(e.target.value)}
                          className="evo-select"
                          disabled={running}
                        >
                          <option value="none">None / Pure Crossover (No Mutation)</option>
                          <option value="prompt">Prompt Variation (Stochastic Template Mutation)</option>
                          <option value="retrieval">Retrieval Tuning (Top-K / Similarity Adjust)</option>
                          <option value="model">Deterministic Model (Low-Temperature Calibration)</option>
                          <option value="operator">Toggle Compression (Invert Context Compression)</option>
                          <option value="composite">Composite Mutation (Multi-Component Refinement)</option>
                        </select>
                      </div>
                      <div className="evo-form-group evo-col-half" style={{ display: 'flex', alignItems: 'flex-end' }}>
                        <button
                          type="button"
                          onClick={handlePreviewCrossover}
                          className="evo-btn-secondary"
                          disabled={isPreviewLoading || running || crossoverParentA === crossoverParentB}
                          style={{ width: '100%', height: '38px' }}
                        >
                          {isPreviewLoading ? 'Assembling Preview...' : '🔍 Preview Child Workflow Genome'}
                        </button>
                      </div>
                    </div>

                    {/* Crossover Genome Preview Card */}
                    {crossoverPreview && (
                      <div className="evo-summary-box" style={{ marginTop: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                          <strong>🧬 Child Workflow Genome Preview:</strong>
                          <span className="evo-status-tag tag-pareto">Valid Complete Configuration</span>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 10 }}>
                          <div className="evo-metric-cell">
                            <span className="evo-metric-label">Model</span>
                            <span className="evo-metric-val" style={{ fontSize: '0.9rem' }}>{crossoverPreview.resulting_configuration.model?.model_id}</span>
                            <span className="evo-cell-sub">Source: {crossoverPreview.crossover_map.model === 'parent_a' ? 'Parent A' : 'Parent B'}</span>
                          </div>
                          <div className="evo-metric-cell">
                            <span className="evo-metric-label">Retrieval (top_k / sim)</span>
                            <span className="evo-metric-val" style={{ fontSize: '0.9rem' }}>
                              k={crossoverPreview.resulting_configuration.retrieval?.top_k}, sim={crossoverPreview.resulting_configuration.retrieval?.similarity_threshold}
                            </span>
                            <span className="evo-cell-sub">Source: {crossoverPreview.crossover_map.retrieval === 'parent_a' ? 'Parent A' : 'Parent B'}</span>
                          </div>
                          <div className="evo-metric-cell">
                            <span className="evo-metric-label">Prompt Template</span>
                            <span className="evo-metric-val" style={{ fontSize: '0.9rem' }}>{crossoverPreview.resulting_configuration.prompt?.template_name}</span>
                            <span className="evo-cell-sub">Source: {crossoverPreview.crossover_map.prompt === 'parent_a' ? 'Parent A' : 'Parent B'}</span>
                          </div>
                          <div className="evo-metric-cell">
                            <span className="evo-metric-label">Compression</span>
                            <span className="evo-metric-val" style={{ fontSize: '0.9rem' }}>
                              {crossoverPreview.resulting_configuration.execution?.context_compression ? 'Enabled' : 'Disabled'}
                            </span>
                            <span className="evo-cell-sub">Source: {crossoverPreview.crossover_map.operators === 'parent_a' ? 'Parent A' : 'Parent B'}</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )
              })()}
                </>
              )}

              {/* Submit / Launch Optimization Button */}
              <div className="evo-form-actions" style={{ marginTop: 20 }}>
                <button
                  type="submit"
                  className="evo-btn-primary"
                  disabled={running || (candidateStrategy === 'crossover' && crossoverParentA === crossoverParentB)}
                >
                  {running ? (
                    <>
                      <span className="evo-spinner" /> Generating & Evaluating Candidates...
                    </>
                  ) : (
                    'Generate & Evaluate Candidates'
                  )}
                </button>
              </div>
            </form>
          </section>

          {/* Progress Tracker */}
          {running && (
            <div className="evo-progress-card">
              <div className="evo-stepper">
                {[
                  { step: 1, label: 'Candidate Generation' },
                  { step: 2, label: 'Phase 6 Claim Evaluation' },
                  { step: 3, label: 'Pareto Frontier Analysis' },
                  { step: 4, label: 'Human Governance Gate' },
                ].map((st) => (
                  <div
                    key={st.step}
                    className={`evo-step ${currentStageIndex >= st.step ? 'active' : ''} ${currentStageIndex > st.step ? 'complete' : ''}`}
                  >
                    <div className="evo-step-circle">{st.step}</div>
                    <span className="evo-step-label">{st.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Pareto Optimization Report Section */}
          {report && (
            <div className="evo-report-container">
              {/* Header */}
              <div className="evo-report-header">
                <div>
                  <div className="evo-report-title-row">
                    <h2 className="evo-report-title">AgentEvo Optimization Run</h2>
                    <span className="evo-run-id-pill">Run: {report.run_id}</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginTop: 4 }}>
                    <p className="evo-report-query" style={{ margin: 0 }}>
                      {report.is_workload ? `Workload (${report.workload_queries?.length || 1} queries)` : `Query: "${report.query}"`}
                    </p>
                    <span
                      className="evo-status-pill"
                      style={{
                        background: 'rgba(59, 130, 246, 0.15)',
                        borderColor: '#3b82f6',
                        color: '#93c5fd',
                        fontSize: '0.8rem',
                        padding: '2px 8px',
                      }}
                    >
                      🎯 {report.scope_claim || (report.is_workload ? 'Optimized for the selected evaluation workload.' : 'Optimized for the selected evaluation query.')}
                    </span>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setIsReportCollapsed(!isReportCollapsed)}
                  className="evo-btn-secondary evo-btn-sm"
                >
                  {isReportCollapsed ? 'Expand Details' : 'Collapse Details'}
                </button>
              </div>

              {!isReportCollapsed && (
                <>
                  {/* Adaptive Scheduler Strategy Decision Box (if adaptive mode was used) */}
                  {report.scheduler_decision && (
                    <div
                      className="evo-summary-box"
                      style={{
                        marginBottom: 16,
                        background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%)',
                        borderLeft: '4px solid #8b5cf6',
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <strong style={{ color: '#c4b5fd', fontSize: '0.95rem' }}>Adaptive Strategy Selection</strong>
                        </div>
                        <span className="evo-version-tag" style={{ background: '#6366f1' }}>
                          Selected Stage: {(report.scheduler_decision.stage_name || report.scheduler_decision.strategy || report.scheduler_decision.stage || 'UNAVAILABLE').toUpperCase()}
                        </span>
                      </div>
                      <div style={{ marginTop: 8, fontSize: '0.88rem', color: '#e2e8f0', lineHeight: 1.5 }}>
                        <strong>Reason:</strong> {report.scheduler_decision.reason}
                      </div>
                      {report.scheduler_decision.rationale && (
                        <div style={{ marginTop: 4, fontSize: '0.82rem', color: '#94a3b8' }}>
                          <strong>Scheduler State:</strong> {report.scheduler_decision.rationale}
                        </div>
                      )}
                    </div>
                  )}

                  {/* Executive Summary Box with Clean Multi-line Formatting */}
                  <div className="evo-summary-box" style={{ whiteSpace: 'pre-line' }}>
                    <strong>Executive Summary:</strong>
                    <p style={{ margin: '8px 0 0 0', lineHeight: 1.6 }}>{report.executive_summary}</p>
                  </div>

                  {/* Summary Metric Cards with Exact Terminology & Grammar */}
                  {(() => {
                    const totalWorkflows = report.total_candidates_evaluated || report.all_candidates.length
                    const baselineCount = report.baseline ? 1 : 0
                    const generatedCount = Math.max(0, totalWorkflows - baselineCount)
                    const paretoCount = report.pareto_frontier_count ?? report.pareto_frontier.length
                    const paretoWfStr = paretoCount === 1 ? 'workflow forms' : 'workflows form'
                    const queryCount = report.workload_queries?.length || 1
                    const totalEvals = report.total_workflow_evaluations || (totalWorkflows * queryCount)

                    return (
                      <div className="evo-summary-grid">
                        <div className="evo-metric-card">
                          <span className="evo-metric-label">
                            {report.is_workload ? 'Workload Evaluations' : 'Total Workflows Evaluated'}
                          </span>
                          <span className="evo-metric-value">{report.is_workload ? totalEvals : totalWorkflows}</span>
                          <div className="evo-cell-sub" style={{ marginTop: 4, fontWeight: 500 }}>
                            {report.is_workload ? (
                              <>{totalWorkflows} Workflows × {queryCount} Queries</>
                            ) : (
                              <>{baselineCount} Baseline + {generatedCount} Generated Candidate{generatedCount === 1 ? '' : 's'}</>
                            )}
                          </div>
                          <span className="evo-metric-delta" style={{ marginTop: 2 }}>
                            {report.successful_candidates} successful, {report.failed_candidates} failed
                          </span>
                        </div>
                        <div className="evo-metric-card">
                          <span className="evo-metric-label">Pareto-Optimal Count</span>
                          <span className="evo-metric-value evo-val-green">{paretoCount}</span>
                          <span className="evo-metric-delta">{paretoCount} {paretoWfStr} the Pareto frontier</span>
                        </div>
                        <div className="evo-metric-card">
                          <span className="evo-metric-label">
                            {report.is_workload ? 'Avg Baseline Groundedness' : 'Baseline Groundedness'}
                          </span>
                          <span className="evo-metric-value">
                            {baselineCandidate ? `${((baselineCandidate.metrics.avg_groundedness ?? baselineCandidate.metrics.groundedness ?? baselineCandidate.metrics.performance) * 100).toFixed(1)}%` : '100.0%'}
                          </span>
                          <span className="evo-metric-delta">Reference Standard</span>
                        </div>
                        <div className="evo-metric-card">
                          <span className="evo-metric-label">
                            {report.is_workload ? 'Avg Baseline Cost / Query' : 'Baseline Cost / Query'}
                          </span>
                          <span className="evo-metric-value">
                            ${baselineCandidate ? (baselineCandidate.metrics.avg_cost_per_query ?? baselineCandidate.metrics.cost).toFixed(6) : '0.000057'}
                          </span>
                          <span className="evo-metric-delta">{baselineCandidate?.workflow?.model?.model_id || 'gemini-3.5-flash-lite'}</span>
                        </div>
                      </div>
                    )
                  })()}

                  {/* Explicit Pareto Objective Explanation Note */}
                  <div className="evo-pareto-objective-note" style={{
                    margin: '18px 0 12px 0',
                    padding: '12px 16px',
                    background: 'rgba(59, 130, 246, 0.08)',
                    borderLeft: '4px solid #3b82f6',
                    borderRadius: '6px',
                    fontSize: '0.875rem',
                    color: '#93c5fd',
                    lineHeight: 1.5
                  }}>
                    ℹ️ <strong>Pareto Objective Note:</strong> Pareto optimization currently balances Groundedness against Cost / Query. Latency and token usage are reported as secondary evaluation metrics.
                  </div>

                  {/* Candidates Comparison Table */}
                  <div className="evo-table-container" style={{ marginTop: 16 }}>
                    <h3 className="evo-section-title" style={{ fontSize: '1.05rem', marginBottom: 10 }}>
                      Evaluated Workflow Candidates (Pareto Frontier & Dominated)
                    </h3>
                    <table className="evo-table">
                      <thead>
                        <tr>
                          <th>Candidate / Strategy</th>
                          <th>Workflow Name</th>
                          <th>Model & Provider</th>
                          <th>Groundedness</th>
                          <th>Cost / Query</th>
                          <th>Latency</th>
                          <th>Pareto Status</th>
                          <th>Governance Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.all_candidates.map((cand) => {
                          const isBase = cand.candidate_id === 'baseline' || cand.workflow?.workflow_id === 'baseline'
                          const isPareto = cand.is_pareto_optimal
                          const isSelected = cand.candidate_id === selectedCandidateId
                          const groundedness = cand.metrics.groundedness ?? cand.metrics.performance
                          const isApproved = report.approved_workflow_id === cand.candidate_id

                          return (
                            <tr
                              key={cand.candidate_id}
                              className={`evo-table-row ${isSelected ? 'selected' : ''}`}
                              onClick={() => setSelectedCandidateId(cand.candidate_id)}
                            >
                              <td>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                  <code>{cand.candidate_id}</code>
                                  {renderMutationBadge(cand)}
                                </div>
                              </td>
                              <td>
                                <strong>{cand.workflow.name}</strong>
                              </td>
                              <td>
                                <div className="evo-model-cell">
                                  <span className="evo-provider-name">{cand.workflow.model.provider || 'Google'}</span>
                                  <span className="evo-model-id">{cand.workflow.model.model_id}</span>
                                </div>
                              </td>
                              <td>
                                <strong className="evo-val-green">{(groundedness * 100).toFixed(1)}%</strong>
                              </td>
                              <td>
                                <span className="evo-cost-val">${cand.metrics.cost.toFixed(6)}</span>
                              </td>
                              <td>
                                <span>{cand.metrics.llm_latency_ms.toFixed(0)} ms</span>
                              </td>
                              <td>
                                {isBase ? (
                                  <span className="evo-status-tag tag-baseline">Baseline</span>
                                ) : isPareto ? (
                                  <span className="evo-pareto-optimal-badge">Pareto</span>
                                ) : (
                                  <span className="evo-dominated-badge">Dominated</span>
                                )}
                              </td>
                              <td>
                                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      setSelectedCandidateId(cand.candidate_id)
                                      setActiveMainTab('review')
                                    }}
                                    className="evo-btn-inspect"
                                    style={{ background: 'rgba(99, 102, 241, 0.15)', borderColor: 'rgba(99, 102, 241, 0.4)', color: '#c7d2fe' }}
                                    title="Deep-dive into configuration diffs and claim evidence in Candidate Review"
                                  >
                                    Review ➔
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation()
                                      setInspectModalCandidate(cand)
                                    }}
                                    className="evo-btn-inspect"
                                  >
                                    Inspect
                                  </button>
                                  {isBase ? (
                                    <span className="evo-cell-sub" style={{ fontWeight: 600, color: '#0284c7', padding: '2px 6px' }}>
                                      Current Workflow
                                    </span>
                                  ) : isPareto ? (
                                    isApproved ? (
                                      <span className="evo-gov-state-pill gov-approved">APPROVED</span>
                                    ) : (
                                      <button
                                        type="button"
                                        onClick={(e) => {
                                          e.stopPropagation()
                                          setConfirmApprovalCandidate(cand)
                                        }}
                                        className="evo-btn-approve"
                                        disabled={approvingId === cand.candidate_id}
                                      >
                                        Approve
                                      </button>
                                    )
                                  ) : null}
                                </div>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Pareto Trade-Off Analyses */}
                  {report.trade_off_analyses && report.trade_off_analyses.length > 0 && (
                    <div className="evo-tradeoff-grid" style={{ marginTop: 20 }}>
                      {report.trade_off_analyses.map((analysis) => {
                        const cand = report.all_candidates.find((c) => c.candidate_id === analysis.candidate_id)
                        const candGroundedness = cand ? ((cand.metrics.groundedness ?? cand.metrics.performance) * 100).toFixed(1) : '100.0'

                        return (
                          <div key={analysis.candidate_id} className="evo-tradeoff-card">
                            <div className="evo-tradeoff-card-head">
                              <span className="evo-tradeoff-tag">{analysis.candidate_id}</span>
                              <span className="evo-tradeoff-title">{analysis.name}</span>
                            </div>
                            <p className="evo-tradeoff-summary">{analysis.trade_off_summary}</p>
                            <div className="evo-tradeoff-metrics">
                              <div className="evo-delta-item">
                                <span>Groundedness:</span>
                                <strong className="evo-val-green">
                                  {candGroundedness}%
                                </strong>
                              </div>
                              <div className="evo-delta-item">
                                <span>Groundedness Delta:</span>
                                <strong className={analysis.performance_delta_pct >= 0 ? 'evo-val-green' : 'evo-val-amber'}>
                                  {analysis.performance_delta_pct > 0 ? '+' : ''}{analysis.performance_delta_pct.toFixed(1)} percentage points
                                </strong>
                              </div>
                              <div className="evo-delta-item">
                                <span>Cost Delta:</span>
                                <strong className={analysis.cost_delta_pct <= 0 ? 'evo-val-green' : 'evo-val-red'}>
                                  {analysis.cost_delta_pct > 0 ? '+' : ''}{analysis.cost_delta_pct.toFixed(1)}%
                                </strong>
                              </div>
                              <div className="evo-delta-item">
                                <span>Latency Delta:</span>
                                <strong className={analysis.latency_delta_pct <= 0 ? 'evo-val-green' : 'evo-val-amber'}>
                                  {analysis.latency_delta_pct > 0 ? '+' : ''}{analysis.latency_delta_pct.toFixed(1)}%
                                </strong>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </>
      )}

      {/* ========================================================================= */}
      {/* SECTION C: CANDIDATE REVIEW & VERIFICATION EVIDENCE                       */}
      {/* ========================================================================= */}
      {activeMainTab === 'review' && (
        <>
          {!selectedCandidate ? (
            <section className="evo-section">
              <div className="evo-empty-state" style={{ textAlign: 'center', padding: '48px 24px' }}>
                <span style={{ fontSize: '2.5rem', marginBottom: 12, display: 'inline-block' }}>🔍</span>
                <h3 style={{ margin: '0 0 8px 0', color: '#f1f5f9', fontSize: '1.2rem' }}>No Candidate Selected for Review</h3>
                <p style={{ margin: '0 auto 20px auto', maxWidth: 500, color: '#94a3b8', fontSize: '0.9rem', lineHeight: 1.5 }}>
                  Generate candidate workflows in the Evolution Lab or select one from the Pareto frontier table to inspect side-by-side configuration diffs and claim-level verification evidence.
                </p>
                <button
                  type="button"
                  onClick={() => setActiveMainTab('evolution')}
                  className="evo-btn-primary"
                >
                  Go to Evolution Lab ➔
                </button>
              </div>
            </section>
          ) : (
            <>
              {/* Selected Candidate Header & Action Bar */}
              <section className="evo-section" style={{ borderLeft: '4px solid #6366f1' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                      <span className={`evo-status-tag ${selectedCandidate.candidate_id === 'baseline' ? 'tag-baseline' : selectedCandidate.is_pareto_optimal ? 'tag-pareto' : 'tag-dominated'}`}>
                        {selectedCandidate.candidate_id === 'baseline' ? 'Baseline' : selectedCandidate.is_pareto_optimal ? 'Pareto-Optimal' : 'Dominated'}
                      </span>
                      {renderMutationBadge(selectedCandidate)}
                      <span className="evo-version-tag" style={{ background: '#4f46e5' }}>
                        Candidate: {selectedCandidate.candidate_id}
                      </span>
                    </div>
                    <h2 className="evo-section-title" style={{ margin: '2px 0 6px 0', color: '#f8fafc', fontSize: '1.35rem' }}>
                      {selectedCandidate.workflow.name}
                    </h2>
                    <div className="evo-cell-sub" style={{ color: '#94a3b8' }}>
                      Parent: <code>{(selectedCandidate.mutation_metadata || selectedCandidate.workflow.mutation_metadata)?.parent_workflow_id || 'baseline'}</code> •
                      Model: <code>{selectedCandidate.workflow.model.model_id}</code> ({selectedCandidate.workflow.model.provider || 'Google'}) •
                      {(selectedCandidate.mutation_metadata || selectedCandidate.workflow.mutation_metadata)?.mutation_description || 'Evaluated workflow candidate'}
                    </div>
                  </div>

                  {/* Candidate Switcher & Governance Actions */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                    {report?.all_candidates && report.all_candidates.length > 1 && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <label className="evo-cell-sub" style={{ fontWeight: 600 }}>Switch Candidate:</label>
                        <select
                          value={selectedCandidate.candidate_id}
                          onChange={(e) => setSelectedCandidateId(e.target.value)}
                          className="evo-select evo-select-sm"
                          style={{ minWidth: 160 }}
                        >
                          {report.all_candidates.map((c) => (
                            <option key={c.candidate_id} value={c.candidate_id}>
                              {c.candidate_id} - {c.workflow.name.slice(0, 24)} {c.is_pareto_optimal ? '(Pareto)' : ''}
                            </option>
                          ))}
                        </select>
                      </div>
                    )}

                    {selectedCandidate.candidate_id !== 'baseline' && (
                      report?.approved_workflow_id === selectedCandidate.candidate_id ? (
                        <span className="evo-gov-state-pill gov-approved" style={{ fontSize: '0.85rem', padding: '6px 14px' }}>
                          ✓ APPROVED (Immutable Version Stored)
                        </span>
                      ) : selectedCandidate.is_pareto_optimal ? (
                        <button
                          type="button"
                          onClick={() => setConfirmApprovalCandidate(selectedCandidate)}
                          className="evo-btn-approve"
                          style={{ padding: '8px 16px', fontSize: '0.88rem' }}
                          disabled={approvingId === selectedCandidate.candidate_id}
                        >
                          {approvingId === selectedCandidate.candidate_id ? 'Approving...' : '✓ Approve Candidate Workflow'}
                        </button>
                      ) : (
                        <span className="evo-cell-sub" style={{ color: '#94a3b8' }}>Dominated (Not Recommended)</span>
                      )
                    )}
                  </div>
                </div>

                {/* Governance Policy Notice */}
                <div style={{ marginTop: 12, padding: '8px 12px', background: 'rgba(99, 102, 241, 0.08)', borderRadius: 6, border: '1px solid rgba(99, 102, 241, 0.2)', fontSize: '0.8rem', color: '#c7d2fe' }}>
                  ℹ️ <strong>Governance Gate Policy:</strong> Approving this candidate creates an immutable record with status <code>APPROVED</code> in the version registry. It does <strong>NOT</strong> silently deploy or alter live production traffic. Production activation requires an explicit separate lifecycle transition.
                </div>
              </section>

              {/* Candidate Metrics Cards */}
              <div className="evo-summary-grid">
                <div className="evo-metric-card">
                  <span className="evo-metric-label">Groundedness</span>
                  <span className="evo-metric-value evo-val-green">
                    {((selectedCandidate.metrics.avg_groundedness ?? selectedCandidate.metrics.groundedness ?? selectedCandidate.metrics.performance) * 100).toFixed(1)}%
                  </span>
                  <span className="evo-metric-delta">
                    Phase 6 Claim Verification
                  </span>
                </div>
                <div className="evo-metric-card">
                  <span className="evo-metric-label">Hallucination Rate</span>
                  <span className="evo-metric-value evo-val-cyan">
                    {((selectedCandidate.metrics.avg_hallucination ?? selectedCandidate.metrics.hallucination_rate ?? 0) * 100).toFixed(1)}%
                  </span>
                  <span className="evo-metric-delta">
                    Unsupported claims ratio
                  </span>
                </div>
                <div className="evo-metric-card">
                  <span className="evo-metric-label">Cost / Query</span>
                  <span className="evo-metric-value">
                    ${(selectedCandidate.metrics.avg_cost_per_query ?? selectedCandidate.metrics.cost).toFixed(6)}
                  </span>
                  <span className="evo-metric-delta">
                    Tokens: In {selectedCandidate.metrics.input_tokens || '—'} / Out {selectedCandidate.metrics.output_tokens || '—'}
                  </span>
                </div>
                <div className="evo-metric-card">
                  <span className="evo-metric-label">LLM Latency</span>
                  <span className="evo-metric-value evo-val-purple">
                    {(selectedCandidate.metrics.avg_latency_ms ?? selectedCandidate.metrics.llm_latency_ms ?? 0).toFixed(0)} ms
                  </span>
                  <span className="evo-metric-delta">
                    Total Exec Time: {selectedCandidate.metrics.execution_time_ms.toFixed(0)} ms
                  </span>
                </div>
              </div>

              {/* Baseline vs Candidate Configuration Diff Table */}
              <section className="evo-section">
                <div className="evo-section-header">
                  <div>
                    <h3 className="evo-section-title">Baseline vs Candidate Configuration Diff</h3>
                    <span className="evo-section-caption">
                      Side-by-side comparison between production reference baseline and candidate <code>{selectedCandidate.candidate_id}</code>.
                    </span>
                  </div>
                </div>

                <div className="evo-table-container">
                  <table className="evo-table">
                    <thead>
                      <tr>
                        <th>Module</th>
                        <th>Parameter</th>
                        <th>Baseline Value</th>
                        <th>Candidate Value</th>
                        <th>Diff Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { module: 'Model', param: 'Model ID', base: baselineCandidate?.workflow.model.model_id || 'gemini-3.5-flash-lite', cand: selectedCandidate.workflow.model.model_id },
                        { module: 'Model', param: 'Provider', base: baselineCandidate?.workflow.model.provider || 'Google', cand: selectedCandidate.workflow.model.provider },
                        { module: 'Model', param: 'Temperature', base: baselineCandidate?.workflow.model.temperature ?? 0.2, cand: selectedCandidate.workflow.model.temperature },
                        { module: 'Model', param: 'Max Tokens', base: baselineCandidate?.workflow.model.max_tokens ?? 1024, cand: selectedCandidate.workflow.model.max_tokens },
                        { module: 'Retrieval', param: 'Top-K Documents', base: baselineCandidate?.workflow.retrieval.top_k ?? 5, cand: selectedCandidate.workflow.retrieval.top_k },
                        { module: 'Retrieval', param: 'Similarity Threshold', base: baselineCandidate?.workflow.retrieval.similarity_threshold ?? 0.35, cand: selectedCandidate.workflow.retrieval.similarity_threshold },
                        { module: 'Retrieval', param: 'Chunk Size', base: baselineCandidate?.workflow.retrieval.chunk_size ?? 500, cand: selectedCandidate.workflow.retrieval.chunk_size },
                        { module: 'Retrieval', param: 'Chunk Overlap', base: baselineCandidate?.workflow.retrieval.chunk_overlap ?? 50, cand: selectedCandidate.workflow.retrieval.chunk_overlap },
                        { module: 'Retrieval', param: 'Reranking Enabled', base: baselineCandidate?.workflow.retrieval.enable_rerank ? 'Yes' : 'No', cand: selectedCandidate.workflow.retrieval.enable_rerank ? 'Yes' : 'No' },
                        { module: 'Prompt', param: 'Template Name', base: baselineCandidate?.workflow.prompt.template_name || 'clinical_summary', cand: selectedCandidate.workflow.prompt.template_name },
                        { module: 'Prompt', param: 'Disclaimer Included', base: baselineCandidate?.workflow.prompt.include_disclaimer ? 'Yes' : 'No', cand: selectedCandidate.workflow.prompt.include_disclaimer ? 'Yes' : 'No' },
                        { module: 'Execution', param: 'Context Compression', base: baselineCandidate?.workflow.execution.context_compression ? 'Enabled' : 'Disabled', cand: selectedCandidate.workflow.execution.context_compression ? 'Enabled' : 'Disabled' },
                        { module: 'Execution', param: 'Timeout Seconds', base: baselineCandidate?.workflow.execution.timeout_seconds ?? 30, cand: selectedCandidate.workflow.execution.timeout_seconds },
                        { module: 'Execution', param: 'Max Retries', base: baselineCandidate?.workflow.execution.max_retries ?? 2, cand: selectedCandidate.workflow.execution.max_retries },
                      ].map((row, rIdx) => {
                        const isModified = String(row.base) !== String(row.cand)
                        return (
                          <tr key={rIdx} className={`evo-table-row ${isModified ? 'evo-row-modified' : ''}`}>
                            <td><span className="evo-cell-sub">{row.module}</span></td>
                            <td><strong>{row.param}</strong></td>
                            <td><code>{String(row.base)}</code></td>
                            <td><code style={isModified ? { color: '#a5b4fc', fontWeight: 700 } : undefined}>{String(row.cand)}</code></td>
                            <td>
                              {isModified ? (
                                <span className="evo-diff-badge diff-changed">MODIFIED</span>
                              ) : (
                                <span className="evo-diff-badge diff-same">UNCHANGED</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Claim-Level Groundedness Verification Evidence */}
              <section className="evo-section">
                <div className="evo-section-header">
                  <div>
                    <h3 className="evo-section-title">Claim-Level Groundedness Verification Evidence</h3>
                    <span className="evo-section-caption">
                      Evidence sentences verified against retrieved clinical source passages using Phase 6 NLI metric.
                    </span>
                  </div>
                </div>

                {/* Workload Per-Query Evaluation Evidence (if multi-query workload) */}
                {selectedCandidate.metrics.raw_details?.per_query_evaluations &&
                Array.isArray(selectedCandidate.metrics.raw_details.per_query_evaluations) &&
                selectedCandidate.metrics.raw_details.per_query_evaluations.length > 0 ? (
                  <div className="evo-table-container">
                    <table className="evo-table" style={{ fontSize: '0.84rem' }}>
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Evaluation Query</th>
                          <th>Groundedness</th>
                          <th>Hallucination</th>
                          <th>Cost</th>
                          <th>Tokens (In/Out)</th>
                          <th>Latency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedCandidate.metrics.raw_details.per_query_evaluations.map((item: any, idx: number) => (
                          <tr key={idx} className="evo-table-row">
                            <td><strong>{idx + 1}</strong></td>
                            <td>
                              <div style={{ maxWidth: 300 }}>
                                <strong>{item.query}</strong>
                                {item.response && (
                                  <details style={{ marginTop: 4 }}>
                                    <summary style={{ cursor: 'pointer', fontSize: '0.75rem', color: '#818cf8' }}>
                                      View Generated Response
                                    </summary>
                                    <p style={{ margin: '4px 0', fontSize: '0.78rem', color: '#cbd5e1', lineHeight: 1.4 }}>
                                      {item.response}
                                    </p>
                                  </details>
                                )}
                              </div>
                            </td>
                            <td>
                              <strong className="evo-val-green">
                                {item.groundedness !== undefined && item.groundedness !== null
                                  ? `${(item.groundedness * 100).toFixed(1)}%`
                                  : 'N/A'}
                              </strong>
                            </td>
                            <td>
                              <span className="evo-val-cyan">
                                {item.hallucination_rate !== undefined && item.hallucination_rate !== null
                                  ? `${(item.hallucination_rate * 100).toFixed(1)}%`
                                  : '0.0%'}
                              </span>
                            </td>
                            <td>
                              <span className="evo-cost-val">${Number(item.cost || 0).toFixed(6)}</span>
                            </td>
                            <td>
                              <span>{item.input_tokens ?? item.tokens ?? 0} / {item.output_tokens ?? 0}</span>
                            </td>
                            <td>
                              <span>{Number(item.llm_latency_ms ?? item.latency_ms ?? 0).toFixed(0)} ms</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div className="evo-summary-box">
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12 }}>
                      <div>
                        <span className="evo-metric-label">Evaluation Target Query</span>
                        <p style={{ margin: '4px 0 0 0', fontSize: '0.88rem', color: '#f1f5f9' }}>
                          "{report?.query || query}"
                        </p>
                      </div>
                      <div>
                        <span className="evo-metric-label">Verification Outcome</span>
                        <p style={{ margin: '4px 0 0 0', fontSize: '0.88rem', color: '#4ade80' }}>
                          Verified Grounded: {((selectedCandidate.metrics.groundedness ?? selectedCandidate.metrics.performance) * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <span className="evo-metric-label">Token Footprint</span>
                        <p style={{ margin: '4px 0 0 0', fontSize: '0.88rem', color: '#94a3b8' }}>
                          {selectedCandidate.metrics.input_tokens} Input / {selectedCandidate.metrics.output_tokens} Output ({selectedCandidate.metrics.total_tokens} Total)
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            </>
          )}
        </>
      )}

      {/* ========================================================================= */}
      {/* SECTION D: PRODUCTION & VERSION HISTORY                                    */}
      {/* ========================================================================= */}
      {(activeMainTab === 'production' || (activeMainTab as any) === 'lifecycle') && (
        <>
          {/* Active Production Workflow Banner Card */}
          <section className="evo-section evo-active-banner">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16 }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                  <span className="evo-status-tag tag-pareto" style={{ fontSize: '0.85rem', padding: '4px 10px' }}>
                    ● CURRENTLY ACTIVE IN PRODUCTION
                  </span>
                  <span className="evo-version-tag" style={{ background: '#10b981' }}>
                    {activeWorkflow?.version_id === 'baseline' ? 'Production Baseline' : `v${activeWorkflow?.version_number}`}
                  </span>
                </div>
                <h2 className="evo-section-title" style={{ margin: '4px 0 6px 0', color: '#f8fafc' }}>
                  {activeWorkflow?.name || 'Production Baseline'}
                </h2>
                <div className="evo-cell-sub" style={{ color: '#94a3b8' }}>
                  Model: <code>{activeWorkflow?.model_id || 'gemini-3.5-flash-lite'}</code> ({activeWorkflow?.provider || 'Google Gemini'}) |
                  Top-K: <code>{activeWorkflow?.configuration?.retrieval?.top_k || 5}</code> |
                  Sim: <code>{activeWorkflow?.configuration?.retrieval?.similarity_threshold || 0.35}</code> |
                  Activated: {activeWorkflow?.activated_at ? new Date(activeWorkflow.activated_at).toLocaleString() : 'System Genesis'}
                </div>
              </div>
              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  type="button"
                  onClick={() => handleTriggerReoptimize(activeWorkflow?.version_id)}
                  className="evo-btn-primary"
                >
                  <span>🔄</span> Trigger Explicit Re-optimization
                </button>
              </div>
            </div>
          </section>

          {/* Approved Workflow Versions Table */}
          <section className="evo-section">
            <div className="evo-section-header">
              <div>
                <h2 className="evo-section-title">Approved Workflow Versions & Governance</h2>
                <span className="evo-section-caption">
                  Immutable stored workflow configurations resulting from explicit developer approval. Only one version is ACTIVE at a time.
                </span>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  type="button"
                  onClick={selectAllVersionsForBenchmark}
                  className="evo-btn-secondary evo-btn-sm"
                >
                  Benchmark All Versions
                </button>
              </div>
            </div>

            <div className="evo-table-container">
              <table className="evo-table">
                <thead>
                  <tr>
                    <th>Version</th>
                    <th>Workflow Name</th>
                    <th>Lineage / Strategy</th>
                    <th>Model & Provider</th>
                    <th>Snapshot Groundedness</th>
                    <th>Snapshot Cost</th>
                    <th>Approved By</th>
                    <th>Governance Status</th>
                    <th>Lifecycle Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {/* Baseline row */}
                  <tr className="evo-table-row">
                    <td>
                      <span className="evo-version-tag" style={{ background: '#0284c7' }}>Base</span>
                      <div className="evo-cell-sub"><code>baseline</code></div>
                    </td>
                    <td>
                      <strong>Production Baseline</strong>
                      <div className="evo-cell-sub">Reference standard workflow</div>
                    </td>
                    <td>
                      <span className="evo-cell-sub">Immutable Baseline</span>
                    </td>
                    <td>
                      <div className="evo-model-cell">
                        <span className="evo-provider-name">Google</span>
                        <span className="evo-model-id">gemini-3.5-flash-lite</span>
                      </div>
                    </td>
                    <td><span className="evo-perf-val">100.0%</span></td>
                    <td><span className="evo-cost-val">$0.000057</span></td>
                    <td><span className="evo-approver-user">System Genesis</span></td>
                    <td>
                      <span className={`evo-gov-state-pill ${activeWorkflow?.version_id === 'baseline' ? 'gov-active' : 'gov-inactive'}`}>
                        {activeWorkflow?.version_id === 'baseline' ? 'ACTIVE' : 'INACTIVE'}
                      </span>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <button
                          type="button"
                          onClick={() => handleInspectVersion('baseline')}
                          className="evo-btn-inspect"
                        >
                          Inspect
                        </button>
                        {activeWorkflow?.version_id !== 'baseline' && (
                          <button
                            type="button"
                            onClick={() => setActionModalVersion({
                              version: {
                                id: 'baseline',
                                version_id: 'baseline',
                                version_number: 0,
                                name: 'Production Baseline',
                                source_run_id: 'genesis',
                                source_candidate_id: 'baseline',
                                model_id: 'gemini-3.5-flash-lite',
                                provider: 'Google',
                                configuration: {},
                                evaluation_snapshot: {},
                                status: 'APPROVED',
                                approved_by: 'System Genesis',
                                approved_at: new Date().toISOString(),
                                created_at: new Date().toISOString(),
                              },
                              type: 'rollback'
                            })}
                            className="evo-btn-rollback"
                          >
                            Rollback
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>

                  {/* Approved Versions List */}
                  {workflowVersions.map((ver) => {
                    const snap = ver.evaluation_snapshot || {}
                    const groundedness = snap.groundedness ?? snap.performance ?? null
                    const cost = snap.cost ?? null
                    const formattedDate = ver.approved_at ? new Date(ver.approved_at).toLocaleDateString() : 'N/A'
                    const isVerActive = activeWorkflow?.version_id === ver.version_id
                    const statusPillClass = isVerActive ? 'gov-active' : ver.status === 'ACTIVE' ? 'gov-active' : ver.status === 'APPROVED' ? 'gov-approved' : 'gov-inactive'

                    return (
                      <tr key={ver.id || ver.version_id} className="evo-table-row">
                        <td>
                          <span className="evo-version-tag">v{ver.version_number}</span>
                          <div className="evo-cell-sub"><code>{ver.version_id}</code></div>
                        </td>
                        <td>
                          <strong>{ver.name}</strong>
                          {ver.description && <div className="evo-cell-sub">{ver.description}</div>}
                        </td>
                        <td>
                          <div className="evo-cell-sub">Cand: <code>{ver.source_candidate_id}</code></div>
                          {ver.configuration?.mutation_metadata?.mutation_type === 'crossover' && (
                            <span className="evo-mut-tag evo-mut-crossover" style={{ fontSize: '0.65rem' }}>CROSSOVER</span>
                          )}
                        </td>
                        <td>
                          <div className="evo-model-cell">
                            <span className="evo-provider-name">{ver.provider}</span>
                            <span className="evo-model-id">{ver.model_id}</span>
                          </div>
                        </td>
                        <td>
                          <span className="evo-perf-val">
                            {groundedness !== null ? `${(groundedness * 100).toFixed(1)}%` : 'N/A'}
                          </span>
                        </td>
                        <td>
                          <span className="evo-cost-val">
                            {cost !== null ? `$${Number(cost).toFixed(6)}` : 'N/A'}
                          </span>
                        </td>
                        <td>
                          <div className="evo-approver-info">
                            <span className="evo-approver-user">{ver.approved_by || 'Developer'}</span>
                            <span className="evo-approved-date">{formattedDate}</span>
                          </div>
                        </td>
                        <td>
                          <span className={`evo-gov-state-pill ${statusPillClass}`}>
                            {isVerActive ? 'ACTIVE' : ver.status.toUpperCase()}
                          </span>
                        </td>
                        <td>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                            <button
                              type="button"
                              onClick={() => handleInspectVersion(ver.version_id)}
                              className="evo-btn-inspect"
                            >
                              Inspect
                            </button>
                            {!isVerActive && (
                              <button
                                type="button"
                                onClick={() => setActionModalVersion({
                                  version: ver,
                                  type: ver.status === 'INACTIVE' ? 'rollback' : 'activate'
                                })}
                                className={ver.status === 'INACTIVE' ? 'evo-btn-rollback' : 'evo-btn-activate'}
                              >
                                {ver.status === 'INACTIVE' ? 'Rollback' : 'Activate'}
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* Lineage Graph Toggle & Provenance Visualization */}
          <section className="evo-section">
            <div className="evo-section-header">
              <div>
                <h3 className="evo-section-title">Workflow Lineage & Provenance Graph</h3>
                <span className="evo-section-caption">
                  Directed acyclic provenance tree showing ancestry, mutations, targeted refinements, and crossover recombination.
                </span>
              </div>
              <button
                type="button"
                onClick={() => {
                  setShowLineageInProduction(!showLineageInProduction)
                  if (!lineageData) fetchLineage()
                }}
                className="evo-btn-secondary evo-btn-sm"
              >
                {showLineageInProduction ? 'Hide Lineage Graph ▴' : 'Show Lineage Graph ▾'}
              </button>
            </div>

            {showLineageInProduction && (
              lineageData ? (
                <div className="evo-lineage-container" style={{ marginTop: 12 }}>
                  <div className="evo-lineage-grid">
                    {lineageData.nodes
                      .filter((n) => n.version_id !== 'baseline')
                      .map((node) => {
                        const edges = lineageData.edges.filter((e) => e.target === node.version_id)
                        const isCrossover = node.candidate_type === 'CROSSOVER' || !!node.parent_b_version_id || edges.length > 1
                        const primaryEdge = edges[0]
                        const secondaryEdge = edges[1]

                        return (
                          <div
                            key={node.version_id}
                            className={`evo-lineage-card ${isCrossover ? 'is-crossover' : node.mutation_type === 'refinement' ? 'is-refined' : ''}`}
                          >
                            <div className="evo-lineage-card-head">
                              <span className="evo-lineage-tag" style={isCrossover ? { background: '#8b5cf6' } : undefined}>
                                {isCrossover ? '🔀 ' : ''}v{node.version_number}
                              </span>
                              {isCrossover ? (
                                <span className="evo-lineage-arrow-tag" style={{ background: '#f3e8ff', color: '#6b21a8' }}>
                                  🔀 Crossover: <code>{node.parent_a_version_id || primaryEdge?.source}</code> + <code>{node.parent_b_version_id || secondaryEdge?.source}</code>
                                </span>
                              ) : (
                                <span className="evo-lineage-arrow-tag">
                                  ↓ from <code>{primaryEdge?.source === 'baseline' ? 'baseline' : primaryEdge?.source}</code> ({primaryEdge?.label || 'mutation'})
                                </span>
                              )}
                            </div>
                            <div className="evo-lineage-name">{node.name}</div>
                            <div className="evo-lineage-meta">
                              <span>Version ID: <code>{node.version_id}</code></span>
                              <span>Candidate: <code>{node.source_candidate_id}</code></span>
                              <span>
                                Mutation: <strong>{isCrossover ? 'CROSSOVER' : (node.mutation_type?.toUpperCase() || 'MUTATION')}</strong>{' '}
                                {node.refinement_operation ? `(${node.refinement_operation})` : ''}
                              </span>
                              {(node.scheduler_strategy || (node.configuration as any)?.scheduler_decision?.selected_strategy || (node.configuration as any)?.scheduler_decision?.strategy) && (
                                <span>
                                  Scheduler Strategy: <strong>{(node.scheduler_strategy || (node.configuration as any)?.scheduler_decision?.selected_strategy || (node.configuration as any)?.scheduler_decision?.strategy)?.toUpperCase()}</strong>
                                </span>
                              )}
                              {isCrossover && node.crossover_map && (
                                <div style={{ marginTop: 4, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                                  {Object.entries(node.crossover_map).map(([comp, par]) => (
                                    <span
                                      key={comp}
                                      style={{
                                        fontSize: '0.68rem',
                                        padding: '1px 5px',
                                        borderRadius: 3,
                                        background: par === 'parent_a' ? '#eff6ff' : '#f0fdf4',
                                        color: par === 'parent_a' ? '#1d4ed8' : '#15803d',
                                        border: '1px solid currentColor'
                                      }}
                                    >
                                      {comp}: {par === 'parent_a' ? 'Parent A' : 'Parent B'}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                            <button
                              type="button"
                              onClick={() => handleInspectVersion(node.version_id)}
                              className="evo-btn-inspect"
                            >
                              Inspect v{node.version_number}
                            </button>
                          </div>
                        )
                      })}
                  </div>
                </div>
              ) : (
                <div className="evo-summary-box" style={{ marginTop: 12 }}>Loading lineage graph...</div>
              )
            )}
          </section>

          {/* Lifecycle Governance Audit Trail */}
          <section className="evo-section">
            <div className="evo-section-header">
              <div>
                <h2 className="evo-section-title">Lifecycle Governance Audit Trail</h2>
                <span className="evo-section-caption">
                  Immutable security audit log of all human approvals, production activations, rollbacks, and explicit re-optimization events.
                </span>
              </div>
            </div>

            {auditEvents.length === 0 ? (
              <div className="evo-summary-box">No audit events recorded yet.</div>
            ) : (
              <div className="evo-table-container">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Event ID</th>
                      <th>Action Type</th>
                      <th>Target Version</th>
                      <th>Previous Version</th>
                      <th>Triggered By</th>
                      <th>Timestamp</th>
                      <th>Audit Details</th>
                    </tr>
                  </thead>
                  <tbody>
                    {auditEvents.map((evt) => (
                      <tr key={evt.event_id} className="evo-table-row">
                        <td><code>{evt.event_id.slice(0, 12)}...</code></td>
                        <td>
                          <span className="evo-status-tag tag-pareto" style={{ fontSize: '0.75rem' }}>
                            {evt.event_type}
                          </span>
                        </td>
                        <td><strong>{evt.workflow_version_id || '—'}</strong></td>
                        <td><span className="evo-cell-sub">{evt.previous_version_id || evt.details?.previous_active_version_id || evt.details?.superseded_by || '—'}</span></td>
                        <td><span className="evo-approver-user">{evt.triggered_by || evt.user_email || 'System'}</span></td>
                        <td><span className="evo-cell-sub">{new Date(evt.created_at).toLocaleString()}</span></td>
                        <td>
                          <span className="evo-cell-sub">
                            {evt.details?.reason ? `Reason: ${evt.details.reason}` : evt.details?.candidate_id ? `Cand: ${evt.details.candidate_id}` : evt.details?.source_candidate_id ? `Approved: ${evt.details.source_candidate_id}` : 'Governed Event'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      {/* ========================================================================= */}
      {/* SECTION E: EXECUTION & EVIDENCE                                           */}
      {/* ========================================================================= */}
      {activeMainTab === 'executions' && (
        <>
          {/* Production Runtime Executions & Monitoring */}
          <section className="evo-section">
            <div className="evo-section-header">
              <div>
                <h2 className="evo-section-title">Production Runtime Executions & Monitoring</h2>
                <span className="evo-section-caption">
                  Real user telemetry logged against the currently ACTIVE production workflow version. LLM latency and Total Execution Time are tracked separately.
                </span>
              </div>
            </div>

            {executions.length === 0 ? (
              <div className="evo-summary-box">
                No production executions recorded yet. Execute queries in the RAG Query workspace to monitor live runtime metrics.
              </div>
            ) : (
              <div className="evo-table-container">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Execution ID</th>
                      <th>Version ID</th>
                      <th>Query</th>
                      <th>Model</th>
                      <th>Tokens (In / Out)</th>
                      <th>Cost</th>
                      <th>LLM Latency</th>
                      <th>Total Time</th>
                      <th>Timestamp</th>
                      <th>Trace & Retrieval</th>
                      <th>User Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {executions.map((exec) => {
                      const isExpanded = expandedExecId === exec.execution_id
                      const configuredTopK = exec.execution_trace?.effective_configuration?.retrieval?.top_k ?? exec.execution_trace?.stages?.retrieval?.top_k ?? 'N/A'
                      const actualChunksCount = exec.sources?.length ?? exec.execution_trace?.stages?.retrieval?.chunks_retrieved ?? (exec.retrieved_doc_ids ? exec.retrieved_doc_ids.length : 0)
                      return (
                        <React.Fragment key={exec.execution_id}>
                          <tr className="evo-table-row">
                            <td><code>{exec.execution_id.slice(0, 12)}...</code></td>
                            <td><span className="evo-version-tag">{exec.workflow_version_id}</span></td>
                            <td>
                              <div style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={exec.query}>
                                {exec.query}
                              </div>
                            </td>
                            <td><span className="evo-model-id">{exec.model_id || exec.model}</span></td>
                            <td>
                              <span className="evo-cell-sub">
                                {exec.total_tokens !== null && exec.total_tokens !== undefined ? exec.total_tokens : 'N/A'}
                                {' '}({exec.input_tokens ?? 'N/A'} / {exec.output_tokens ?? 'N/A'})
                              </span>
                            </td>
                            <td><span className="evo-cost-val">{exec.cost !== null && exec.cost !== undefined ? `$${exec.cost.toFixed(6)}` : 'N/A'}</span></td>
                            <td><strong className="evo-val-purple">{exec.llm_latency_ms !== null && exec.llm_latency_ms !== undefined ? `${exec.llm_latency_ms.toFixed(0)} ms` : 'N/A'}</strong></td>
                            <td><span>{exec.total_execution_time_ms.toFixed(0)} ms</span></td>
                            <td><span className="evo-cell-sub">{new Date(exec.created_at).toLocaleTimeString()}</span></td>
                            <td>
                              <button
                                type="button"
                                onClick={() => setExpandedExecId(isExpanded ? null : exec.execution_id)}
                                className="evo-btn-feedback"
                                style={{ background: isExpanded ? '#0369a1' : '#f1f5f9', color: isExpanded ? '#fff' : '#0f172a' }}
                              >
                                {isExpanded ? 'Hide Trace ▴' : 'Inspect Trace ▾'}
                              </button>
                            </td>
                            <td>
                              <button
                                type="button"
                                onClick={() => {
                                  setFeedbackModalExecution(exec)
                                  setFeedbackRating(5)
                                  setFeedbackText('')
                                }}
                                className="evo-btn-feedback"
                              >
                                Record Feedback
                              </button>
                            </td>
                          </tr>
                          {isExpanded && (
                            <tr className="evo-table-expanded-row">
                              <td colSpan={11} style={{ padding: '14px 18px', background: '#0e1422', borderBottom: '1px solid rgba(148, 163, 184, 0.15)' }}>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 14, marginBottom: 12 }}>
                                  {/* Retrieval & Top-K Verification */}
                                  <div style={{ padding: 12, background: '#131b2c', border: '1px solid rgba(148, 163, 184, 0.12)', borderRadius: 6, color: '#f1f5f9' }}>
                                    <div style={{ fontWeight: 600, color: '#818cf8', marginBottom: 6, fontSize: '0.85rem' }}>Retrieval & Chunk Verification</div>
                                    <div style={{ display: 'flex', gap: 12, marginBottom: 6, fontSize: '0.82rem' }}>
                                      <div>Configured Top-K: <strong style={{ color: '#818cf8' }}>{configuredTopK}</strong></div>
                                      <div>Actual Chunks Retrieved: <strong style={{ color: actualChunksCount <= (typeof configuredTopK === 'number' ? configuredTopK : 99) ? '#34d399' : '#f87171' }}>{actualChunksCount}</strong></div>
                                    </div>
                                    <div style={{ fontSize: '0.82rem' }}>Reranking: <strong>{exec.execution_trace?.stages?.retrieval?.reranking_enabled ? 'Enabled' : 'Bypassed'}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Similarity Threshold: <strong>{exec.execution_trace?.stages?.retrieval?.similarity_threshold ?? '0.35'}</strong></div>
                                    {exec.sources && exec.sources.length > 0 && (
                                      <div style={{ marginTop: 8 }}>
                                        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8' }}>Retrieved Chunks & Documents:</span>
                                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                                          {exec.sources.map((src, sIdx) => (
                                            <span key={sIdx} style={{ fontSize: '0.72rem', background: 'rgba(99, 102, 241, 0.12)', color: '#c7d2fe', border: '1px solid rgba(99, 102, 241, 0.25)', padding: '2px 6px', borderRadius: 4 }}>
                                              {src.chunk_id || `chunk-${src.chunk_index}`} (doc: {src.document_id ? src.document_id.slice(0, 8) : 'doc'}) {src.relevance_score ? `• ${(src.relevance_score * 100).toFixed(0)}%` : ''}
                                            </span>
                                          ))}
                                        </div>
                                      </div>
                                    )}
                                  </div>

                                  {/* Execution Stages */}
                                  <div style={{ padding: 12, background: '#131b2c', border: '1px solid rgba(148, 163, 184, 0.12)', borderRadius: 6, color: '#f1f5f9' }}>
                                    <div style={{ fontWeight: 600, color: '#818cf8', marginBottom: 6, fontSize: '0.85rem' }}>Stage Latencies & Configuration</div>
                                    <div style={{ fontSize: '0.82rem' }}>Retrieval Latency: <strong>{exec.execution_trace?.stages?.retrieval?.latency_ms ?? 0} ms</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Context Compression: <strong>{exec.execution_trace?.stages?.compression?.enabled ? 'Active' : 'Bypassed'}</strong> ({exec.execution_trace?.stages?.compression?.effective_context_length ?? 0} chars)</div>
                                    <div style={{ fontSize: '0.82rem' }}>Prompt Template: <strong>{exec.execution_trace?.stages?.prompt?.template || 'default'}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>LLM Latency: <strong>{exec.llm_latency_ms !== null && exec.llm_latency_ms !== undefined ? `${exec.llm_latency_ms.toFixed(1)} ms` : 'N/A'}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Total Pipeline Time: <strong>{exec.total_execution_time_ms.toFixed(1)} ms</strong></div>
                                  </div>

                                  {/* Groundedness & Claim Verification */}
                                  <div style={{ padding: 12, background: '#131b2c', border: '1px solid rgba(148, 163, 184, 0.12)', borderRadius: 6, color: '#f1f5f9' }}>
                                    <div style={{ fontWeight: 600, color: '#818cf8', marginBottom: 6, fontSize: '0.85rem' }}>Groundedness & Claim Verification</div>
                                    <div style={{ fontSize: '0.82rem' }}>Groundedness: <strong>{exec.groundedness !== null && exec.groundedness !== undefined ? `${(exec.groundedness * 100).toFixed(1)}%` : 'N/A'}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Evaluated Claims: <strong>{exec.execution_trace?.stages?.evidence_verification?.total_claims ?? 0}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Supported Claims: <strong style={{ color: '#34d399' }}>{exec.execution_trace?.stages?.evidence_verification?.supported_claims ?? 0}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Unsupported Claims: <strong style={{ color: (exec.execution_trace?.stages?.evidence_verification?.unsupported_claims ?? 0) > 0 ? '#f87171' : '#94a3b8' }}>{exec.execution_trace?.stages?.evidence_verification?.unsupported_claims ?? 0}</strong></div>
                                    {exec.execution_trace?.stages?.evidence_verification?.claim_breakdown && exec.execution_trace.stages.evidence_verification.claim_breakdown.length > 0 && (
                                      <div style={{ marginTop: 6, maxHeight: 90, overflowY: 'auto' }}>
                                        {exec.execution_trace.stages.evidence_verification.claim_breakdown.map((cb: any, cIdx: number) => (
                                          <div key={cIdx} style={{ fontSize: '0.72rem', color: cb.status === 'supported' ? '#16a34a' : '#dc2626' }}>
                                            {cb.status === 'supported' ? '✓' : '✗'} {cb.claim}
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </div>
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      {/* ========================================================================= */}
      {/* SECTION F: BENCHMARKING & MULTI-VERSION COMPARISON                        */}
      {/* ========================================================================= */}
      {(activeMainTab === 'benchmarking' || (activeMainTab as any) === 'benchmark') && (
        <>
          {/* Stale Benchmark Warning Notice */}
          {benchmarkReport && activeWorkflow && (
            (() => {
              const isStale = !benchmarkReport.selected_versions?.includes(activeWorkflow.version_id) ||
                (benchmarkReport.query && benchmarkReport.query !== benchQuery)
              if (!isStale) return null
              return (
                <div className="evo-summary-box" style={{ marginBottom: 16, background: 'rgba(245, 158, 11, 0.08)', borderLeft: '4px solid #f59e0b', color: '#fcd34d' }}>
                  <strong>⚠️ Stale Benchmark Notice:</strong> Current active production workflow is <code>{activeWorkflow.version_id}</code> ({activeWorkflow.name}). The benchmark results below reflect earlier configuration conditions. Run a new benchmark comparison to record verified metrics for active production traffic.
                </div>
              )
            })()
          )}

          <section className="evo-section evo-controls-card">
            <div className="evo-section-header">
              <div>
                <h2 className="evo-section-title">Controlled Version Benchmark Runner</h2>
                <span className="evo-section-caption">
                  Evaluate selected workflow versions (baseline & approved) under the EXACT SAME query, document, and Phase 6 metrics.
                </span>
              </div>
            </div>

            {/* Benchmark Setup Form */}
            <div className="evo-form">
              <div className="evo-form-group">
                <label className="evo-label">Evaluation Preset</label>
                <div className="evo-preset-buttons">
                  {(Object.keys(PRESET_QUERIES) as PresetType[]).map((presetKey) => (
                    <button
                      key={presetKey}
                      type="button"
                      onClick={() => handleBenchPresetChange(presetKey)}
                      className={`evo-preset-btn ${benchPreset === presetKey ? 'active' : ''}`}
                      disabled={isBenchmarking}
                    >
                      {PRESET_QUERIES[presetKey].label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="evo-form-group">
                <label className="evo-label">Identical Benchmark Query</label>
                <textarea
                  className="evo-textarea"
                  value={benchQuery}
                  onChange={(e) => {
                    setBenchQuery(e.target.value)
                    setBenchPreset('custom')
                  }}
                  rows={2}
                  disabled={isBenchmarking}
                />
              </div>

              <div className="evo-form-row">
                <div className="evo-form-group evo-col-half">
                  <label className="evo-label">Target Document Context</label>
                  <select
                    className="evo-select"
                    value={benchDocId}
                    onChange={(e) => setBenchDocId(e.target.value)}
                    disabled={isBenchmarking}
                  >
                    <option value="">All Uploaded Documents (Global)</option>
                    {documents.map((doc) => (
                      <option key={doc.id} value={doc.id}>
                        {doc.filename || doc.original_filename || doc.id}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="evo-form-group evo-col-half">
                  <label className="evo-label">Selected Versions for Benchmark ({selectedBenchmarkVersions.length})</label>
                  <div className="evo-bench-pills-wrap">
                    <span
                      className={`evo-ver-pill ${selectedBenchmarkVersions.includes('baseline') ? 'selected' : ''}`}
                      onClick={() => toggleBenchmarkVersion('baseline')}
                    >
                      {selectedBenchmarkVersions.includes('baseline') ? '✓ ' : ''}Baseline
                    </span>
                    {workflowVersions.map((v) => (
                      <span
                        key={v.version_id}
                        className={`evo-ver-pill ${selectedBenchmarkVersions.includes(v.version_id) ? 'selected' : ''}`}
                        onClick={() => toggleBenchmarkVersion(v.version_id)}
                      >
                        {selectedBenchmarkVersions.includes(v.version_id) ? '✓ ' : ''}v{v.version_number} ({v.name.slice(0, 18)})
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              <div className="evo-form-actions">
                <button
                  type="button"
                  onClick={handleRunBenchmark}
                  className="evo-btn-benchmark"
                  disabled={isBenchmarking || selectedBenchmarkVersions.length === 0}
                >
                  {isBenchmarking ? (
                    <>
                      <span className="evo-spinner" /> Running Multi-Version Benchmark...
                    </>
                  ) : (
                    'Run Reproducible Benchmark'
                  )}
                </button>
              </div>
            </div>
          </section>

          {/* Benchmark Results Section */}
          {benchmarkReport && (
            <section className="evo-section">
              <div className="evo-section-header">
                <div>
                  <div className="evo-header-with-badge">
                    <h2 className="evo-section-title">{benchmarkReport.name}</h2>
                    <span className="evo-count-badge">ID: {benchmarkReport.benchmark_id}</span>
                  </div>
                  <span className="evo-section-caption">
                    Evaluated by {benchmarkReport.created_by} on {new Date(benchmarkReport.created_at).toLocaleString()}
                  </span>
                </div>
              </div>

              {/* Stale Benchmark Warning if any workflow version was approved after this benchmark run */}
              {workflowVersions.some((v) => v.version_id !== 'baseline' && new Date(v.approved_at || v.created_at).getTime() > new Date(benchmarkReport.created_at).getTime()) && (
                <div style={{ marginBottom: 16, padding: '10px 16px', background: '#fef3c7', border: '1px solid #f59e0b', borderRadius: 6, color: '#92400e', fontSize: '0.86rem' }}>
                  <strong>⚠️ Stale Benchmark Notice:</strong> Newer approved workflow versions exist in the database that were created after this benchmark was executed ({new Date(benchmarkReport.created_at).toLocaleString()}). Consider re-running the benchmark across all current active versions.
                </div>
              )}

              <div className="evo-summary-box" style={{ marginBottom: 16 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '8px' }}>
                  <div><strong>Benchmark ID:</strong> <code>{benchmarkReport.benchmark_id}</code></div>
                  <div><strong>Timestamp:</strong> <span>{new Date(benchmarkReport.created_at).toLocaleString()}</span></div>
                  <div><strong>Evaluation Preset:</strong> <span>{benchmarkReport.preset_name || 'clinical'}</span></div>
                  <div><strong>Versions Evaluated:</strong> <span>{benchmarkReport.selected_versions?.length || benchmarkReport.results?.length || 0}</span></div>
                </div>
                <div style={{ marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '8px' }}>
                  <strong>Query:</strong> <em>"{benchmarkReport.query}"</em>
                </div>
              </div>

              {/* Table */}
              <div className="evo-table-container">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Workflow</th>
                      <th>Model</th>
                      <th>Groundedness</th>
                      <th>Hallucination</th>
                      <th>Cost / Query</th>
                      <th>LLM Latency</th>
                      <th>Tokens</th>
                      <th>Pareto Status</th>
                      <th>Evidence & Stages</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkReport.results.map((r) => {
                      const isBase = r.version_id === 'baseline' || r.version === 'baseline'
                      const isPareto = r.is_pareto_optimal
                      const isExpanded = expandedBenchmarkVersion === r.version_id
                      const mutType = (r.configuration as any)?.mutation_metadata?.mutation_type || (isBase ? 'BASELINE' : 'MUTATION')
                      const schedStrat = (r.configuration as any)?.scheduler_decision?.selected_strategy || (r.configuration as any)?.scheduler_decision?.strategy || 'ADAPTIVE'

                      return (
                        <React.Fragment key={r.version_id}>
                          <tr className="evo-table-row">
                            <td>
                              <span className="evo-version-tag" style={isBase ? { background: '#0284c7' } : undefined}>
                                {isBase ? 'Base' : `v${r.version_number ?? ''}`}
                              </span>
                              <div className="evo-cell-sub"><code>{r.version_id}</code></div>
                            </td>
                            <td><strong>{r.workflow_name || r.workflow}</strong></td>
                            <td><span className="evo-model-id">{r.model_id || r.model}</span></td>
                            <td><strong className="evo-val-green">{(r.groundedness * 100).toFixed(1)}%</strong></td>
                            <td><span className="evo-val-cyan">{((r.hallucination_rate ?? r.hallucination ?? 0) * 100).toFixed(1)}%</span></td>
                            <td><span className="evo-cost-val">${(r.cost_per_query ?? r.cost).toFixed(6)}</span></td>
                            <td><span className="evo-val-purple">{(r.llm_latency_ms ?? r.latency_ms ?? 0).toFixed(0)} ms</span></td>
                            <td><span>{r.total_tokens}</span></td>
                            <td>
                              <span className={isPareto ? 'evo-pareto-optimal-badge' : 'evo-dominated-badge'}>
                                {r.pareto_status}
                              </span>
                            </td>
                            <td>
                              <button
                                type="button"
                                className="evo-btn-inspect"
                                style={{ padding: '3px 8px', fontSize: '0.72rem' }}
                                onClick={() => setExpandedBenchmarkVersion(isExpanded ? null : r.version_id)}
                              >
                                {isExpanded ? 'Hide Trace' : '🔍 Evidence & Stages'}
                              </button>
                            </td>
                          </tr>
                          {isExpanded && (
                            <tr className="evo-table-detail-row">
                              <td colSpan={10} style={{ padding: '12px 16px', background: '#f8fafc', borderBottom: '2px solid #e2e8f0' }}>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 12, marginBottom: 12 }}>
                                  <div style={{ padding: 10, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 6 }}>
                                    <div style={{ fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>🧬 Mutation & Scheduler Provenance</div>
                                    <div style={{ fontSize: '0.82rem' }}>Candidate Mutation: <strong>{String(mutType).toUpperCase()}</strong></div>
                                    <div style={{ fontSize: '0.82rem' }}>Scheduler Strategy: <strong>{String(schedStrat).toUpperCase()}</strong></div>
                                    {r.parent_workflow_id && <div style={{ fontSize: '0.82rem' }}>Parent: <code>{r.parent_workflow_id}</code></div>}
                                  </div>
                                  <div style={{ padding: 10, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 6 }}>
                                    <div style={{ fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>⚙️ Executed Stages</div>
                                    <div style={{ fontSize: '0.82rem' }}>Retrieval: top_k={r.retrieval?.top_k ?? 5}, rerank={r.retrieval?.reranking ? 'Active' : 'Bypassed'}</div>
                                    <div style={{ fontSize: '0.82rem' }}>Compression: {r.context_compression ? 'Active' : 'Bypassed'}</div>
                                    <div style={{ fontSize: '0.82rem' }}>Prompt: {r.prompt_template || 'default'}</div>
                                  </div>
                                  <div style={{ padding: 10, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 6 }}>
                                    <div style={{ fontWeight: 700, color: '#1e293b', marginBottom: 4 }}>📊 Latency & Timing Breakdown</div>
                                    <div style={{ fontSize: '0.82rem' }}>LLM Latency: {(r.llm_latency_ms ?? 0).toFixed(0)} ms</div>
                                    <div style={{ fontSize: '0.82rem' }}>Total Pipeline: {(r.execution_time_ms ?? 0).toFixed(0)} ms</div>
                                    <div style={{ fontSize: '0.82rem' }}>Total Cost: ${(r.cost_per_query ?? r.cost).toFixed(6)}</div>
                                  </div>
                                </div>

                                {r.claim_breakdown && r.claim_breakdown.length > 0 && (
                                  <div style={{ marginTop: 8 }}>
                                    <div style={{ fontWeight: 700, fontSize: '0.84rem', marginBottom: 6 }}>📑 Factual Evidence Claims ({r.claim_breakdown.length}):</div>
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 200, overflowY: 'auto' }}>
                                      {r.claim_breakdown.map((claimItem: any, idx: number) => (
                                        <div
                                          key={idx}
                                          style={{
                                            padding: '6px 10px',
                                            borderRadius: 4,
                                            fontSize: '0.8rem',
                                            background: claimItem.status === 'supported' ? '#f0fdf4' : '#fef2f2',
                                            border: `1px solid ${claimItem.status === 'supported' ? '#86efac' : '#fca5a5'}`
                                          }}
                                        >
                                          <div><strong>Claim:</strong> {claimItem.claim}</div>
                                          {claimItem.matched_context_snippet && (
                                            <div style={{ color: '#475569', fontSize: '0.76rem', marginTop: 2 }}>
                                              <em>Evidence Quote:</em> "{claimItem.matched_context_snippet}"
                                            </div>
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </td>
                            </tr>
                          )}
                        </React.Fragment>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}



      {/* ========================================================================= */}
      {/* CONFIRM APPROVAL MODAL                                                    */}
      {/* ========================================================================= */}
      {confirmApprovalCandidate && (
        <div className="evo-modal-backdrop" onClick={() => setConfirmApprovalCandidate(null)} role="dialog" aria-modal="true">
          <div className="evo-modal-panel" onClick={(e) => e.stopPropagation()}>
            <div className="evo-modal-head">
              <h3>Confirm Workflow Approval</h3>
              <button type="button" onClick={() => setConfirmApprovalCandidate(null)} className="evo-modal-close">✕</button>
            </div>
            <div className="evo-modal-content">
              <p>Are you sure you want to approve this candidate workflow and persist it as an immutable version?</p>
              <div className="evo-summary-box" style={{ marginTop: 12 }}>
                <strong>Workflow:</strong> {confirmApprovalCandidate.workflow.name} (<code>{confirmApprovalCandidate.candidate_id}</code>)<br />
                <strong>Groundedness:</strong> {((confirmApprovalCandidate.metrics.groundedness ?? confirmApprovalCandidate.metrics.performance) * 100).toFixed(1)}%<br />
                <strong>Cost / Query:</strong> ${confirmApprovalCandidate.metrics.cost.toFixed(6)}<br />
                <strong>Model:</strong> {confirmApprovalCandidate.workflow.model.model_id}
              </div>
              <p style={{ marginTop: 10, fontSize: '0.85rem', color: '#94a3b8' }}>
                Note: Approval creates an immutable version with status <code>APPROVED</code>. It does NOT automatically activate or deploy the workflow.
              </p>
            </div>
            <div className="evo-modal-foot">
              <button type="button" onClick={() => setConfirmApprovalCandidate(null)} className="evo-btn-secondary">
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleApproveWorkflow(confirmApprovalCandidate.candidate_id)}
                className="evo-btn-primary"
                disabled={approvingId === confirmApprovalCandidate.candidate_id}
              >
                {approvingId === confirmApprovalCandidate.candidate_id ? 'Approving...' : 'Confirm Approval'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* ACTIVATE / ROLLBACK CONFIRMATION MODAL                                    */}
      {/* ========================================================================= */}
      {actionModalVersion && (
        <div className="evo-modal-backdrop" onClick={() => setActionModalVersion(null)} role="dialog" aria-modal="true">
          <div className="evo-modal-panel" onClick={(e) => e.stopPropagation()}>
            <div className="evo-modal-head">
              <h3>{actionModalVersion.type === 'activate' ? 'Activate Production Workflow' : 'Rollback to Workflow Version'}</h3>
              <button type="button" onClick={() => setActionModalVersion(null)} className="evo-modal-close">✕</button>
            </div>
            <div className="evo-modal-content">
              <p>
                {actionModalVersion.type === 'activate'
                  ? `Promote version ${actionModalVersion.version.version_id} to be the ACTIVE production workflow.`
                  : `Rollback production execution to immutable version ${actionModalVersion.version.version_id}.`}
              </p>
              <div className="evo-summary-box" style={{ marginTop: 12 }}>
                <strong>Target Version:</strong> {actionModalVersion.version.name} (<code>{actionModalVersion.version.version_id}</code>)<br />
                <strong>Model:</strong> {actionModalVersion.version.model_id}<br />
                <strong>Current Active:</strong> {activeWorkflow?.version_id || 'baseline'}
              </div>
              <div className="evo-form-group" style={{ marginTop: 14 }}>
                <label className="evo-label">Reason / Governance Note (Optional)</label>
                <input
                  type="text"
                  className="evo-input"
                  placeholder="e.g., Clinical trial validation complete / rolled back due to latency spike"
                  value={actionReason}
                  onChange={(e) => setActionReason(e.target.value)}
                  disabled={isActionSubmitting}
                />
              </div>
            </div>
            <div className="evo-modal-foot">
              <button type="button" onClick={() => setActionModalVersion(null)} className="evo-btn-secondary">
                Cancel
              </button>
              <button
                type="button"
                onClick={handleExecuteAction}
                className={actionModalVersion.type === 'activate' ? 'evo-btn-primary' : 'evo-btn-rollback'}
                disabled={isActionSubmitting}
              >
                {isActionSubmitting ? 'Processing...' : actionModalVersion.type === 'activate' ? 'Confirm Activation' : 'Confirm Rollback'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* RECORD USER FEEDBACK MODAL                                                */}
      {/* ========================================================================= */}
      {feedbackModalExecution && (
        <div className="evo-modal-backdrop" onClick={() => setFeedbackModalExecution(null)} role="dialog" aria-modal="true">
          <div className="evo-modal-panel" onClick={(e) => e.stopPropagation()}>
            <div className="evo-modal-head">
              <h3>Record Runtime Feedback</h3>
              <button type="button" onClick={() => setFeedbackModalExecution(null)} className="evo-modal-close">✕</button>
            </div>
            <div className="evo-modal-content">
              <div className="evo-summary-box">
                <div><strong>Execution ID:</strong> <code>{feedbackModalExecution.execution_id}</code></div>
                <div><strong>Workflow Version:</strong> <span>{feedbackModalExecution.workflow_version_id}</span></div>
                <div><strong>Query:</strong> <em>"{feedbackModalExecution.query}"</em></div>
              </div>

              <div className="evo-form-group" style={{ marginTop: 16 }}>
                <label className="evo-label">Clinical Rating (1 to 5 Stars)</label>
                <div style={{ display: 'flex', gap: 10 }}>
                  {[1, 2, 3, 4, 5].map((star) => (
                    <button
                      key={star}
                      type="button"
                      onClick={() => setFeedbackRating(star)}
                      className={`evo-btn-sm ${feedbackRating >= star ? 'evo-btn-primary' : 'evo-btn-secondary'}`}
                      style={{ fontSize: '1.2rem', padding: '6px 14px' }}
                    >
                      ★ {star}
                    </button>
                  ))}
                </div>
              </div>

              <div className="evo-form-group" style={{ marginTop: 12 }}>
                <label className="evo-label">Feedback Notes / Observations</label>
                <textarea
                  className="evo-textarea"
                  rows={3}
                  placeholder="Note any medical accuracy issues, missing context, or high groundedness observations..."
                  value={feedbackText}
                  onChange={(e) => setFeedbackText(e.target.value)}
                  disabled={isSubmittingFeedback}
                />
              </div>
            </div>
            <div className="evo-modal-foot">
              <button type="button" onClick={() => setFeedbackModalExecution(null)} className="evo-btn-secondary">
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSubmitFeedback}
                className="evo-btn-primary"
                disabled={isSubmittingFeedback}
              >
                {isSubmittingFeedback ? 'Submitting...' : 'Submit Feedback'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* INSPECT CANDIDATE MODAL                                                   */}
      {/* ========================================================================= */}
      {inspectModalCandidate && baselineCandidate && (
        <div className="evo-modal-backdrop" onClick={() => setInspectModalCandidate(null)} role="dialog" aria-modal="true">
          <div className="evo-modal-panel" onClick={(e) => e.stopPropagation()} ref={modalRef}>
            <div className="evo-modal-head">
              <div>
                <div className="evo-modal-badges">
                  <span className={`evo-status-tag ${inspectModalCandidate.candidate_id === 'baseline' ? 'tag-baseline' : inspectModalCandidate.is_pareto_optimal ? 'tag-pareto' : 'tag-dominated'}`}>
                    {inspectModalCandidate.candidate_id === 'baseline' ? 'Baseline' : inspectModalCandidate.is_pareto_optimal ? 'Pareto-Optimal' : 'Dominated'}
                  </span>
                  {renderMutationBadge(inspectModalCandidate)}
                </div>
                <h3 className="evo-modal-heading">{inspectModalCandidate.workflow.name}</h3>
                <span className="evo-modal-subheading">Candidate ID: <code>{inspectModalCandidate.candidate_id}</code></span>
              </div>
              <button type="button" onClick={() => setInspectModalCandidate(null)} className="evo-modal-close">✕</button>
            </div>

            <div className="evo-modal-content">
              {/* Metrics */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Phase 6 Evaluation Metrics</h4>
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Groundedness</span>
                    <span className="evo-field-value evo-val-green">
                      {((inspectModalCandidate.metrics.groundedness ?? inspectModalCandidate.metrics.performance) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Hallucination Rate</span>
                    <span className="evo-field-value evo-val-cyan">
                      {((inspectModalCandidate.metrics.hallucination_rate ?? 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Cost / Query</span>
                    <span className="evo-field-value evo-val-amber">
                      ${inspectModalCandidate.metrics.cost.toFixed(6)}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">LLM Latency</span>
                    <span className="evo-field-value evo-val-purple">
                      {inspectModalCandidate.metrics.llm_latency_ms.toFixed(0)} ms
                    </span>
                  </div>
                </div>
              </div>

              {/* Mutation Metadata */}
              {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata) && (
                <div className="evo-modal-block">
                  <h4 className="evo-modal-block-title">Generation & Lineage Details</h4>
                  <div className="evo-modal-grid">
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Parent Workflow</span>
                      <span className="evo-field-value">
                        <code>{(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.parent_workflow_id}</code>
                      </span>
                    </div>
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Category</span>
                      <span className="evo-field-value">
                        {renderMutationBadge(inspectModalCandidate)}
                      </span>
                    </div>
                    <div className="evo-modal-field evo-metric-card-span">
                      <span className="evo-field-key">Rationale / Description</span>
                      <span className="evo-field-value">
                        {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.mutation_description}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Workload Per-Query Evaluation Evidence (if multi-query workload) */}
              {inspectModalCandidate.metrics.raw_details?.per_query_evaluations &&
                Array.isArray(inspectModalCandidate.metrics.raw_details.per_query_evaluations) &&
                inspectModalCandidate.metrics.raw_details.per_query_evaluations.length > 0 && (
                  <div className="evo-modal-block">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <h4 className="evo-modal-block-title" style={{ margin: 0 }}>
                        Workload Per-Query Evaluation Evidence ({inspectModalCandidate.metrics.raw_details.per_query_evaluations.length} Queries)
                      </h4>
                      <span className="evo-cell-sub">Independent claim & cost measurements</span>
                    </div>

                    <div className="evo-table-container">
                      <table className="evo-table" style={{ fontSize: '0.82rem' }}>
                        <thead>
                          <tr>
                            <th>#</th>
                            <th>Evaluation Query</th>
                            <th>Groundedness</th>
                            <th>Hallucination</th>
                            <th>Cost</th>
                            <th>Tokens (In/Out)</th>
                            <th>Latency</th>
                          </tr>
                        </thead>
                        <tbody>
                          {inspectModalCandidate.metrics.raw_details.per_query_evaluations.map((item: any, idx: number) => (
                            <tr key={idx} className="evo-table-row">
                              <td><strong>Q{idx + 1}</strong></td>
                              <td style={{ maxWidth: 280 }}>
                                <div style={{ fontWeight: 500, color: '#f1f5f9' }}>{item.query}</div>
                                {item.response && (
                                  <details style={{ marginTop: 4 }}>
                                    <summary style={{ fontSize: '0.75rem', color: '#94a3b8', cursor: 'pointer' }}>
                                      View Generated Response
                                    </summary>
                                    <p style={{ margin: '4px 0', fontSize: '0.78rem', color: '#cbd5e1', lineHeight: 1.4 }}>
                                      {item.response}
                                    </p>
                                  </details>
                                )}
                              </td>
                              <td>
                                <strong className="evo-val-green">
                                  {item.groundedness !== undefined && item.groundedness !== null
                                    ? `${(item.groundedness * 100).toFixed(1)}%`
                                    : 'N/A'}
                                </strong>
                              </td>
                              <td>
                                <span className="evo-val-cyan">
                                  {item.hallucination_rate !== undefined && item.hallucination_rate !== null
                                    ? `${(item.hallucination_rate * 100).toFixed(1)}%`
                                    : '0.0%'}
                                </span>
                              </td>
                              <td>
                                <span className="evo-cost-val">${Number(item.cost || 0).toFixed(6)}</span>
                              </td>
                              <td>
                                <span>{item.input_tokens ?? item.tokens ?? 0} / {item.output_tokens ?? 0}</span>
                              </td>
                              <td>
                                <span>{Number(item.llm_latency_ms ?? item.latency_ms ?? 0).toFixed(0)} ms</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
            </div>

            <div className="evo-modal-foot">
              <button type="button" onClick={() => setInspectModalCandidate(null)} className="evo-btn-secondary">
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* INSPECT APPROVED WORKFLOW VERSION MODAL                                    */}
      {/* ========================================================================= */}
      {inspectModalVersion && (
        <div className="evo-modal-backdrop" onClick={() => setInspectModalVersion(null)} role="dialog" aria-modal="true">
          <div className="evo-modal-panel" onClick={(e) => e.stopPropagation()}>
            <div className="evo-modal-head">
              <div>
                <div className="evo-modal-badges">
                  <span className="evo-version-tag">v{inspectModalVersion.version_number ?? 0}</span>
                  <span className="evo-gov-state-pill gov-approved">{inspectModalVersion.status.toUpperCase()}</span>
                </div>
                <h3 className="evo-modal-heading">{inspectModalVersion.name}</h3>
                <span className="evo-modal-subheading">Version ID: <code>{inspectModalVersion.version_id}</code></span>
              </div>
              <button type="button" onClick={() => setInspectModalVersion(null)} className="evo-modal-close">✕</button>
            </div>

            <div className="evo-modal-content">
              {/* Lineage & Derivation */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Workflow Lineage & Strategy</h4>
                <div className="evo-modal-grid">
                  {inspectModalVersion.candidate_type === 'CROSSOVER' || inspectModalVersion.parent_a_version_id ? (
                    <>
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Parent A</span>
                        <span className="evo-field-value">
                          <code>{inspectModalVersion.parent_a_version_id || inspectModalVersion.parent_workflow_id}</code>
                        </span>
                      </div>
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Parent B</span>
                        <span className="evo-field-value">
                          <code>{inspectModalVersion.parent_b_version_id}</code>
                        </span>
                      </div>
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Strategy</span>
                        <span className="evo-field-value">
                          <strong style={{ color: '#8b5cf6' }}>CROSSOVER</strong>
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Parent Workflow</span>
                        <span className="evo-field-value">
                          <code>{inspectModalVersion.parent_workflow_id || 'Production Baseline'}</code>
                        </span>
                      </div>
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Strategy</span>
                        <span className="evo-field-value">
                          <strong>{inspectModalVersion.mutation_type?.toUpperCase() || 'BASELINE'}</strong>
                        </span>
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Raw JSON */}
              <details className="evo-details-block" style={{ marginTop: 12 }}>
                <summary className="evo-details-summary">Immutable Configuration Spec (JSON)</summary>
                <pre className="evo-json-pre">{JSON.stringify(inspectModalVersion.configuration, null, 2)}</pre>
              </details>
            </div>

            <div className="evo-modal-foot">
              <button type="button" onClick={() => setInspectModalVersion(null)} className="evo-btn-secondary">
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
