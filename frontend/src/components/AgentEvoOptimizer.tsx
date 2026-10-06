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
  version_number?: number | null
  name: string
  status: string
  activated_at?: string | null
  configuration: Record<string, any>
  evaluation_snapshot?: Record<string, any> | null
  model_id: string
  provider: string
}

export interface WorkflowExecution {
  execution_id: string
  workflow_version_id: string
  query: string
  model_id: string
  provider: string
  input_tokens: number
  output_tokens: number
  total_tokens: number
  cost: number
  llm_latency_ms: number
  total_execution_time_ms: number
  response: string
  retrieved_doc_ids?: string[] | null
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
  triggered_by: string
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

export const AgentEvoOptimizer: React.FC<AgentEvoOptimizerProps> = ({
  apiBase,
  documents,
  selectedDocId,
  setSelectedDocId,
  isLoggedIn,
}) => {
  // Top-level Navigation Sub-Tabs
  const [activeMainTab, setActiveMainTab] = useState<'optimizer' | 'lifecycle' | 'benchmark' | 'lineage'>('optimizer')

  // Candidate Generation Strategy Selector inside Optimizer: Mutation, Refinement, Crossover
  const [candidateStrategy, setCandidateStrategy] = useState<'mutation' | 'refinement' | 'crossover'>('mutation')

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
        setActiveWorkflow(data)
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
      const res = await fetch(`${apiBase}/api/agent-evo/audit?limit=30`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        if (data.audit_events && Array.isArray(data.audit_events)) {
          setAuditEvents(data.audit_events)
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
    if (!query.trim()) {
      setError('Please provide an evaluation query.')
      return
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
        query: query.trim(),
        strategy: candidateStrategy,
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

  const baselineCandidate = report?.baseline || null

  return (
    <div className="evo-container">
      {/* Top Header */}
      <div className="evo-page-header">
        <div>
          <h1 className="evo-title">AgentEvo Optimization & Workflow Lifecycle</h1>
          <p className="evo-subtitle">
            Autonomous multi-candidate evolution, crossover recombination, Phase 6 claim-groundedness evaluation, Pareto optimization, and production lifecycle governance.
          </p>
        </div>
        <div className="evo-header-meta">
          {activeWorkflow ? (
            <span className="evo-status-pill" style={{ background: 'rgba(16, 185, 129, 0.15)', borderColor: '#10b981', color: '#10b981' }}>
              <span className="evo-dot" style={{ background: '#10b981' }} /> Active: {activeWorkflow.version_id === 'baseline' ? 'Baseline' : `v${activeWorkflow.version_number}`}
            </span>
          ) : (
            <span className="evo-status-pill">
              <span className="evo-dot" /> Autonomous Evolutionary RAG
            </span>
          )}
        </div>
      </div>

      {/* Main Navigation Sub-Tabs */}
      <div className="evo-nav-tabs">
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'optimizer' ? 'active' : ''}`}
          onClick={() => setActiveMainTab('optimizer')}
        >
          <span>🧬</span> Live Evolutionary Optimizer
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'lifecycle' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('lifecycle')
            fetchVersions()
            fetchActiveWorkflow()
            fetchExecutions()
            fetchAuditEvents()
          }}
        >
          <span>🚀</span> Production Versions & Lifecycle
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'benchmark' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('benchmark')
            fetchBenchmarks()
          }}
        >
          <span>⚖️</span> Version Benchmark & Comparison
        </button>
        <button
          type="button"
          className={`evo-nav-tab ${activeMainTab === 'lineage' ? 'active' : ''}`}
          onClick={() => {
            setActiveMainTab('lineage')
            fetchLineage()
          }}
        >
          <span>🌳</span> Workflow Version Lineage
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
      {/* TAB 1: LIVE EVOLUTIONARY OPTIMIZER (Mutation, Refinement, Crossover)       */}
      {/* ========================================================================= */}
      {activeMainTab === 'optimizer' && (
        <>
          <section className="evo-section evo-controls-card">
            <div className="evo-section-header">
              <h2 className="evo-section-title">Optimization Setup & Candidate Generation</h2>
              <span className="evo-section-caption">Configure evaluation preset, target context, and evolutionary candidate generation strategy.</span>
            </div>

            <form onSubmit={handleRunOptimization} className="evo-form">
              {/* Preset Selector */}
              <div className="evo-form-group">
                <label className="evo-label">Evaluation Preset</label>
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

              {/* Evaluation Query Input */}
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
                  placeholder="Enter medical query to evaluate candidates against..."
                  disabled={running}
                />
              </div>

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

              {/* Strategy Selector: Mutation / Exploration | Refinement | Crossover / Recombination */}
              <div className="evo-form-group" style={{ marginTop: 16 }}>
                <label className="evo-label">Candidate Generation Strategy</label>
                <div className="evo-strat-tabs">
                  <button
                    type="button"
                    className={`evo-strat-tab-btn ${candidateStrategy === 'mutation' ? 'active' : ''}`}
                    onClick={() => setCandidateStrategy('mutation')}
                    disabled={running}
                  >
                    <span>🧬</span>
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
                    <span>🎯</span>
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
                    <span>🔀</span>
                    <div>
                      <strong>Crossover / Recombination</strong>
                      <div className="evo-cell-sub">Inherit components from two parent workflows</div>
                    </div>
                  </button>
                </div>
              </div>

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
                    <>
                      <span>🧬</span> Generate & Evaluate Candidates
                    </>
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
                  <p className="evo-report-query">Query: "{report.query}"</p>
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

                    return (
                      <div className="evo-summary-grid">
                        <div className="evo-metric-card">
                          <span className="evo-metric-label">Total Workflows Evaluated</span>
                          <span className="evo-metric-value">{totalWorkflows}</span>
                          <div className="evo-cell-sub" style={{ marginTop: 4, fontWeight: 500 }}>
                            {baselineCount} Baseline + {generatedCount} Generated Candidate{generatedCount === 1 ? '' : 's'}
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
                          <span className="evo-metric-label">Baseline Groundedness</span>
                          <span className="evo-metric-value">
                            {baselineCandidate ? `${((baselineCandidate.metrics.groundedness ?? baselineCandidate.metrics.performance) * 100).toFixed(1)}%` : '100.0%'}
                          </span>
                          <span className="evo-metric-delta">Reference Standard</span>
                        </div>
                        <div className="evo-metric-card">
                          <span className="evo-metric-label">Baseline Cost / Query</span>
                          <span className="evo-metric-value">
                            ${baselineCandidate ? baselineCandidate.metrics.cost.toFixed(6) : '0.000057'}
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
      {/* TAB 2: PRODUCTION VERSIONS & LIFECYCLE                                    */}
      {/* ========================================================================= */}
      {activeMainTab === 'lifecycle' && (
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
                  ⚖️ Benchmark All in Head-to-Head
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
                      <th>User Feedback</th>
                    </tr>
                  </thead>
                  <tbody>
                    {executions.map((exec) => (
                      <tr key={exec.execution_id} className="evo-table-row">
                        <td><code>{exec.execution_id.slice(0, 12)}...</code></td>
                        <td><span className="evo-version-tag">{exec.workflow_version_id}</span></td>
                        <td>
                          <div style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={exec.query}>
                            {exec.query}
                          </div>
                        </td>
                        <td><span className="evo-model-id">{exec.model_id}</span></td>
                        <td>
                          <span className="evo-cell-sub">{exec.total_tokens} ({exec.input_tokens} / {exec.output_tokens})</span>
                        </td>
                        <td><span className="evo-cost-val">${exec.cost.toFixed(6)}</span></td>
                        <td><strong className="evo-val-purple">{exec.llm_latency_ms.toFixed(0)} ms</strong></td>
                        <td><span>{exec.total_execution_time_ms.toFixed(0)} ms</span></td>
                        <td><span className="evo-cell-sub">{new Date(exec.created_at).toLocaleTimeString()}</span></td>
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
                            ⭐ Record Feedback
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
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
                        <td><span className="evo-cell-sub">{evt.previous_version_id || '—'}</span></td>
                        <td><span className="evo-approver-user">{evt.triggered_by}</span></td>
                        <td><span className="evo-cell-sub">{new Date(evt.created_at).toLocaleString()}</span></td>
                        <td>
                          <span className="evo-cell-sub">
                            {evt.details?.reason ? `Reason: ${evt.details.reason}` : evt.details?.candidate_id ? `Cand: ${evt.details.candidate_id}` : 'Governed Event'}
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
      {/* TAB 3: VERSION BENCHMARK & COMPARISON                                     */}
      {/* ========================================================================= */}
      {activeMainTab === 'benchmark' && (
        <>
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
                    <>
                      <span>⚖️</span> Run Reproducible Benchmark
                    </>
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
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkReport.results.map((r) => {
                      const isBase = r.version_id === 'baseline' || r.version === 'baseline'
                      const isPareto = r.is_pareto_optimal
                      return (
                        <tr key={r.version_id} className="evo-table-row">
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
                        </tr>
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
      {/* TAB 4: WORKFLOW VERSION LINEAGE GRAPH                                     */}
      {/* ========================================================================= */}
      {activeMainTab === 'lineage' && (
        <section className="evo-section">
          <div className="evo-section-header">
            <div>
              <h2 className="evo-section-title">Workflow Lineage & Evolutionary Provenance</h2>
              <span className="evo-section-caption">
                Complete directed acyclic provenance graph depicting ancestry, mutations, targeted refinements, and crossover recombination.
              </span>
            </div>
          </div>

          {lineageData && (
            <div className="evo-lineage-container">
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
                            Strategy: <strong>{isCrossover ? 'CROSSOVER' : node.mutation_type?.toUpperCase()}</strong>{' '}
                            {node.refinement_operation ? `(${node.refinement_operation})` : ''}
                          </span>
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
          )}
        </section>
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
