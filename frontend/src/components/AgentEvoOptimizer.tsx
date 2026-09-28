import React, { useState, useEffect, useRef } from 'react'

export type MutationType = 'prompt' | 'retrieval' | 'model' | 'operator' | 'mixed' | 'refinement'

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
  parent_version?: {
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
  // Main Top-level Tab Navigation
  const [activeMainTab, setActiveMainTab] = useState<'optimizer' | 'benchmark' | 'lineage'>('optimizer')

  // Optimization Setup State
  const [activePreset, setActivePreset] = useState<PresetType>('clinical')
  const [query, setQuery] = useState(PRESET_QUERIES.clinical.query)
  const [generatedCandidateCount, setGeneratedCandidateCount] = useState<number>(4)
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all')
  const [selectedParentWorkflowId, setSelectedParentWorkflowId] = useState<string>('baseline')

  // Execution & Pipeline State
  const [running, setRunning] = useState(false)
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0)
  const [report, setReport] = useState<OptimizationReport | null>(null)
  const [, setRuns] = useState<OptimizationReport[]>([])
  const [workflowVersions, setWorkflowVersions] = useState<WorkflowVersion[]>([])
  const [error, setError] = useState<string | null>(null)

  // Interactive Selection State
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null)
  const [inspectModalCandidate, setInspectModalCandidate] = useState<EvaluatedCandidate | null>(null)
  const [inspectModalVersion, setInspectModalVersion] = useState<VersionInspectData | null>(null)
  const [confirmApprovalCandidate, setConfirmApprovalCandidate] = useState<EvaluatedCandidate | null>(null)
  const [approvingId, setApprovingId] = useState<string | null>(null)
  const [isReportCollapsed, setIsReportCollapsed] = useState<boolean>(false)

  // Benchmarking State
  const [selectedBenchmarkVersions, setSelectedBenchmarkVersions] = useState<string[]>(['baseline'])
  const [benchPreset, setBenchPreset] = useState<PresetType>('clinical')
  const [benchQuery, setBenchQuery] = useState<string>(PRESET_QUERIES.clinical.query)
  const [benchDocId, setBenchDocId] = useState<string>(selectedDocId)
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false)
  const [benchmarkReport, setBenchmarkReport] = useState<BenchmarkReport | null>(null)
  const [benchmarkHistory, setBenchmarkHistory] = useState<BenchmarkReport[]>([])
  const [hoveredBenchItem, setHoveredBenchItem] = useState<BenchmarkItemResult | null>(null)

  // Lineage State
  const [lineageData, setLineageData] = useState<LineageData | null>(null)
  const [isLineageLoading, setIsLineageLoading] = useState<boolean>(false)

  // Modal accessibility ref
  const modalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isLoggedIn) {
      fetchRuns()
      fetchVersions()
      fetchBenchmarks()
      fetchLineage()
    }
  }, [isLoggedIn])

  // Sync selected doc with benchmark doc
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
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [inspectModalCandidate, inspectModalVersion, confirmApprovalCandidate])

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
          // Default benchmark versions to baseline + all approved
          const allIds = ['baseline', ...data.versions.map((v: WorkflowVersion) => v.version_id)]
          setSelectedBenchmarkVersions(allIds)
        }
      }
    } catch (err) {
      console.warn('Could not fetch workflow versions', err)
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

  const handleRunOptimization = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!query.trim()) {
      setError('Please provide an evaluation query.')
      return
    }

    setError(null)
    setRunning(true)
    setCurrentStageIndex(1) // Stage 1: Candidate Generation

    const t1 = setTimeout(() => setCurrentStageIndex(2), 1200) // Evaluating
    const t2 = setTimeout(() => setCurrentStageIndex(3), 2800) // Pareto Analysis

    try {
      const payload: {
        query: string
        document_id?: string
        candidate_count: number
        parent_workflow_id?: string
        allowed_mutation_types?: string[]
      } = {
        query: query.trim(),
        candidate_count: generatedCandidateCount,
      }
      if (selectedDocId) {
        payload.document_id = selectedDocId
      }
      if (selectedParentWorkflowId && selectedParentWorkflowId !== 'baseline') {
        payload.parent_workflow_id = selectedParentWorkflowId
      }
      if (selectedStrategy === 'refinement') {
        payload.allowed_mutation_types = ['refinement']
      } else if (selectedStrategy !== 'all') {
        payload.allowed_mutation_types = [selectedStrategy]
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
    } catch (err: any) {
      setError(err.message || 'Failed to record developer approval')
    } finally {
      setApprovingId(null)
    }
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

  const compareSelectedVersions = () => {
    if (selectedBenchmarkVersions.length === 0) {
      setSelectedBenchmarkVersions(['baseline', ...workflowVersions.map((v) => v.version_id)])
    }
    setActiveMainTab('benchmark')
  }

  const renderMutationBadge = (cand: EvaluatedCandidate) => {
    const meta = cand.mutation_metadata || cand.workflow.mutation_metadata
    if (!meta) {
      return <span className="evo-mut-tag mut-baseline">baseline</span>
    }
    const type = meta.mutation_type || 'custom'
    return (
      <span className={`evo-mut-tag mut-${type} ${type === 'refinement' ? 'evo-mut-refinement' : ''}`}>
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
          <h1 className="evo-title">AgentEvo Optimization & Benchmarking Engine</h1>
          <p className="evo-subtitle">
            Autonomous multi-candidate evolution, Phase 6 claim-groundedness evaluation, Pareto trade-off optimization, and reproducible head-to-head benchmarking.
          </p>
        </div>
        <div className="evo-header-meta">
          <span className="evo-status-pill">
            <span className="evo-dot" /> Autonomous Evolutionary RAG
          </span>
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
      {/* TAB 1: LIVE EVOLUTIONARY OPTIMIZER                                       */}
      {/* ========================================================================= */}
      {activeMainTab === 'optimizer' && (
        <>
          {/* Optimization Controls Section */}
          <section className="evo-section evo-controls-card">
            <div className="evo-section-header">
              <h2 className="evo-section-title">Optimization Setup & Candidate Generation</h2>
              <span className="evo-section-caption">Configure query preset, target document context, and evolutionary mutation strategy.</span>
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

              {/* Document Selection, Candidate Count & Mutation Strategy */}
              <div className="evo-form-row">
                <div className="evo-form-group evo-col-third">
                  <label htmlFor="evo-doc-select" className="evo-label">Target Document Context</label>
                  <select
                    id="evo-doc-select"
                    className="evo-select"
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    disabled={running}
                  >
                    <option value="">All Uploaded Documents (Global)</option>
                    {documents.map((doc) => (
                      <option key={doc.id} value={doc.id}>
                        {doc.filename || doc.original_filename || doc.id}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="evo-form-group evo-col-third">
                  <label htmlFor="evo-parent-select" className="evo-label">Parent Workflow (Baseline / Approved)</label>
                  <select
                    id="evo-parent-select"
                    className="evo-select"
                    value={selectedParentWorkflowId}
                    onChange={(e) => setSelectedParentWorkflowId(e.target.value)}
                    disabled={running}
                  >
                    <option value="baseline">Production Baseline (Default)</option>
                    {workflowVersions.map((v) => (
                      <option key={v.version_id} value={v.version_id}>
                        v{v.version_number}: {v.name} ({v.model_id})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="evo-form-group evo-col-third">
                  <label htmlFor="evo-count-input" className="evo-label">Candidate Mutations ({generatedCandidateCount})</label>
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

              {/* Strategy Selector Pills */}
              <div className="evo-form-group">
                <label className="evo-label">Mutation & Refinement Strategy</label>
                <div className="evo-strategy-pills-wrap">
                  {[
                    { id: 'all', label: 'All Dimensions', desc: 'Diverse Phase 7.1/7.2 exploration' },
                    { id: 'refinement', label: 'Refinement Only', desc: 'Phase 7.3 simplification & cost optimization' },
                    { id: 'prompt', label: 'Prompt Mutations', desc: 'Prompt template & detail variations' },
                    { id: 'retrieval', label: 'Retrieval Mutations', desc: 'Top-K & threshold tuning' },
                    { id: 'model', label: 'Model Mutations', desc: 'Model provider switches' },
                    { id: 'operator', label: 'Operator Mutations', desc: 'Execution & compression nodes' },
                    { id: 'mixed', label: 'Composite Mixed', desc: 'Multi-parameter exploration' },
                  ].map((strat) => (
                    <button
                      key={strat.id}
                      type="button"
                      onClick={() => setSelectedStrategy(strat.id)}
                      className={`evo-strategy-pill ${selectedStrategy === strat.id ? 'evo-strat-active' : ''} evo-strat-${strat.id}`}
                      disabled={running}
                    >
                      <span className="evo-strat-title">
                        {selectedStrategy === strat.id ? '✓ ' : ''}{strat.label}
                      </span>
                      <span className="evo-strat-desc">{strat.desc}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="evo-form-actions">
                <button
                  type="submit"
                  className="evo-btn-primary"
                  disabled={running || !query.trim()}
                >
                  {running ? (
                    <>
                      <span className="evo-spinner" /> Running Optimization...
                    </>
                  ) : (
                    <>
                      <span>🧬</span> Run Evolutionary Optimization
                    </>
                  )}
                </button>
              </div>
            </form>
          </section>

          {/* Stepper Pipeline */}
          {running && (
            <section className="evo-section evo-pipeline-section" aria-live="polite">
              <div className="evo-process-stepper">
                {[
                  { step: 1, name: 'Reference Workflow', desc: 'Evaluating Baseline/Parent' },
                  { step: 2, name: 'Candidate Generation', desc: `${selectedStrategy.toUpperCase()} mutations` },
                  { step: 3, name: 'Phase 6 Evaluation', desc: 'Decomposing claims & calculating cost' },
                  { step: 4, name: 'Pareto Optimization', desc: 'Groundedness ↑ vs Cost ↓ Frontier' },
                ].map((s) => {
                  const isDone = currentStageIndex > s.step
                  const isCurrent = currentStageIndex === s.step
                  return (
                    <div key={s.step} className={`evo-process-step ${isDone ? 'step-done' : ''} ${isCurrent ? 'step-running' : ''}`}>
                      <div className="evo-step-indicator">
                        {isDone ? '✓' : s.step}
                      </div>
                      <div className="evo-step-details">
                        <span className="evo-step-name">{s.name}</span>
                        <span className="evo-step-caption">{s.desc}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            </section>
          )}

          {/* Optimization Report / Pareto Frontier */}
          {report && (
            <section className="evo-section evo-report-section">
              <div className="evo-section-header">
                <div className="evo-header-with-badge">
                  <h2 className="evo-section-title">Pareto Optimization Report</h2>
                  <span className="evo-frontier-badge">{report.pareto_frontier_count} Pareto-Optimal Workflows</span>
                </div>
                <div className="evo-report-actions">
                  <button
                    type="button"
                    onClick={() => setIsReportCollapsed(!isReportCollapsed)}
                    className="evo-btn-secondary evo-btn-sm"
                  >
                    {isReportCollapsed ? 'Expand Report' : 'Collapse'}
                  </button>
                </div>
              </div>

              {!isReportCollapsed && (
                <div className="evo-report-body">
                  {/* Candidates Comparison Table */}
                  <div className="evo-table-container">
                    <table className="evo-table">
                      <thead>
                        <tr>
                          <th>Status</th>
                          <th>Workflow Candidate</th>
                          <th>Strategy</th>
                          <th>Model</th>
                          <th>Groundedness ↑</th>
                          <th>Cost / Query ↓</th>
                          <th>Latency</th>
                          <th>Pareto Dominance</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {report.all_candidates.map((cand) => {
                          const isBaseline = cand.candidate_id === 'baseline' || cand.candidate_id === report.baseline.candidate_id
                          const isPareto = cand.is_pareto_optimal
                          const groundedness = cand.metrics.groundedness ?? cand.metrics.performance
                          const isApproved = report.approved_workflow_id === cand.candidate_id

                          return (
                            <tr
                              key={cand.candidate_id}
                              className={`evo-table-row ${selectedCandidateId === cand.candidate_id ? 'row-selected' : ''} ${isPareto ? 'row-pareto' : ''}`}
                              onClick={() => setSelectedCandidateId(cand.candidate_id)}
                            >
                              <td>
                                {isBaseline ? (
                                  <span className="evo-status-tag tag-baseline">Baseline</span>
                                ) : isPareto ? (
                                  <span className="evo-status-tag tag-pareto">Pareto</span>
                                ) : (
                                  <span className="evo-status-tag tag-dominated">Dominated</span>
                                )}
                              </td>
                              <td>
                                <strong>{cand.workflow.name}</strong>
                                <div className="evo-cell-sub"><code>{cand.candidate_id}</code></div>
                              </td>
                              <td>{renderMutationBadge(cand)}</td>
                              <td>
                                <div className="evo-model-cell">
                                  <span className="evo-provider-name">{cand.workflow.model.provider}</span>
                                  <span className="evo-model-id">{cand.workflow.model.model_id}</span>
                                </div>
                              </td>
                              <td>
                                <span className="evo-perf-val">{(groundedness * 100).toFixed(1)}%</span>
                              </td>
                              <td>
                                <span className="evo-cost-val">${cand.metrics.cost.toFixed(6)}</span>
                              </td>
                              <td>
                                <span className="evo-latency-val">{cand.metrics.llm_latency_ms.toFixed(0)} ms</span>
                              </td>
                              <td>
                                {isPareto ? (
                                  <span className="evo-pareto-optimal-badge">✓ Non-dominated</span>
                                ) : (
                                  <span className="evo-dominated-badge">Dominated by {cand.dominated_by?.length || 1}</span>
                                )}
                              </td>
                              <td>
                                <div className="evo-action-buttons">
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
                                  {isPareto && !isBaseline && !isApproved && (
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        setConfirmApprovalCandidate(cand)
                                      }}
                                      className="evo-btn-approve"
                                      disabled={approvingId === cand.candidate_id}
                                    >
                                      {approvingId === cand.candidate_id ? 'Approving...' : 'Approve'}
                                    </button>
                                  )}
                                  {isApproved && (
                                    <span className="evo-approved-tag">✓ Approved</span>
                                  )}
                                </div>
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Approved Workflow Versions Section */}
          <section className="evo-section evo-versions-section">
            <div className="evo-section-header">
              <div className="evo-header-with-badge">
                <h2 className="evo-section-title">Approved Workflow Versions</h2>
                <span className="evo-count-badge">{workflowVersions.length} Available</span>
              </div>
              <div className="evo-bench-actions">
                <button
                  type="button"
                  onClick={compareSelectedVersions}
                  className="evo-btn-secondary evo-btn-sm"
                >
                  ⚖️ Compare Selected ({selectedBenchmarkVersions.length})
                </button>
                <button
                  type="button"
                  onClick={selectAllVersionsForBenchmark}
                  className="evo-btn-primary evo-btn-sm"
                >
                  🚀 Benchmark All Versions
                </button>
                <button
                  type="button"
                  onClick={() => setActiveMainTab('lineage')}
                  className="evo-btn-secondary evo-btn-sm"
                >
                  🌳 View Lineage Graph
                </button>
              </div>
            </div>

            {workflowVersions.length === 0 ? (
              <div className="evo-versions-empty">
                <p>No approved workflow versions recorded yet. Inspect and approve a Pareto-optimal candidate above to persist a version.</p>
              </div>
            ) : (
              <div className="evo-table-container">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th style={{ width: 40 }}>Select</th>
                      <th>Version</th>
                      <th>Workflow Name</th>
                      <th>Source Candidate</th>
                      <th>Model</th>
                      <th>Groundedness</th>
                      <th>Cost / Query</th>
                      <th>Approved At</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {/* Baseline row */}
                    <tr className="evo-table-row">
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedBenchmarkVersions.includes('baseline')}
                          onChange={() => toggleBenchmarkVersion('baseline')}
                          aria-label="Select baseline for benchmark"
                        />
                      </td>
                      <td>
                        <span className="evo-version-tag" style={{ background: '#0284c7' }}>Base</span>
                        <div className="evo-cell-sub"><code>baseline</code></div>
                      </td>
                      <td><strong>Production Baseline</strong></td>
                      <td><span className="evo-cell-sub">Reference System</span></td>
                      <td><span className="evo-model-id">gemini-3.5-flash-lite</span></td>
                      <td><span className="evo-perf-val">100.0%</span></td>
                      <td><span className="evo-cost-val">$0.000057</span></td>
                      <td><span className="evo-cell-sub">Immutable Baseline</span></td>
                      <td><span className="evo-gov-state-pill gov-approved">ACTIVE</span></td>
                      <td>
                        <button
                          type="button"
                          onClick={() => handleInspectVersion('baseline')}
                          className="evo-btn-inspect"
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>

                    {workflowVersions.map((ver) => {
                      const snap = ver.evaluation_snapshot || {}
                      const groundedness = snap.groundedness ?? snap.performance ?? null
                      const cost = snap.cost ?? null
                      const formattedDate = ver.approved_at ? new Date(ver.approved_at).toLocaleString() : 'N/A'
                      const isChecked = selectedBenchmarkVersions.includes(ver.version_id)

                      return (
                        <tr key={ver.id || ver.version_id} className="evo-table-row">
                          <td>
                            <input
                              type="checkbox"
                              checked={isChecked}
                              onChange={() => toggleBenchmarkVersion(ver.version_id)}
                              aria-label={`Select version ${ver.version_id} for benchmark`}
                            />
                          </td>
                          <td>
                            <span className="evo-version-tag">v{ver.version_number}</span>
                            <div className="evo-cell-sub"><code>{ver.version_id}</code></div>
                          </td>
                          <td>
                            <strong>{ver.name}</strong>
                            {ver.description && <div className="evo-cell-sub">{ver.description}</div>}
                          </td>
                          <td>
                            <div className="evo-cell-sub">Candidate: <code>{ver.source_candidate_id}</code></div>
                            <div className="evo-cell-sub">Run: <code>{ver.source_run_id}</code></div>
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
                            <span className="evo-gov-state-pill gov-approved">{ver.status.toUpperCase()}</span>
                          </td>
                          <td>
                            <button
                              type="button"
                              onClick={() => handleInspectVersion(ver.version_id)}
                              className="evo-btn-inspect"
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
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
      {/* TAB 2: VERSION BENCHMARK & COMPARISON                                     */}
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

              {/* Benchmark Metadata Card */}
              <div className="evo-summary-box" style={{ marginBottom: 16 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '8px' }}>
                  <div><strong>Benchmark ID:</strong> <code>{benchmarkReport.benchmark_id}</code></div>
                  <div><strong>Timestamp:</strong> <span>{new Date(benchmarkReport.created_at).toLocaleString()}</span></div>
                  <div><strong>Evaluation Preset:</strong> <span>{benchmarkReport.preset_name || 'clinical'}</span></div>
                  <div><strong>Versions Evaluated:</strong> <span>{benchmarkReport.selected_versions?.length || benchmarkReport.results?.length || 0}</span></div>
                  <div><strong>Repetitions:</strong> <span>{benchmarkReport.repetitions || benchmarkReport.results[0]?.iterations_count || 1} per version</span></div>
                  <div><strong>Documents:</strong> <span>{benchmarkReport.document_ids?.join(', ') || benchmarkReport.document_id || 'Global Ingested Knowledge Base'}</span></div>
                </div>
                <div style={{ marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '8px' }}>
                  <strong>Query:</strong> <em>"{benchmarkReport.query}"</em>
                </div>
                <div style={{ marginTop: '4px' }}>
                  <strong>Summary:</strong> {benchmarkReport.summary}
                </div>
              </div>

              {/* Pareto Scatter Plot Visualization */}
              <div className="evo-bench-scatter-container">
                <div className="evo-bench-scatter-head">
                  <div className="evo-bench-scatter-title">
                    <span>📊</span> Pareto Trade-off Frontier (Groundedness ↑ vs Cost / Query ↓)
                  </div>
                  <div className="evo-bench-legend">
                    <div className="evo-legend-item">
                      <span className="evo-legend-dot baseline" /> Production Baseline
                    </div>
                    <div className="evo-legend-item">
                      <span className="evo-legend-dot pareto" /> Pareto-Optimal (Non-dominated)
                    </div>
                    <div className="evo-legend-item">
                      <span className="evo-legend-dot dominated" /> Dominated
                    </div>
                  </div>
                </div>

                {/* SVG Scatter Plot */}
                <div className="evo-bench-svg-wrap">
                  <svg width="100%" height="100%" viewBox="0 0 800 240" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="60" y1="20" x2="760" y2="20" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />
                    <line x1="60" y1="75" x2="760" y2="75" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />
                    <line x1="60" y1="130" x2="760" y2="130" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />
                    <line x1="60" y1="185" x2="760" y2="185" stroke="rgba(255,255,255,0.06)" strokeDasharray="4" />
                    <line x1="60" y1="200" x2="760" y2="200" stroke="rgba(255,255,255,0.2)" />
                    <line x1="60" y1="20" x2="60" y2="200" stroke="rgba(255,255,255,0.2)" />

                    {/* Y-axis Labels (Groundedness %) */}
                    <text x="15" y="25" fill="#94a3b8" fontSize="10">100%</text>
                    <text x="20" y="80" fill="#94a3b8" fontSize="10">75%</text>
                    <text x="20" y="135" fill="#94a3b8" fontSize="10">50%</text>
                    <text x="20" y="190" fill="#94a3b8" fontSize="10">25%</text>

                    {/* X-axis Label */}
                    <text x="400" y="228" fill="#94a3b8" fontSize="11" textAnchor="middle">Cost per Query ($) →</text>
                    <text x="20" y="110" fill="#94a3b8" fontSize="11" transform="rotate(-90 20,110)" textAnchor="middle">Groundedness ↑</text>

                    {/* Data Points */}
                    {benchmarkReport.results.map((item) => {
                      const itemCost = item.cost_per_query ?? item.cost
                      const maxCost = Math.max(...benchmarkReport.results.map((r) => r.cost_per_query ?? r.cost), 0.0001)
                      const minCost = Math.min(...benchmarkReport.results.map((r) => r.cost_per_query ?? r.cost), 0.0)
                      const costRange = maxCost - minCost || 0.0001

                      // X scale from 90 to 730
                      const cx = 90 + ((itemCost - minCost) / costRange) * 620
                      // Y scale from 200 (0%) to 30 (100%)
                      const cy = 200 - item.groundedness * 170

                      const isBaseline = item.version_id === 'baseline' || item.version === 'baseline'
                      const isPareto = item.is_pareto_optimal

                      return (
                        <g
                          key={item.version_id}
                          onMouseEnter={() => setHoveredBenchItem(item)}
                          onMouseLeave={() => setHoveredBenchItem(null)}
                          style={{ cursor: 'pointer' }}
                        >
                          {isBaseline ? (
                            <rect
                              x={cx - 7}
                              y={cy - 7}
                              width={14}
                              height={14}
                              fill="#38bdf8"
                              stroke="#ffffff"
                              strokeWidth={2}
                              rx={2}
                            />
                          ) : (
                            <circle
                              cx={cx}
                              cy={cy}
                              r={isPareto ? 8 : 6}
                              fill={isPareto ? '#10b981' : '#64748b'}
                              stroke={isPareto ? '#34d399' : '#94a3b8'}
                              strokeWidth={isPareto ? 2.5 : 1}
                            />
                          )}
                          <text
                            x={cx}
                            y={cy - 12}
                            fill="#f8fafc"
                            fontSize="10"
                            fontWeight="bold"
                            textAnchor="middle"
                          >
                            {isBaseline ? 'Base' : `v${item.version_number}`}
                          </text>
                        </g>
                      )
                    })}
                  </svg>

                  {/* Tooltip */}
                  {hoveredBenchItem && (
                    <div className="evo-bench-tooltip" style={{ top: 10, right: 10 }}>
                      <div className="evo-bench-tooltip-title">
                        {hoveredBenchItem.version_id === 'baseline' ? 'Production Baseline' : `v${hoveredBenchItem.version_number}: ${hoveredBenchItem.workflow || hoveredBenchItem.workflow_name}`}
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>Pareto Status:</span>
                        <strong style={{ color: hoveredBenchItem.is_pareto_optimal ? '#34d399' : '#94a3b8' }}>
                          {hoveredBenchItem.pareto_status}
                        </strong>
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>Groundedness:</span>
                        <strong>{(hoveredBenchItem.groundedness * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>Hallucination Rate:</span>
                        <strong>{((hoveredBenchItem.hallucination ?? hoveredBenchItem.hallucination_rate) * 100).toFixed(1)}%</strong>
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>Cost / Query:</span>
                        <strong>${(hoveredBenchItem.cost_per_query ?? hoveredBenchItem.cost).toFixed(6)}</strong>
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>LLM Latency:</span>
                        <span>{(hoveredBenchItem.latency_ms ?? hoveredBenchItem.llm_latency_ms).toFixed(0)} ms</span>
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>Total Tokens:</span>
                        <span>{hoveredBenchItem.total_tokens}</span>
                      </div>
                      <div className="evo-bench-tooltip-row">
                        <span>Model:</span>
                        <span>{hoveredBenchItem.model || hoveredBenchItem.model_id}</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* 1. Canonical Aggregate Results Table */}
              <div className="evo-table-container" style={{ marginTop: 20 }}>
                <h3 className="evo-section-title" style={{ padding: '8px 12px', margin: 0, fontSize: '1rem' }}>
                  📋 Fresh Cross-Version Benchmark Results (Aggregate Mean)
                </h3>
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Workflow</th>
                      <th>Groundedness</th>
                      <th>Hallucination</th>
                      <th>Cost/Query</th>
                      <th>Tokens</th>
                      <th>Latency</th>
                      <th>Model</th>
                      <th>Provider</th>
                      <th>Retrieval</th>
                      <th>Prompt</th>
                      <th>Operators</th>
                      <th>Pareto</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkReport.results.map((res) => {
                      const isBaseline = res.version_id === 'baseline' || res.version === 'baseline'
                      const isPareto = res.is_pareto_optimal
                      const versionLabel = isBaseline ? 'Base' : `v${res.version_number ?? '?'}`
                      const workflowName = res.workflow || res.workflow_name
                      const groundedness = res.groundedness
                      const hallucination = res.hallucination ?? res.hallucination_rate
                      const cost = res.cost_per_query ?? res.cost
                      const tokens = res.total_tokens
                      const latency = res.latency_ms ?? res.llm_latency_ms
                      const model = res.model || res.model_id
                      const provider = res.provider
                      const retrievalStr = res.retrieval_config || res.retrieval_config_summary || `top_k=${res.retrieval?.top_k ?? 5}`
                      const promptStr = res.prompt_strategy || res.prompt_template
                      const operatorStr = `compress=${res.context_compression ? 'True' : 'False'}`

                      return (
                        <tr
                          key={res.version_id}
                          className={`evo-table-row ${isPareto ? 'row-pareto' : ''}`}
                        >
                          <td>
                            <span className="evo-version-tag" style={{ background: isBaseline ? '#0284c7' : undefined }}>
                              {versionLabel}
                            </span>
                            <div className="evo-cell-sub"><code>{res.version_id}</code></div>
                          </td>
                          <td>
                            <strong>{workflowName}</strong>
                          </td>
                          <td>
                            <span className="evo-perf-val">{(groundedness * 100).toFixed(1)}%</span>
                          </td>
                          <td>
                            <span className="evo-val-cyan">{(hallucination * 100).toFixed(1)}%</span>
                          </td>
                          <td>
                            <span className="evo-cost-val">${cost.toFixed(6)}</span>
                          </td>
                          <td>
                            <span className="evo-cell-sub">{tokens}</span>
                          </td>
                          <td>
                            <span className="evo-latency-val">{latency.toFixed(0)} ms</span>
                          </td>
                          <td>
                            <span className="evo-model-id">{model}</span>
                          </td>
                          <td>
                            <span className="evo-provider-name">{provider}</span>
                          </td>
                          <td>
                            <span className="evo-cell-sub">{retrievalStr}</span>
                          </td>
                          <td>
                            <span className="evo-cell-sub">{promptStr}</span>
                          </td>
                          <td>
                            <span className="evo-cell-sub">{operatorStr}</span>
                          </td>
                          <td>
                            {isPareto ? (
                              <span className="evo-pareto-optimal-badge">✓ Pareto-optimal</span>
                            ) : (
                              <span className="evo-dominated-badge">Dominated</span>
                            )}
                          </td>
                          <td>
                            <button
                              type="button"
                              onClick={() => handleInspectVersion(res.version_id)}
                              className="evo-btn-inspect"
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* 2. Relative to Production Baseline Table */}
              <div className="evo-table-container" style={{ marginTop: 20 }}>
                <h3 className="evo-section-title" style={{ padding: '8px 12px', margin: 0, fontSize: '1rem' }}>
                  📈 Performance & Efficiency Relative to Production Baseline
                </h3>
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Workflow</th>
                      <th>Groundedness Δ</th>
                      <th>Cost Δ%</th>
                      <th>Latency Δ%</th>
                      <th>Tokens Δ%</th>
                      <th>Pareto Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkReport.results.map((res) => {
                      const isBaseline = res.version_id === 'baseline' || res.version === 'baseline'
                      const gDelta = res.relative_changes?.groundedness_delta ?? res.relative_changes?.groundedness_change ?? 0
                      const cChange = res.relative_changes?.cost_change_pct ?? -(res.relative_changes?.cost_reduction_pct ?? 0)
                      const lChange = res.relative_changes?.latency_change_pct ?? -(res.relative_changes?.latency_reduction_pct ?? 0)
                      const tChange = res.relative_changes?.token_change_pct ?? -(res.relative_changes?.token_reduction_pct ?? 0)

                      return (
                        <tr key={res.version_id} className="evo-table-row">
                          <td>
                            <span className="evo-version-tag" style={{ background: isBaseline ? '#0284c7' : undefined }}>
                              {isBaseline ? 'Base' : `v${res.version_number}`}
                            </span>
                          </td>
                          <td><strong>{res.workflow || res.workflow_name}</strong></td>
                          <td>
                            <span className={gDelta >= 0 ? 'evo-perf-val' : 'evo-cost-val'}>
                              {gDelta >= 0 ? '+' : ''}{gDelta.toFixed(4)}
                            </span>
                          </td>
                          <td>
                            <span className={cChange <= 0 ? 'evo-perf-val' : 'evo-cost-val'}>
                              {cChange > 0 ? '+' : ''}{cChange.toFixed(2)}%
                            </span>
                          </td>
                          <td>
                            <span className={lChange <= 0 ? 'evo-perf-val' : 'evo-cost-val'}>
                              {lChange > 0 ? '+' : ''}{lChange.toFixed(2)}%
                            </span>
                          </td>
                          <td>
                            <span className={tChange <= 0 ? 'evo-perf-val' : 'evo-cost-val'}>
                              {tChange > 0 ? '+' : ''}{tChange.toFixed(2)}%
                            </span>
                          </td>
                          <td>
                            {res.is_pareto_optimal ? (
                              <span className="evo-pareto-optimal-badge">✓ Pareto-optimal</span>
                            ) : (
                              <span className="evo-dominated-badge">Dominated</span>
                            )}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>

              {/* 3. Repetition Details Table (R1, R2, R3, Mean, Std Dev) */}
              <div className="evo-table-container" style={{ marginTop: 20 }}>
                <h3 className="evo-section-title" style={{ padding: '8px 12px', margin: 0, fontSize: '1rem' }}>
                  🔬 Repetition Breakdown & Variance (R1, R2, R3, Mean, Std Dev)
                </h3>
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Version</th>
                      <th>Workflow / Run</th>
                      <th>Groundedness</th>
                      <th>Hallucination</th>
                      <th>Cost / Query</th>
                      <th>Tokens</th>
                      <th>Latency</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkReport.results.flatMap((res) => {
                      const runs = res.runs || []
                      const rows = runs.map((r) => (
                        <tr key={`${res.version_id}-r${r.iteration}`} className="evo-table-row">
                          <td><code>{res.version_id}</code></td>
                          <td><span className="evo-count-badge">Repetition {r.iteration}</span></td>
                          <td>{(r.groundedness * 100).toFixed(1)}%</td>
                          <td>{(r.hallucination_rate * 100).toFixed(1)}%</td>
                          <td>${r.cost.toFixed(6)}</td>
                          <td>{r.total_tokens}</td>
                          <td>{r.llm_latency_ms.toFixed(0)} ms</td>
                        </tr>
                      ))
                      if (runs.length > 0) {
                        rows.push(
                          <tr key={`${res.version_id}-mean`} style={{ background: 'rgba(59, 130, 246, 0.08)', fontWeight: 'bold' }}>
                            <td><code>{res.version_id}</code></td>
                            <td><span className="evo-status-tag tag-pareto">Mean ± Std Dev</span></td>
                            <td>{(res.groundedness * 100).toFixed(1)}% ± {((res.std_dev?.groundedness ?? 0) * 100).toFixed(2)}%</td>
                            <td>{((res.hallucination ?? res.hallucination_rate) * 100).toFixed(1)}% ± {((res.std_dev?.hallucination ?? 0) * 100).toFixed(2)}%</td>
                            <td>${(res.cost_per_query ?? res.cost).toFixed(6)} ± ${(res.std_dev?.cost ?? 0).toFixed(6)}</td>
                            <td>{res.total_tokens} ± {(res.std_dev?.tokens ?? 0).toFixed(1)}</td>
                            <td>{(res.latency_ms ?? res.llm_latency_ms).toFixed(0)} ms ± {(res.std_dev?.latency_ms ?? 0).toFixed(0)} ms</td>
                          </tr>
                        )
                      }
                      return rows
                    })}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {/* Past Benchmark History */}
          {benchmarkHistory.length > 0 && (
            <section className="evo-section">
              <div className="evo-section-header">
                <h3 className="evo-section-title">Benchmark History</h3>
                <span className="evo-section-caption">Prior reproducible benchmark executions</span>
              </div>
              <div className="evo-table-container">
                <table className="evo-table">
                  <thead>
                    <tr>
                      <th>Benchmark ID</th>
                      <th>Benchmark Name</th>
                      <th>Preset</th>
                      <th>Versions Evaluated</th>
                      <th>Created By</th>
                      <th>Timestamp</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkHistory.map((b) => (
                      <tr key={b.benchmark_id} className="evo-table-row">
                        <td><code>{b.benchmark_id}</code></td>
                        <td><strong>{b.name}</strong></td>
                        <td><span className="evo-cell-sub">{b.preset_name || 'custom'}</span></td>
                        <td><span className="evo-count-badge">{b.selected_versions?.length || 0} Versions</span></td>
                        <td><span className="evo-cell-sub">{b.created_by}</span></td>
                        <td><span className="evo-cell-sub">{new Date(b.created_at).toLocaleString()}</span></td>
                        <td>
                          <button
                            type="button"
                            onClick={() => setBenchmarkReport(b)}
                            className="evo-btn-secondary evo-btn-sm"
                          >
                            Load Report
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: WORKFLOW VERSION LINEAGE GRAPH                                     */}
      {/* ========================================================================= */}
      {activeMainTab === 'lineage' && (
        <section className="evo-section evo-lineage-container">
          <div className="evo-section-header">
            <div>
              <h2 className="evo-section-title">Workflow Lineage & Derivation Tree</h2>
              <span className="evo-section-caption">
                Visualizing true parent-to-child relationships from database metadata. Baseline remains immutable reference root.
              </span>
            </div>
            <button
              type="button"
              onClick={fetchLineage}
              className="evo-btn-secondary evo-btn-sm"
              disabled={isLineageLoading}
            >
              🔄 Refresh Lineage
            </button>
          </div>

          {isLineageLoading ? (
            <div className="evo-versions-empty">
              <span className="evo-spinner" /> Loading lineage graph...
            </div>
          ) : !lineageData || lineageData.nodes.length === 0 ? (
            <div className="evo-versions-empty">
              <p>No workflow version lineage found.</p>
            </div>
          ) : (
            <div className="evo-lineage-flow">
              {/* Root node: Baseline */}
              <div className="evo-lineage-level">
                {lineageData.nodes
                  .filter((n) => n.version_id === 'baseline')
                  .map((rootNode) => (
                    <div key={rootNode.version_id} className="evo-lineage-card is-baseline">
                      <div className="evo-lineage-card-head">
                        <span className="evo-lineage-tag" style={{ background: '#0284c7' }}>ROOT BASELINE</span>
                        <span className="evo-gov-state-pill gov-approved">IMMUTABLE</span>
                      </div>
                      <div className="evo-lineage-name">{rootNode.name}</div>
                      <div className="evo-lineage-meta">
                        <span>Model: <code>{rootNode.model_id}</code></span>
                        <span>Provider: {rootNode.provider}</span>
                        <span>{rootNode.rationale}</span>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleInspectVersion('baseline')}
                        className="evo-btn-inspect"
                      >
                        Inspect Baseline
                      </button>
                    </div>
                  ))}
              </div>

              {/* Child Nodes */}
              <div className="evo-lineage-level">
                {lineageData.nodes
                  .filter((n) => n.version_id !== 'baseline')
                  .map((node) => {
                    const edge = lineageData.edges.find((e) => e.target === node.version_id)
                    const parentLabel = edge?.source === 'baseline' ? 'baseline' : edge?.source

                    return (
                      <div
                        key={node.version_id}
                        className={`evo-lineage-card ${node.mutation_type === 'refinement' ? 'is-refined' : ''}`}
                      >
                        <div className="evo-lineage-card-head">
                          <span className="evo-lineage-tag">v{node.version_number}</span>
                          <span className="evo-lineage-arrow-tag">
                            ↓ from <code>{parentLabel}</code> ({edge?.label || 'mutation'})
                          </span>
                        </div>
                        <div className="evo-lineage-name">{node.name}</div>
                        <div className="evo-lineage-meta">
                          <span>Version ID: <code>{node.version_id}</code></span>
                          <span>Candidate: <code>{node.source_candidate_id}</code></span>
                          <span>Strategy: <strong>{node.mutation_type?.toUpperCase()}</strong> {node.refinement_operation ? `(${node.refinement_operation})` : ''}</span>
                          {node.fields_changed && node.fields_changed.length > 0 && (
                            <span>Fields: {node.fields_changed.join(', ')}</span>
                          )}
                          {node.rationale && <span>Rationale: {node.rationale}</span>}
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
      {/* INSPECT WORKFLOW CANDIDATE MODAL                                          */}
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
              {/* Mutation Metadata */}
              {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata) && (
                <div className="evo-modal-block">
                  <h4 className="evo-modal-block-title">Mutation & Lineage Details</h4>
                  <div className="evo-modal-grid">
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Parent Workflow</span>
                      <span className="evo-field-value">
                        <code>{(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.parent_workflow_id}</code>
                      </span>
                    </div>
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Mutation Category</span>
                      <span className="evo-field-value">
                        {renderMutationBadge(inspectModalCandidate)}
                      </span>
                    </div>
                    {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.refinement_operation && (
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Refinement Operation</span>
                        <span className="evo-field-value">
                          <code className="evo-refine-op-tag">
                            {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.refinement_operation}
                          </code>
                        </span>
                      </div>
                    )}
                    {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.fields_changed && (
                      <div className="evo-modal-field">
                        <span className="evo-field-key">Fields Changed</span>
                        <span className="evo-field-value">
                          {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.fields_changed?.join(', ')}
                        </span>
                      </div>
                    )}
                    <div className="evo-modal-field evo-metric-card-span">
                      <span className="evo-field-key">Rationale</span>
                      <span className="evo-field-value">
                        {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.mutation_description}
                      </span>
                    </div>

                    {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.before_configuration && (
                      <div className="evo-modal-field evo-metric-card-span evo-refine-diff-box">
                        <div className="evo-refine-diff-col">
                          <span className="evo-field-key">Before Configuration (Parent)</span>
                          <pre className="evo-refine-cfg-pre">
                            {JSON.stringify((inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.before_configuration, null, 2)}
                          </pre>
                        </div>
                        <div className="evo-refine-arrow">→</div>
                        <div className="evo-refine-diff-col">
                          <span className="evo-field-key">After Configuration (Refined)</span>
                          <pre className="evo-refine-cfg-pre">
                            {JSON.stringify((inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.generation_metadata?.after_configuration, null, 2)}
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Evaluation Metrics */}
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
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Parent Workflow</span>
                    <span className="evo-field-value">
                      <code>{inspectModalVersion.parent_workflow_id || 'Production Baseline'}</code>
                      {inspectModalVersion.parent_version && (
                        <span className="evo-cell-sub"> (v{inspectModalVersion.parent_version.version_number}: {inspectModalVersion.parent_version.name})</span>
                      )}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Strategy</span>
                    <span className="evo-field-value">
                      <strong>{inspectModalVersion.mutation_type?.toUpperCase() || 'BASELINE'}</strong>
                      {inspectModalVersion.refinement_operation ? ` (${inspectModalVersion.refinement_operation})` : ''}
                    </span>
                  </div>
                  {inspectModalVersion.source_candidate_id && (
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Source Candidate / Run</span>
                      <span className="evo-field-value">
                        <code>{inspectModalVersion.source_candidate_id}</code> in <code>{inspectModalVersion.source_run_id}</code>
                      </span>
                    </div>
                  )}
                  {inspectModalVersion.fields_changed && inspectModalVersion.fields_changed.length > 0 && (
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Fields Changed</span>
                      <span className="evo-field-value">{inspectModalVersion.fields_changed.join(', ')}</span>
                    </div>
                  )}
                  {inspectModalVersion.rationale && (
                    <div className="evo-modal-field evo-metric-card-span">
                      <span className="evo-field-key">Rationale</span>
                      <span className="evo-field-value">{inspectModalVersion.rationale}</span>
                    </div>
                  )}
                </div>

                {/* Before vs After Diffs */}
                {inspectModalVersion.before_configuration && inspectModalVersion.after_configuration && (
                  <div className="evo-inspect-diff-grid">
                    <div className="evo-diff-pane">
                      <div className="evo-diff-pane-title">
                        <span>Before (Parent Configuration)</span>
                      </div>
                      <pre className="evo-refine-cfg-pre">
                        {JSON.stringify(inspectModalVersion.before_configuration, null, 2)}
                      </pre>
                    </div>
                    <div className="evo-diff-pane">
                      <div className="evo-diff-pane-title">
                        <span>After (Refined Configuration)</span>
                      </div>
                      <pre className="evo-refine-cfg-pre">
                        {JSON.stringify(inspectModalVersion.after_configuration, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>

              {/* Evaluation Snapshot */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Evaluation Snapshot (At Approval)</h4>
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Groundedness</span>
                    <span className="evo-field-value evo-val-green">
                      {inspectModalVersion.evaluation_snapshot?.groundedness !== null && inspectModalVersion.evaluation_snapshot?.groundedness !== undefined
                        ? `${(inspectModalVersion.evaluation_snapshot.groundedness * 100).toFixed(1)}%`
                        : inspectModalVersion.evaluation_snapshot?.performance !== undefined
                        ? `${(inspectModalVersion.evaluation_snapshot.performance * 100).toFixed(1)}%`
                        : 'N/A'}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Hallucination Rate</span>
                    <span className="evo-field-value evo-val-cyan">
                      {inspectModalVersion.evaluation_snapshot?.hallucination_rate !== null && inspectModalVersion.evaluation_snapshot?.hallucination_rate !== undefined
                        ? `${(inspectModalVersion.evaluation_snapshot.hallucination_rate * 100).toFixed(1)}%`
                        : '0.0%'}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Cost / Query</span>
                    <span className="evo-field-value evo-val-amber">
                      ${Number(inspectModalVersion.evaluation_snapshot?.cost || 0).toFixed(6)}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">LLM Latency</span>
                    <span className="evo-field-value evo-val-purple">
                      {Number(inspectModalVersion.evaluation_snapshot?.llm_latency_ms || 0).toFixed(0)} ms
                    </span>
                  </div>
                </div>
              </div>

              {/* Approval Info */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Governance & Approval Record</h4>
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Approved By</span>
                    <span className="evo-field-value">{inspectModalVersion.approved_by}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Approved At</span>
                    <span className="evo-field-value">{new Date(inspectModalVersion.approved_at).toLocaleString()}</span>
                  </div>
                </div>
              </div>

              {/* Raw JSON */}
              <details className="evo-details-block">
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
