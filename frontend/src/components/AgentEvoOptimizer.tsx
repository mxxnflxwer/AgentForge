import React, { useState, useEffect, useRef } from 'react'

export type MutationType = 'prompt' | 'retrieval' | 'model' | 'operator' | 'mixed'

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
  // Optimization Setup State
  const [activePreset, setActivePreset] = useState<PresetType>('clinical')
  const [query, setQuery] = useState(PRESET_QUERIES.clinical.query)
  const [generatedCandidateCount, setGeneratedCandidateCount] = useState<number>(4)

  // Execution & Pipeline State
  const [running, setRunning] = useState(false)
  const [currentStageIndex, setCurrentStageIndex] = useState<number>(0)
  const [report, setReport] = useState<OptimizationReport | null>(null)
  const [runs, setRuns] = useState<OptimizationReport[]>([])
  const [workflowVersions, setWorkflowVersions] = useState<WorkflowVersion[]>([])
  const [error, setError] = useState<string | null>(null)

  // Interactive Selection State
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null)
  const [inspectModalCandidate, setInspectModalCandidate] = useState<EvaluatedCandidate | null>(null)
  const [inspectModalVersion, setInspectModalVersion] = useState<WorkflowVersion | null>(null)
  const [confirmApprovalCandidate, setConfirmApprovalCandidate] = useState<EvaluatedCandidate | null>(null)
  const [approvingId, setApprovingId] = useState<string | null>(null)
  const [hoveredPoint, setHoveredPoint] = useState<EvaluatedCandidate | null>(null)
  const [isReportCollapsed, setIsReportCollapsed] = useState<boolean>(false)

  // Modal accessibility ref
  const modalRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (isLoggedIn) {
      fetchRuns()
      fetchVersions()
    }
  }, [isLoggedIn])

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
        }
      }
    } catch (err) {
      console.warn('Could not fetch workflow versions', err)
    }
  }

  const handlePresetChange = (preset: PresetType) => {
    setActivePreset(preset)
    if (preset !== 'custom') {
      setQuery(PRESET_QUERIES[preset].query)
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
      } = {
        query: query.trim(),
        candidate_count: generatedCandidateCount,
      }
      if (selectedDocId) {
        payload.document_id = selectedDocId
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
    } catch (err: any) {
      setError(err.message || 'The optimization run could not be completed.')
    } finally {
      clearTimeout(t1)
      clearTimeout(t2)
      setRunning(false)
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
    } catch (err: any) {
      setError(err.message || 'Failed to record developer approval')
    } finally {
      setApprovingId(null)
    }
  }

  // Active selected candidate details
  const activeCandidate = report?.all_candidates.find((c) => c.candidate_id === selectedCandidateId) || report?.baseline
  const baselineCandidate = report?.baseline

  // Dynamic Key Findings calculation
  const generateDynamicFindings = (cand: EvaluatedCandidate, base: EvaluatedCandidate): string[] => {
    const findings: string[] = []
    if (base.metrics.status !== 'success') {
      findings.push(
        `Production baseline evaluation was unavailable (${base.metrics.error_message || base.metrics.status}). Direct metric comparisons against baseline are unavailable.`
      )
      return findings
    }
    if (cand.metrics.status !== 'success') {
      findings.push(
        `Candidate workflow evaluation failed (${cand.metrics.error_message || cand.metrics.status}).`
      )
      return findings
    }

    const candGroundedness = cand.metrics.groundedness ?? cand.metrics.performance
    const baseGroundedness = base.metrics.groundedness ?? base.metrics.performance
    const groundednessDiff = candGroundedness - baseGroundedness
    const costDiff = cand.metrics.cost - base.metrics.cost
    const latencyDiff = cand.metrics.llm_latency_ms - base.metrics.llm_latency_ms

    if (Math.abs(groundednessDiff) < 0.001) {
      if (costDiff < 0) {
        findings.push('The candidate achieved the same measured groundedness as the baseline with lower measured cost.')
      } else if (costDiff > 0) {
        findings.push('The candidate achieved the same measured groundedness as the baseline at a higher measured cost.')
      } else {
        findings.push('The candidate recorded identical groundedness and cost metrics as the baseline.')
      }
    } else if (groundednessDiff > 0) {
      if (costDiff <= 0) {
        findings.push('The candidate achieved higher measured groundedness than the baseline with equal or lower measured cost.')
      } else {
        findings.push('The candidate achieved higher measured groundedness with a measurable cost increase.')
      }
    } else {
      findings.push('The candidate recorded lower measured groundedness compared to the baseline.')
    }

    if (latencyDiff < 0) {
      findings.push('The candidate also recorded lower measured latency in this evaluation run.')
    } else if (latencyDiff > 0) {
      findings.push('The candidate recorded higher measured latency in this evaluation run.')
    }

    return findings
  }

  // Helper to render exploration mutation badge
  const renderMutationBadge = (cand: EvaluatedCandidate) => {
    const meta = cand.mutation_metadata || cand.workflow.mutation_metadata
    if (!meta) return null
    const typeMap: Record<string, { label: string; className: string }> = {
      prompt: { label: 'Prompt', className: 'evo-mut-prompt' },
      retrieval: { label: 'Retrieval', className: 'evo-mut-retrieval' },
      model: { label: 'Model', className: 'evo-mut-model' },
      operator: { label: 'Operator', className: 'evo-mut-operator' },
      mixed: { label: 'Mixed', className: 'evo-mut-mixed' },
    }
    const info = typeMap[meta.mutation_type] || { label: meta.mutation_type, className: 'evo-mut-default' }
    return (
      <span className={`evo-mutation-tag ${info.className}`} title={meta.mutation_description}>
        {info.label}
      </span>
    )
  }

  // Pipeline stages definition
  const pipelineStages = [
    { label: 'Baseline', desc: 'Reference workflow' },
    { label: 'Candidate Generation', desc: 'Mutating parameters' },
    { label: 'Evaluation', desc: 'Phase 6 claim metrics' },
    { label: 'Pareto Analysis', desc: 'Dominance evaluation' },
    { label: 'Human Review', desc: 'Developer inspection' },
  ]

  // Render Scatter Plot
  const renderScatterPlot = () => {
    if (!report || !report.all_candidates || report.all_candidates.length === 0) return null

    const candidates = report.all_candidates
    const maxPerf = 1.0
    const minPerf = 0.0

    const costs = candidates.map((c) => c.metrics.cost)
    const minCost = 0.0
    const maxCost = Math.max(...costs, 0.00005) * 1.25 || 0.0001

    const width = 680
    const height = 280
    const padding = { top: 25, right: 35, bottom: 45, left: 65 }
    const plotWidth = width - padding.left - padding.right
    const plotHeight = height - padding.top - padding.bottom

    const getX = (cost: number) => padding.left + ((cost - minCost) / (maxCost - minCost)) * plotWidth
    const getY = (perf: number) => padding.top + (1 - (perf - minPerf) / (maxPerf - minPerf)) * plotHeight

    // Sorted Pareto points for frontier connection
    const paretoPoints = [...report.pareto_frontier].sort((a, b) => a.metrics.cost - b.metrics.cost)

    return (
      <div className="evo-scatter-container">
        <svg viewBox={`0 0 ${width} ${height}`} className="evo-scatter-svg" role="img" aria-label="Pareto Frontier Scatter Plot">
          {/* Background Grid Lines (Performance / Y-Axis) */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((level, i) => {
            const y = getY(level)
            return (
              <g key={`grid-y-${i}`}>
                <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                <text x={padding.left - 10} y={y + 4} fill="#94a3b8" fontSize="10.5" textAnchor="end" fontFamily="system-ui, sans-serif">
                  {(level * 100).toFixed(0)}%
                </text>
              </g>
            )
          })}

          {/* Cost Ticks (X-Axis) */}
          {[0, 0.33, 0.66, 1.0].map((ratio, i) => {
            const costVal = minCost + ratio * (maxCost - minCost)
            const x = getX(costVal)
            return (
              <g key={`grid-x-${i}`}>
                <line x1={x} y1={padding.top} x2={x} y2={height - padding.bottom} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                <text x={x} y={height - padding.bottom + 18} fill="#94a3b8" fontSize="10" textAnchor="middle" fontFamily="system-ui, sans-serif">
                  ${costVal.toFixed(6)}
                </text>
              </g>
            )
          })}

          {/* Axes Lines */}
          <line x1={padding.left} y1={padding.top} x2={padding.left} y2={height - padding.bottom} stroke="rgba(255,255,255,0.3)" strokeWidth="1.5" />
          <line x1={padding.left} y1={height - padding.bottom} x2={width - padding.right} y2={height - padding.bottom} stroke="rgba(255,255,255,0.3)" strokeWidth="1.5" />

          {/* Axis Labels */}
          <text x={width / 2} y={height - 8} fill="#cbd5e1" fontSize="11" fontWeight="600" textAnchor="middle">
            Cost per Query (USD) → (Lower is better)
          </text>
          <text
            x={-height / 2}
            y={18}
            fill="#cbd5e1"
            fontSize="11"
            fontWeight="600"
            textAnchor="middle"
            transform="rotate(-90)"
          >
            Groundedness / Performance ↑ (Higher is better)
          </text>

          {/* Pareto Frontier Connecting Line */}
          {paretoPoints.length > 1 && (
            <polyline
              points={paretoPoints.map((p) => `${getX(p.metrics.cost)},${getY(p.metrics.groundedness ?? p.metrics.performance)}`).join(' ')}
              fill="none"
              stroke="#22c55e"
              strokeWidth="2"
              strokeDasharray="4 4"
              opacity="0.75"
            />
          )}

          {/* Candidate Points */}
          {candidates.map((c) => {
            const isBaseline = c.candidate_id === 'baseline'
            const isPareto = c.is_pareto_optimal
            const isSelected = selectedCandidateId === c.candidate_id
            const cx = getX(c.metrics.cost)
            const cy = getY(c.metrics.groundedness ?? c.metrics.performance)

            return (
              <g
                key={c.candidate_id}
                className="evo-scatter-point-group"
                style={{ cursor: 'pointer' }}
                onClick={() => setSelectedCandidateId(c.candidate_id)}
                onMouseEnter={() => setHoveredPoint(c)}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                {/* Selection Halo */}
                {isSelected && (
                  <circle cx={cx} cy={cy} r="13" fill="none" stroke="#3b82f6" strokeWidth="2.5" opacity="0.95" />
                )}

                {/* Baseline shape (Diamond) */}
                {isBaseline && (
                  <polygon
                    points={`${cx},${cy - 8} ${cx + 8},${cy} ${cx},${cy + 8} ${cx - 8},${cy}`}
                    fill="#38bdf8"
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                )}

                {/* Pareto-optimal shape (Green solid circle) */}
                {!isBaseline && isPareto && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r="7.5"
                    fill="#22c55e"
                    stroke="#ffffff"
                    strokeWidth="1.5"
                  />
                )}

                {/* Dominated shape (Muted slate with subtle stroke) */}
                {!isBaseline && !isPareto && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r="6"
                    fill="#475569"
                    stroke="#94a3b8"
                    strokeWidth="1.2"
                    opacity="0.85"
                  />
                )}

                {/* Point Label */}
                <text
                  x={cx}
                  y={cy - 11}
                  fill={isPareto ? '#4ade80' : isBaseline ? '#38bdf8' : '#94a3b8'}
                  fontSize="9.5"
                  fontWeight={isPareto || isBaseline ? '700' : '500'}
                  textAnchor="middle"
                >
                  {isBaseline ? 'Baseline' : c.candidate_id}
                </text>
              </g>
            )
          })}
        </svg>

        {/* Dynamic Tooltip on Hover */}
        {hoveredPoint && (
          <div className="evo-scatter-tooltip">
            <div className="evo-tt-header">
              <span className={`evo-tt-badge ${hoveredPoint.candidate_id === 'baseline' ? 'evo-badge-baseline' : hoveredPoint.is_pareto_optimal ? 'evo-badge-pareto' : 'evo-badge-dominated'}`}>
                {hoveredPoint.candidate_id === 'baseline' ? 'Baseline' : hoveredPoint.is_pareto_optimal ? 'Pareto-Optimal' : 'Dominated'}
              </span>
              <strong>{hoveredPoint.workflow.name}</strong>
            </div>
            <div className="evo-tt-body">
              <div><strong>Candidate ID:</strong> {hoveredPoint.candidate_id}</div>
              <div><strong>Model:</strong> {hoveredPoint.workflow.model.provider} / {hoveredPoint.workflow.model.model_id}</div>
              <div><strong>Retrieval:</strong> Top-{hoveredPoint.workflow.retrieval.top_k} (θ={hoveredPoint.workflow.retrieval.similarity_threshold})</div>
              <div><strong>Groundedness:</strong> {((hoveredPoint.metrics.groundedness ?? hoveredPoint.metrics.performance) * 100).toFixed(1)}%</div>
              <div><strong>Hallucination:</strong> {((hoveredPoint.metrics.hallucination_rate ?? 0) * 100).toFixed(1)}%</div>
              <div><strong>Cost / Query:</strong> ${hoveredPoint.metrics.cost.toFixed(6)}</div>
              <div><strong>Latency:</strong> {hoveredPoint.metrics.llm_latency_ms.toFixed(0)} ms</div>
            </div>
          </div>
        )}

        {/* Legend */}
        <div className="evo-scatter-legend">
          <div className="evo-legend-items">
            <div className="evo-legend-item">
              <span className="evo-legend-symbol evo-symbol-baseline">◆</span>
              <span>Baseline</span>
            </div>
            <div className="evo-legend-item">
              <span className="evo-legend-symbol evo-symbol-pareto">●</span>
              <span>Pareto-Optimal</span>
            </div>
            <div className="evo-legend-item">
              <span className="evo-legend-symbol evo-symbol-dominated">○</span>
              <span>Dominated</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="agent-evo-dashboard">
      {/* 1. AGENT EVO HEADER */}
      <header className="evo-header">
        <div className="evo-header-left">
          <h1 className="evo-header-title">AgentEvo</h1>
          <span className="evo-header-subtitle">Autonomous Workflow Optimization</span>
          <p className="evo-header-description">
            Evaluate and evolve RAG and LLM workflows using measurable performance and cost trade-offs.
          </p>
        </div>
        <div className="evo-header-right">
          <div className="evo-badge-stack">
            <span className="evo-phase-badge">PHASE 7.1</span>
            <span className="evo-active-badge">
              <span className="evo-status-dot"></span> ACTIVE
            </span>
          </div>
        </div>
      </header>

      {/* 2. OPTIMIZATION SETUP */}
      <section className="evo-section evo-setup-section" aria-labelledby="evo-setup-title">
        <div className="evo-section-header">
          <h2 id="evo-setup-title" className="evo-section-title">Optimization Setup</h2>
        </div>

        {error && (
          <div className="evo-alert evo-alert-error" role="alert">
            <div className="evo-alert-content">
              <strong>Optimization Notice:</strong> {error}
            </div>
            <button
              type="button"
              onClick={() => setError(null)}
              className="evo-btn-ghost-sm"
              aria-label="Dismiss error"
            >
              Dismiss
            </button>
          </div>
        )}

        <form onSubmit={handleRunOptimization} className="evo-setup-form">
          {/* Exploration Mutation Strategies Bar */}
          <div className="evo-strategy-badges">
            <span className="evo-strategy-title">Mutation Strategies:</span>
            <span className="evo-strategy-pill evo-strat-prompt">Prompt</span>
            <span className="evo-strategy-pill evo-strat-retrieval">Retrieval</span>
            <span className="evo-strategy-pill evo-strat-model">Model</span>
            <span className="evo-strategy-pill evo-strat-operator">Logic / Operator</span>
            <span className="evo-strategy-pill evo-strat-mixed">Mixed</span>
          </div>

          {/* Preset Selector */}
          <div className="evo-form-group">
            <label className="evo-label">Evaluation Preset</label>
            <div className="evo-preset-group" role="tablist">
              {(['clinical', 'medications', 'ood', 'custom'] as PresetType[]).map((presetKey) => (
                <button
                  key={presetKey}
                  type="button"
                  role="tab"
                  aria-selected={activePreset === presetKey}
                  className={`evo-preset-btn ${activePreset === presetKey ? 'evo-preset-btn-active' : ''}`}
                  onClick={() => handlePresetChange(presetKey)}
                >
                  {PRESET_QUERIES[presetKey].label}
                </button>
              ))}
            </div>
          </div>

          {/* Query Input */}
          <div className="evo-form-group">
            <label className="evo-label" htmlFor="evo-query-textarea">Query</label>
            <textarea
              id="evo-query-textarea"
              rows={2}
              value={query}
              onChange={(e) => {
                setQuery(e.target.value)
                setActivePreset('custom')
              }}
              placeholder="Enter benchmark question to optimize against..."
              className="evo-textarea"
              required
            />
          </div>

          {/* Context & Candidate Controls */}
          <div className="evo-form-row">
            <div className="evo-form-group evo-col-half">
              <label className="evo-label" htmlFor="evo-context-select">Target Context</label>
              <select
                id="evo-context-select"
                value={selectedDocId}
                onChange={(e) => setSelectedDocId(e.target.value)}
                className="evo-select"
              >
                <option value="">All Documents (Global Knowledge Base)</option>
                {documents.map((d) => (
                  <option key={d.id} value={d.id}>
                    📄 {d.original_filename || d.filename || d.id}
                  </option>
                ))}
              </select>
            </div>

            <div className="evo-form-group evo-col-half">
              <label className="evo-label">Generated Candidates</label>
              <div className="evo-candidate-selector" role="radiogroup" aria-label="Generated Candidates">
                {[3, 4, 5].map((cnt) => (
                  <button
                    key={cnt}
                    type="button"
                    role="radio"
                    aria-checked={generatedCandidateCount === cnt}
                    className={`evo-candidate-option ${generatedCandidateCount === cnt ? 'evo-candidate-option-active' : ''}`}
                    onClick={() => setGeneratedCandidateCount(cnt)}
                  >
                    [ {cnt} ]
                  </button>
                ))}
              </div>
              <span className="evo-field-caption">
                Baseline is evaluated separately.
              </span>
            </div>
          </div>

          {/* Run Action */}
          <div className="evo-setup-actions">
            <button
              type="submit"
              disabled={running || !isLoggedIn}
              className="evo-btn-primary"
            >
              {running ? 'Running Optimization...' : 'Run AgentEvo Optimization'}
            </button>
          </div>
        </form>
      </section>

      {/* 3. OPTIMIZATION PIPELINE */}
      <section className="evo-section evo-pipeline-section" aria-label="Optimization Pipeline">
        <div className="evo-process-stepper">
          {pipelineStages.map((st, idx) => {
            const isCompleted = report ? true : currentStageIndex > idx
            const isActive = running && currentStageIndex === idx
            return (
              <div
                key={idx}
                className={`evo-process-step ${isCompleted ? 'step-done' : ''} ${isActive ? 'step-running' : ''}`}
              >
                <div className="evo-step-indicator">
                  {isCompleted ? '✓' : isActive ? '●' : '○'}
                </div>
                <div className="evo-step-details">
                  <span className="evo-step-name">{st.label}</span>
                  <span className="evo-step-sub">{st.desc}</span>
                </div>
                {idx < pipelineStages.length - 1 && <span className="evo-step-arrow">→</span>}
              </div>
            )
          })}
        </div>
      </section>

      {/* LOADING STATE */}
      {running && (
        <section className="evo-section evo-loading-section">
          <div className="evo-loading-header">
            <div className="rag-spinner"></div>
            <div>
              <h3 className="evo-loading-title">AgentEvo Optimization</h3>
              <p className="evo-loading-desc">Executing baseline evaluation, candidate generation, Phase 6 metrics calculation, and Pareto dominance analysis...</p>
            </div>
          </div>
        </section>
      )}

      {/* EMPTY STATE */}
      {!report && !running && !error && (
        <section className="evo-section evo-empty-section">
          <div className="evo-empty-icon">⚙️</div>
          <h3 className="evo-empty-title">AgentEvo is ready</h3>
          <p className="evo-empty-desc">
            Configure an evaluation query and generate workflow candidates to begin optimization.
          </p>
          <button
            type="button"
            onClick={() => handleRunOptimization()}
            disabled={!isLoggedIn}
            className="evo-btn-primary"
          >
            Run AgentEvo Optimization
          </button>
        </section>
      )}

      {/* ERROR STATE */}
      {error && !report && !running && (
        <section className="evo-section evo-error-section">
          <div className="evo-empty-icon">⚠</div>
          <h3 className="evo-empty-title">Optimization could not be completed.</h3>
          <p className="evo-empty-desc">No workflow changes were applied.</p>
          <button
            type="button"
            onClick={() => handleRunOptimization()}
            className="evo-btn-secondary"
          >
            Retry
          </button>
        </section>
      )}

      {/* OPTIMIZATION RESULTS */}
      {report && !running && (
        <>
          {/* Runs Switcher if multiple runs exist */}
          {runs.length > 1 && (
            <div className="evo-runs-bar">
              <span className="evo-runs-label">Completed Optimization Runs:</span>
              <div className="evo-runs-chips">
                {runs.map((r) => (
                  <button
                    key={r.run_id}
                    onClick={() => setReport(r)}
                    className={`evo-run-chip ${report?.run_id === r.run_id ? 'evo-run-chip-active' : ''}`}
                  >
                    <span>{r.run_id}</span>
                    <span className="evo-chip-time">
                      {new Date(r.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="evo-chip-badge">{r.pareto_frontier_count} Pareto</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 4. OPTIMIZATION SUMMARY */}
          <section className="evo-section evo-summary-section" aria-labelledby="evo-summary-title">
            <div className="evo-section-header">
              <h2 id="evo-summary-title" className="evo-section-title">Optimization Summary</h2>
              <span className="evo-run-id-pill">Run ID: <code>{report.run_id}</code></span>
            </div>

            <div className="evo-summary-grid">
              <div className="evo-summary-cell">
                <span className="evo-sum-label">Run ID</span>
                <span className="evo-sum-val-code">{report.run_id}</span>
              </div>

              <div className="evo-summary-cell">
                <span className="evo-sum-label">Total Workflows Evaluated</span>
                <span className="evo-sum-val">{report.total_candidates_evaluated}</span>
                <span className="evo-sum-breakdown">
                  1 Baseline + {report.total_candidates_evaluated - 1} Generated candidates
                </span>
              </div>

              <div className="evo-summary-cell">
                <span className="evo-sum-label">Generated Candidates</span>
                <span className="evo-sum-val">{report.total_candidates_evaluated - 1}</span>
                <span className="evo-sum-breakdown">Mutated parameter sets</span>
              </div>

              <div className="evo-summary-cell evo-cell-highlight">
                <span className="evo-sum-label">Pareto-Optimal Workflows</span>
                <span className="evo-sum-val evo-val-pareto">{report.pareto_frontier_count}</span>
                <span className="evo-sum-breakdown">Non-dominated workflows</span>
              </div>

              <div className="evo-summary-cell">
                <span className="evo-sum-label">Governance</span>
                <span className={`evo-gov-tag ${report.approved_workflow_id ? 'gov-tag-approved' : 'gov-tag-pending'}`}>
                  {report.approved_workflow_id ? 'Approved' : 'Pending Review'}
                </span>
                <span className="evo-sum-breakdown">
                  {report.approved_workflow_id ? `Authorized: ${report.approved_workflow_id}` : 'Human review required'}
                </span>
              </div>
            </div>
          </section>

          {/* 5. PARETO ANALYSIS */}
          <section className="evo-section evo-pareto-section" aria-labelledby="evo-pareto-title">
            <div className="evo-section-header">
              <h2 id="evo-pareto-title" className="evo-section-title">Pareto Analysis</h2>
            </div>

            <div className="evo-pareto-layout">
              <div className="evo-pareto-chart-col">
                {renderScatterPlot()}

                {/* Pareto Objective Explanation */}
                <div className="evo-objective-box">
                  <div className="evo-objective-grid">
                    <div className="evo-obj-item">
                      <span className="evo-obj-label">Performance Objective</span>
                      <span className="evo-obj-val evo-obj-up">Groundedness ↑</span>
                    </div>
                    <div className="evo-obj-item">
                      <span className="evo-obj-label">Cost Objective</span>
                      <span className="evo-obj-val evo-obj-down">Cost / Query ↓</span>
                    </div>
                  </div>
                  <p className="evo-objective-note">
                    Pareto optimization currently balances Groundedness against Cost / Query. Latency and token usage are reported as secondary evaluation metrics.
                  </p>
                </div>
              </div>

              {/* 6. SELECTED WORKFLOW PANEL */}
              {activeCandidate && (
                <div className="evo-selected-panel">
                  <div className="evo-selected-header">
                    <div>
                      <span className="evo-panel-title">Selected Workflow</span>
                      <h3 className="evo-selected-name">{activeCandidate.workflow.name}</h3>
                      <span className="evo-selected-id">Candidate ID: <code>{activeCandidate.candidate_id}</code></span>
                    </div>
                    <span className={`evo-status-pill ${activeCandidate.candidate_id === 'baseline' ? 'pill-baseline' : activeCandidate.is_pareto_optimal ? 'pill-pareto' : 'pill-dominated'}`}>
                      {activeCandidate.candidate_id === 'baseline' ? 'Baseline' : activeCandidate.is_pareto_optimal ? 'Pareto-Optimal' : 'Dominated'}
                    </span>
                  </div>

                  <div className="evo-metrics-grid">
                    <div className="evo-metric-card">
                      <span className="evo-metric-label">Groundedness</span>
                      <span className="evo-metric-number evo-num-green">
                        {activeCandidate.metrics.status !== 'success'
                          ? 'Unavailable'
                          : `${((activeCandidate.metrics.groundedness ?? activeCandidate.metrics.performance) * 100).toFixed(1)}%`}
                      </span>
                    </div>

                    <div className="evo-metric-card">
                      <span className="evo-metric-label">Hallucination</span>
                      <span className="evo-metric-number evo-num-cyan">
                        {activeCandidate.metrics.status !== 'success'
                          ? 'Unavailable'
                          : `${((activeCandidate.metrics.hallucination_rate ?? 0) * 100).toFixed(1)}%`}
                      </span>
                    </div>

                    <div className="evo-metric-card">
                      <span className="evo-metric-label">Cost / Query</span>
                      <span className="evo-metric-number evo-num-amber">
                        ${activeCandidate.metrics.cost.toFixed(6)}
                      </span>
                    </div>

                    <div className="evo-metric-card">
                      <span className="evo-metric-label">Total Tokens</span>
                      <span className="evo-metric-number">
                        {activeCandidate.metrics.total_tokens}
                      </span>
                    </div>

                    <div className="evo-metric-card">
                      <span className="evo-metric-label">Latency</span>
                      <span className="evo-metric-number evo-num-purple">
                        {activeCandidate.metrics.llm_latency_ms.toFixed(0)} ms
                      </span>
                    </div>

                    <div className="evo-metric-card">
                      <span className="evo-metric-label">Model</span>
                      <span className="evo-metric-text">
                        <strong>{activeCandidate.workflow.model.provider}</strong>
                        <span className="evo-model-id-sub">{activeCandidate.workflow.model.model_id}</span>
                      </span>
                    </div>

                    <div className="evo-metric-card evo-metric-card-span">
                      <span className="evo-metric-label">Retrieval Configuration</span>
                      <span className="evo-metric-text">
                        Top-K: <strong>{activeCandidate.workflow.retrieval.top_k}</strong> &nbsp;|&nbsp;
                        Threshold: <strong>{activeCandidate.workflow.retrieval.similarity_threshold}</strong>
                      </span>
                    </div>

                    {/* Exploration Mutation Strategy Card */}
                    {(activeCandidate.mutation_metadata || activeCandidate.workflow.mutation_metadata) && (
                      <div className="evo-metric-card evo-metric-card-span evo-mutation-card">
                        <span className="evo-metric-label">Exploration Mutation Strategy</span>
                        <div className="evo-mutation-card-content">
                          {renderMutationBadge(activeCandidate)}
                          <span className="evo-mutation-card-desc">
                            {(activeCandidate.mutation_metadata || activeCandidate.workflow.mutation_metadata)?.mutation_description}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Baseline Comparison (if not baseline) */}
                  {baselineCandidate && activeCandidate.candidate_id !== 'baseline' && (
                    <div className="evo-baseline-comparison">
                      <h4 className="evo-comp-title">Compared with Production Baseline</h4>
                      {baselineCandidate.metrics.status !== 'success' ? (
                        <div className="evo-confirm-notice" style={{ marginTop: '8px' }}>
                          <p>
                            <strong>Baseline Unavailable:</strong> The production baseline evaluation failed (<code>{baselineCandidate.metrics.error_message || baselineCandidate.metrics.status}</code>).
                          </p>
                          <p>
                            Direct percentage comparison deltas cannot be computed against an unavailable baseline.
                          </p>
                        </div>
                      ) : activeCandidate.metrics.status !== 'success' ? (
                        <div className="evo-confirm-notice" style={{ marginTop: '8px' }}>
                          <p>
                            <strong>Candidate Evaluation Failed:</strong> <code>{activeCandidate.metrics.error_message || activeCandidate.metrics.status}</code>.
                          </p>
                        </div>
                      ) : (
                        <div className="evo-comp-grid">
                          <div className="evo-comp-row">
                            <span className="evo-comp-label">Groundedness</span>
                            <span className="evo-comp-values">
                              {((baselineCandidate.metrics.groundedness ?? baselineCandidate.metrics.performance) * 100).toFixed(1)}% → {((activeCandidate.metrics.groundedness ?? activeCandidate.metrics.performance) * 100).toFixed(1)}%
                            </span>
                          </div>

                          <div className="evo-comp-row">
                            <span className="evo-comp-label">Cost</span>
                            <span className="evo-comp-values">
                              ${baselineCandidate.metrics.cost.toFixed(6)} → ${activeCandidate.metrics.cost.toFixed(6)}
                            </span>
                            <span className={`evo-delta-tag ${activeCandidate.metrics.cost <= baselineCandidate.metrics.cost ? 'delta-good' : 'delta-bad'}`}>
                              {baselineCandidate.metrics.cost > 0
                                ? `${((activeCandidate.metrics.cost - baselineCandidate.metrics.cost) / baselineCandidate.metrics.cost * 100).toFixed(1)}%`
                                : '0.0%'}
                            </span>
                          </div>

                          <div className="evo-comp-row">
                            <span className="evo-comp-label">Latency</span>
                            <span className="evo-comp-values">
                              {baselineCandidate.metrics.llm_latency_ms.toFixed(0)} ms → {activeCandidate.metrics.llm_latency_ms.toFixed(0)} ms
                            </span>
                            <span className={`evo-delta-tag ${activeCandidate.metrics.llm_latency_ms <= baselineCandidate.metrics.llm_latency_ms ? 'delta-good' : 'delta-bad'}`}>
                              {baselineCandidate.metrics.llm_latency_ms > 0
                                ? `${((activeCandidate.metrics.llm_latency_ms - baselineCandidate.metrics.llm_latency_ms) / baselineCandidate.metrics.llm_latency_ms * 100).toFixed(1)}%`
                                : '0.0%'}
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Key Findings */}
                      <div className="evo-findings-box">
                        <span className="evo-findings-header">Key Findings:</span>
                        <ul className="evo-findings-list">
                          {generateDynamicFindings(activeCandidate, baselineCandidate).map((finding, fIdx) => (
                            <li key={fIdx}>{finding}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </section>

          {/* 7. CANDIDATE COMPARISON */}
          <section className="evo-section evo-comparison-section" aria-labelledby="evo-comparison-title">
            <div className="evo-section-header">
              <h2 id="evo-comparison-title" className="evo-section-title">Candidate Comparison</h2>
            </div>

            <div className="evo-table-container">
              <table className="evo-table">
                <thead>
                  <tr>
                    <th>Status</th>
                    <th>Workflow</th>
                    <th>Model</th>
                    <th>Groundedness</th>
                    <th>Cost</th>
                    <th>Latency</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {report.all_candidates.map((cand) => {
                    const isBaseline = cand.candidate_id === 'baseline'
                    const isPareto = cand.is_pareto_optimal
                    const isApproved = report.approved_workflow_id === cand.candidate_id
                    const isSelected = selectedCandidateId === cand.candidate_id
                    const isFailed = cand.metrics.status !== 'success'

                    return (
                      <tr
                        key={cand.candidate_id}
                        className={`evo-table-row ${isPareto ? 'row-pareto' : 'row-dominated'} ${isSelected ? 'row-selected' : ''}`}
                        onClick={() => setSelectedCandidateId(cand.candidate_id)}
                      >
                        <td>
                          {isBaseline ? (
                            <span className={`evo-status-tag ${isFailed ? 'tag-dominated' : 'tag-baseline'}`}>
                              {isFailed ? 'Baseline (Error)' : 'Baseline'}
                            </span>
                          ) : isFailed ? (
                            <span className="evo-status-tag tag-dominated">Failed</span>
                          ) : isPareto ? (
                            <span className="evo-status-tag tag-pareto">Pareto</span>
                          ) : (
                            <span className="evo-status-tag tag-dominated">Dominated</span>
                          )}
                        </td>
                        <td>
                          <div className="evo-wf-cell-header">
                            <strong>{cand.workflow.name}</strong>
                            {renderMutationBadge(cand)}
                          </div>
                          <div className="evo-cell-sub">
                            <code>{cand.candidate_id}</code>
                            {(cand.mutation_metadata || cand.workflow.mutation_metadata) && (
                              <span className="evo-mut-desc-sub">
                                {(cand.mutation_metadata || cand.workflow.mutation_metadata)?.mutation_description}
                              </span>
                            )}
                          </div>
                        </td>
                        <td>
                          <div className="evo-model-cell">
                            <span className="evo-provider-name">{cand.workflow.model.provider}</span>
                            <span className="evo-model-id">{cand.workflow.model.model_id}</span>
                          </div>
                        </td>
                        <td>
                          <span className="evo-perf-val">
                            {isFailed
                              ? 'Unavailable'
                              : `${((cand.metrics.groundedness ?? cand.metrics.performance) * 100).toFixed(1)}%`}
                          </span>
                        </td>
                        <td>
                          <span className="evo-cost-val">
                            ${cand.metrics.cost.toFixed(6)}
                          </span>
                        </td>
                        <td>
                          <span className="evo-lat-val">
                            {cand.metrics.llm_latency_ms.toFixed(0)} ms
                          </span>
                        </td>
                        <td>
                          <div className="evo-actions-cell" onClick={(e) => e.stopPropagation()}>
                            <button
                              type="button"
                              onClick={() => setInspectModalCandidate(cand)}
                              className="evo-btn-inspect"
                            >
                              Inspect
                            </button>

                            {isBaseline ? (
                              <span className="evo-current-tag">Current Baseline</span>
                            ) : isPareto ? (
                              <button
                                type="button"
                                disabled={approvingId === cand.candidate_id || isApproved}
                                onClick={() => setConfirmApprovalCandidate(cand)}
                                className={`evo-btn-approve ${isApproved ? 'evo-btn-approved' : ''}`}
                              >
                                {approvingId === cand.candidate_id ? '...' : isApproved ? '✓ Approved' : 'Approve'}
                              </button>
                            ) : null}
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {/* 8. HUMAN GOVERNANCE */}
          <section className="evo-section evo-governance-section" aria-labelledby="evo-gov-title">
            <div className="evo-gov-banner">
              <div className="evo-gov-icon-wrap">🛡️</div>
              <div className="evo-gov-text">
                <h3 id="evo-gov-title" className="evo-gov-title">Human Review Required</h3>
                <p className="evo-gov-description">
                  AgentEvo does not automatically replace the current workflow. A developer must inspect and approve an optimization candidate before it can be applied.
                </p>
              </div>
              <div className="evo-gov-status-wrap">
                <span className={`evo-gov-state-pill ${report.approved_workflow_id ? 'gov-approved' : 'gov-pending'}`}>
                  {report.approved_workflow_id ? 'Approved' : 'Pending Review'}
                </span>
              </div>
            </div>
          </section>

          {/* 9. OPTIMIZATION REPORT */}
          <section className="evo-section evo-report-section" aria-labelledby="evo-report-title">
            <div
              className="evo-section-header evo-collapsible-header"
              onClick={() => setIsReportCollapsed(!isReportCollapsed)}
            >
              <h2 id="evo-report-title" className="evo-section-title">Optimization Report</h2>
              <button
                type="button"
                className="evo-collapse-toggle"
                aria-expanded={!isReportCollapsed}
              >
                {isReportCollapsed ? '▶ Expand' : '▼ Collapse'}
              </button>
            </div>

            {!isReportCollapsed && (
              <div className="evo-report-content">
                <div className="evo-report-dynamic-text">
                  <p>
                    Optimization run <code>{report.run_id}</code> evaluated {report.total_candidates_evaluated} workflows: 1 baseline + {report.total_candidates_evaluated - 1} generated candidates.
                  </p>
                  <p>
                    {report.pareto_frontier_count === 1
                      ? '1 workflow forms the Pareto-optimal frontier.'
                      : `${report.pareto_frontier_count} workflows form the Pareto-optimal frontier.`}
                  </p>
                  <p>
                    Developer review is required before an optimization candidate can be applied.
                  </p>
                </div>

                <div className="evo-report-meta-grid">
                  <div><strong>Run ID:</strong> <code>{report.run_id}</code></div>
                  <div><strong>Baseline:</strong> {report.baseline.workflow.name}</div>
                  <div><strong>Total Workflows Evaluated:</strong> {report.total_candidates_evaluated}</div>
                  <div><strong>Pareto-Optimal Workflows:</strong> {report.pareto_frontier_count}</div>
                  <div><strong>Performance Objective:</strong> Groundedness ↑</div>
                  <div><strong>Cost Objective:</strong> Cost / Query ↓</div>
                </div>
              </div>
            )}
          </section>

          {/* 10. WORKFLOW VERSIONS */}
          <section className="evo-section evo-versions-section" aria-labelledby="evo-versions-title">
            <div className="evo-section-header">
              <div className="evo-header-with-badge">
                <h2 id="evo-versions-title" className="evo-section-title">Approved Workflow Versions</h2>
                <span className="evo-count-badge">{workflowVersions.length} Available</span>
              </div>
              <span className="evo-section-caption">
                Immutable versioned workflow configurations created via explicit developer approval. Baseline remains intact.
              </span>
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
                    {workflowVersions.map((ver) => {
                      const snap = ver.evaluation_snapshot || {}
                      const groundedness = snap.groundedness ?? snap.performance ?? null
                      const cost = snap.cost ?? null
                      const formattedDate = ver.approved_at ? new Date(ver.approved_at).toLocaleString() : 'N/A'

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
                              onClick={() => setInspectModalVersion(ver)}
                              className="evo-btn-inspect"
                            >
                              Inspect Version
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}

      {/* INSPECT WORKFLOW MODAL / DRAWER */}
      {inspectModalCandidate && baselineCandidate && (
        <div
          className="evo-modal-backdrop"
          onClick={() => setInspectModalCandidate(null)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="evo-modal-panel"
            onClick={(e) => e.stopPropagation()}
            ref={modalRef}
          >
            <div className="evo-modal-head">
              <div>
                <div className="evo-modal-badges">
                  <span className={`evo-status-tag ${inspectModalCandidate.candidate_id === 'baseline' ? 'tag-baseline' : inspectModalCandidate.is_pareto_optimal ? 'tag-pareto' : 'tag-dominated'}`}>
                    {inspectModalCandidate.candidate_id === 'baseline' ? 'Baseline' : inspectModalCandidate.is_pareto_optimal ? 'Pareto-Optimal' : 'Dominated'}
                  </span>
                  {renderMutationBadge(inspectModalCandidate)}
                  {report?.approved_workflow_id === inspectModalCandidate.candidate_id && (
                    <span className="evo-status-tag tag-pareto">✓ Approved</span>
                  )}
                </div>
                <h3 className="evo-modal-heading">{inspectModalCandidate.workflow.name}</h3>
                <span className="evo-modal-subheading">Candidate ID: <code>{inspectModalCandidate.candidate_id}</code></span>
              </div>
              <button
                type="button"
                onClick={() => setInspectModalCandidate(null)}
                className="evo-modal-close"
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="evo-modal-content">
              {/* Evolutionary Mutation Details */}
              {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata) && (
                <div className="evo-modal-block">
                  <h4 className="evo-modal-block-title">Evolutionary Exploration Mutation</h4>
                  <div className="evo-modal-grid">
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Mutation Category</span>
                      <span className="evo-field-value">
                        {renderMutationBadge(inspectModalCandidate)}
                      </span>
                    </div>
                    <div className="evo-modal-field">
                      <span className="evo-field-key">Parent Workflow</span>
                      <span className="evo-field-value">
                        <code>{(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.parent_workflow_id}</code>
                      </span>
                    </div>
                    <div className="evo-modal-field evo-metric-card-span">
                      <span className="evo-field-key">Mutation Rationale & Description</span>
                      <span className="evo-field-value">
                        {(inspectModalCandidate.mutation_metadata || inspectModalCandidate.workflow.mutation_metadata)?.mutation_description}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Configuration Section */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Configuration</h4>
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Provider</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.model.provider}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Model ID</span>
                    <span className="evo-field-value"><code>{inspectModalCandidate.workflow.model.model_id}</code></span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Top-K</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.retrieval.top_k}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Similarity Threshold</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.retrieval.similarity_threshold}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Reranking</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.retrieval.enable_rerank ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Prompt Template</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.prompt.template_name}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Chunk Size</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.retrieval.chunk_size} tokens</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Chunk Overlap</span>
                    <span className="evo-field-value">{inspectModalCandidate.workflow.retrieval.chunk_overlap} tokens</span>
                  </div>
                </div>
              </div>

              {/* Evaluation Metrics Section */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Evaluation</h4>
                {inspectModalCandidate.metrics.status !== 'success' && (
                  <div className="evo-confirm-notice" style={{ marginBottom: '10px' }}>
                    <p>
                      <strong>Evaluation Failed:</strong> <code>{inspectModalCandidate.metrics.error_message || inspectModalCandidate.metrics.status}</code>
                    </p>
                  </div>
                )}
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Groundedness</span>
                    <span className="evo-field-value evo-val-green">
                      {inspectModalCandidate.metrics.status !== 'success'
                        ? 'Unavailable'
                        : inspectModalCandidate.metrics.groundedness !== null && inspectModalCandidate.metrics.groundedness !== undefined
                        ? `${(inspectModalCandidate.metrics.groundedness * 100).toFixed(1)}%`
                        : `${(inspectModalCandidate.metrics.performance * 100).toFixed(1)}%`}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Hallucination</span>
                    <span className="evo-field-value evo-val-cyan">
                      {inspectModalCandidate.metrics.status !== 'success'
                        ? 'Unavailable'
                        : inspectModalCandidate.metrics.hallucination_rate !== null && inspectModalCandidate.metrics.hallucination_rate !== undefined
                        ? `${(inspectModalCandidate.metrics.hallucination_rate * 100).toFixed(1)}%`
                        : '0.0%'}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Accuracy</span>
                    <span className="evo-field-value">
                      {inspectModalCandidate.metrics.accuracy !== null && inspectModalCandidate.metrics.accuracy !== undefined
                        ? `${(inspectModalCandidate.metrics.accuracy * 100).toFixed(1)}%`
                        : 'N/A (Reference required)'}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Cost</span>
                    <span className="evo-field-value evo-val-amber">${inspectModalCandidate.metrics.cost.toFixed(6)}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">LLM Latency</span>
                    <span className="evo-field-value evo-val-purple">{inspectModalCandidate.metrics.llm_latency_ms.toFixed(0)} ms</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Execution Time</span>
                    <span className="evo-field-value">{inspectModalCandidate.metrics.execution_time_ms.toFixed(0)} ms</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Input Tokens</span>
                    <span className="evo-field-value">{inspectModalCandidate.metrics.input_tokens}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Output Tokens</span>
                    <span className="evo-field-value">{inspectModalCandidate.metrics.output_tokens}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Total Tokens</span>
                    <span className="evo-field-value">{inspectModalCandidate.metrics.total_tokens}</span>
                  </div>
                </div>
              </div>

              {/* Changes from Baseline Section */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Changes from Baseline</h4>
                <div className="evo-diff-table-wrap">
                  <table className="evo-diff-table">
                    <thead>
                      <tr>
                        <th>Parameter</th>
                        <th>Baseline</th>
                        <th>Candidate</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        {
                          param: 'Model ID',
                          base: baselineCandidate.workflow.model.model_id,
                          cand: inspectModalCandidate.workflow.model.model_id,
                        },
                        {
                          param: 'Provider',
                          base: baselineCandidate.workflow.model.provider,
                          cand: inspectModalCandidate.workflow.model.provider,
                        },
                        {
                          param: 'Top-K Retrieval',
                          base: baselineCandidate.workflow.retrieval.top_k,
                          cand: inspectModalCandidate.workflow.retrieval.top_k,
                        },
                        {
                          param: 'Similarity Threshold',
                          base: baselineCandidate.workflow.retrieval.similarity_threshold,
                          cand: inspectModalCandidate.workflow.retrieval.similarity_threshold,
                        },
                        {
                          param: 'Prompt Template',
                          base: baselineCandidate.workflow.prompt.template_name,
                          cand: inspectModalCandidate.workflow.prompt.template_name,
                        },
                        {
                          param: 'Reranking',
                          base: baselineCandidate.workflow.retrieval.enable_rerank ? 'Enabled' : 'Disabled',
                          cand: inspectModalCandidate.workflow.retrieval.enable_rerank ? 'Enabled' : 'Disabled',
                        },
                      ].map((row, idx) => {
                        const isChanged = String(row.base) !== String(row.cand)
                        return (
                          <tr key={idx} className={isChanged ? 'row-mutated' : ''}>
                            <td><strong>{row.param}</strong></td>
                            <td><code>{String(row.base)}</code></td>
                            <td><code>{String(row.cand)}</code></td>
                            <td>
                              {isChanged ? (
                                <span className="evo-tag-mutated">Mutated</span>
                              ) : (
                                <span className="evo-tag-same">Same</span>
                              )}
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Claim Level Verification if available */}
              {inspectModalCandidate.metrics.raw_details?.claim_breakdown &&
                Array.isArray(inspectModalCandidate.metrics.raw_details.claim_breakdown) &&
                inspectModalCandidate.metrics.raw_details.claim_breakdown.length > 0 && (
                  <details className="evo-details-block">
                    <summary className="evo-details-summary">
                      Claim-Level Verification Details ({inspectModalCandidate.metrics.raw_details.claim_breakdown.length} claims)
                    </summary>
                    <div className="evo-claims-list">
                      {inspectModalCandidate.metrics.raw_details.claim_breakdown.map((claimObj: any, cIdx: number) => {
                        const isSupp = claimObj.is_supported ?? claimObj.supported
                        return (
                          <div key={cIdx} className={`evo-claim-card ${isSupp ? 'claim-supported' : 'claim-unsupported'}`}>
                            <div className="evo-claim-head">
                              <span className="evo-claim-badge">{isSupp ? '✓ Supported' : '⚠ Unsupported'}</span>
                              <span className="evo-claim-text">"{claimObj.claim_text || claimObj.claim}"</span>
                            </div>
                            {claimObj.matched_evidence && (
                              <div className="evo-claim-evidence">
                                <strong>Evidence:</strong> "{claimObj.matched_evidence}"
                              </div>
                            )}
                            {!isSupp && claimObj.reason && (
                              <div className="evo-claim-reason">
                                <strong>Reason:</strong> {claimObj.reason}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </details>
                )}

              {/* Raw JSON Spec */}
              <details className="evo-details-block">
                <summary className="evo-details-summary">Raw Workflow Configuration (JSON)</summary>
                <pre className="evo-json-pre">{JSON.stringify(inspectModalCandidate.workflow, null, 2)}</pre>
              </details>
            </div>

            <div className="evo-modal-foot">
              <button
                type="button"
                onClick={() => setInspectModalCandidate(null)}
                className="evo-btn-secondary"
              >
                Close
              </button>

              {inspectModalCandidate.is_pareto_optimal && inspectModalCandidate.candidate_id !== 'baseline' && (
                <button
                  type="button"
                  disabled={approvingId === inspectModalCandidate.candidate_id || report?.approved_workflow_id === inspectModalCandidate.candidate_id}
                  onClick={() => setConfirmApprovalCandidate(inspectModalCandidate)}
                  className={`evo-btn-primary ${report?.approved_workflow_id === inspectModalCandidate.candidate_id ? 'evo-btn-approved' : ''}`}
                >
                  {report?.approved_workflow_id === inspectModalCandidate.candidate_id ? '✓ Approved' : 'Approve Workflow Candidate'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* APPROVAL CONFIRMATION MODAL */}
      {confirmApprovalCandidate && (
        <div
          className="evo-modal-backdrop"
          onClick={() => setConfirmApprovalCandidate(null)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="evo-confirm-panel"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="evo-confirm-head">
              <span className="evo-confirm-icon">🛡️</span>
              <div>
                <h3 className="evo-confirm-title">Approve Workflow Candidate?</h3>
                <span className="evo-confirm-sub">Explicit Governance & Versioning Action</span>
              </div>
            </div>

            <div className="evo-confirm-body">
              <div className="evo-confirm-meta">
                <div className="evo-confirm-row">
                  <span className="evo-confirm-label">Candidate:</span>
                  <span className="evo-confirm-value"><strong>{confirmApprovalCandidate.workflow.name}</strong> <code>({confirmApprovalCandidate.candidate_id})</code></span>
                </div>
                <div className="evo-confirm-row">
                  <span className="evo-confirm-label">Model:</span>
                  <span className="evo-confirm-value">{confirmApprovalCandidate.workflow.model.provider} / <code>{confirmApprovalCandidate.workflow.model.model_id}</code></span>
                </div>
                <div className="evo-confirm-row">
                  <span className="evo-confirm-label">Groundedness:</span>
                  <span className="evo-confirm-value evo-val-green">
                    {((confirmApprovalCandidate.metrics.groundedness ?? confirmApprovalCandidate.metrics.performance) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="evo-confirm-row">
                  <span className="evo-confirm-label">Cost / Query:</span>
                  <span className="evo-confirm-value evo-val-amber">
                    ${confirmApprovalCandidate.metrics.cost.toFixed(6)}
                  </span>
                </div>
                <div className="evo-confirm-row">
                  <span className="evo-confirm-label">Retrieval:</span>
                  <span className="evo-confirm-value">
                    Top-K: {confirmApprovalCandidate.workflow.retrieval.top_k} | Threshold: {confirmApprovalCandidate.workflow.retrieval.similarity_threshold}
                  </span>
                </div>
              </div>

              <div className="evo-confirm-notice">
                <p>
                  <strong>Governance Policy:</strong> Approving creates a new immutable workflow version for future reference.
                </p>
                <p>
                  The current production baseline will <strong>not</strong> be replaced or automatically deployed.
                </p>
              </div>
            </div>

            <div className="evo-confirm-actions">
              <button
                type="button"
                onClick={() => setConfirmApprovalCandidate(null)}
                className="evo-btn-secondary"
                disabled={approvingId === confirmApprovalCandidate.candidate_id}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleApproveWorkflow(confirmApprovalCandidate.candidate_id)}
                className="evo-btn-primary"
                disabled={approvingId === confirmApprovalCandidate.candidate_id}
              >
                {approvingId === confirmApprovalCandidate.candidate_id ? 'Approving...' : 'Approve Workflow'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* INSPECT WORKFLOW VERSION MODAL */}
      {inspectModalVersion && (
        <div
          className="evo-modal-backdrop"
          onClick={() => setInspectModalVersion(null)}
          role="dialog"
          aria-modal="true"
        >
          <div
            className="evo-modal-panel"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="evo-modal-head">
              <div>
                <div className="evo-modal-badges">
                  <span className="evo-version-tag">v{inspectModalVersion.version_number}</span>
                  <span className="evo-status-tag tag-pareto">✓ {inspectModalVersion.status.toUpperCase()}</span>
                </div>
                <h3 className="evo-modal-heading">{inspectModalVersion.name}</h3>
                <span className="evo-modal-subheading">Version ID: <code>{inspectModalVersion.version_id}</code></span>
              </div>
              <button
                type="button"
                onClick={() => setInspectModalVersion(null)}
                className="evo-modal-close"
                aria-label="Close modal"
              >
                ✕
              </button>
            </div>

            <div className="evo-modal-content">
              {/* Governance & Provenance */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Governance & Provenance</h4>
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Approved By</span>
                    <span className="evo-field-value">{inspectModalVersion.approved_by || 'Developer'}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Approved At</span>
                    <span className="evo-field-value" style={{ fontSize: '12px' }}>
                      {inspectModalVersion.approved_at ? new Date(inspectModalVersion.approved_at).toLocaleString() : 'N/A'}
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Source Candidate</span>
                    <span className="evo-field-value"><code>{inspectModalVersion.source_candidate_id}</code></span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Source Run</span>
                    <span className="evo-field-value"><code>{inspectModalVersion.source_run_id}</code></span>
                  </div>
                </div>
              </div>

              {/* Workflow Configuration */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Workflow Configuration</h4>
                <div className="evo-modal-grid">
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Provider</span>
                    <span className="evo-field-value">{inspectModalVersion.provider}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Model ID</span>
                    <span className="evo-field-value"><code>{inspectModalVersion.model_id}</code></span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Top-K</span>
                    <span className="evo-field-value">{inspectModalVersion.configuration?.retrieval?.top_k ?? 'N/A'}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Similarity Threshold</span>
                    <span className="evo-field-value">{inspectModalVersion.configuration?.retrieval?.similarity_threshold ?? 'N/A'}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Reranking</span>
                    <span className="evo-field-value">{inspectModalVersion.configuration?.retrieval?.enable_rerank ? 'Enabled' : 'Disabled'}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Prompt Template</span>
                    <span className="evo-field-value">{inspectModalVersion.configuration?.prompt?.template_name ?? 'N/A'}</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Chunk Size</span>
                    <span className="evo-field-value">{inspectModalVersion.configuration?.retrieval?.chunk_size ?? 'N/A'} tokens</span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Chunk Overlap</span>
                    <span className="evo-field-value">{inspectModalVersion.configuration?.retrieval?.chunk_overlap ?? 'N/A'} tokens</span>
                  </div>
                </div>
              </div>

              {/* Evaluation Snapshot */}
              <div className="evo-modal-block">
                <h4 className="evo-modal-block-title">Evaluation Snapshot (At Time of Approval)</h4>
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
                    <span className="evo-field-key">Hallucination</span>
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
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Execution Time</span>
                    <span className="evo-field-value">
                      {Number(inspectModalVersion.evaluation_snapshot?.execution_time_ms || 0).toFixed(0)} ms
                    </span>
                  </div>
                  <div className="evo-modal-field">
                    <span className="evo-field-key">Total Tokens</span>
                    <span className="evo-field-value">
                      {inspectModalVersion.evaluation_snapshot?.total_tokens ?? 'N/A'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Raw JSON Spec */}
              <details className="evo-details-block">
                <summary className="evo-details-summary">Raw Configuration Spec (JSON)</summary>
                <pre className="evo-json-pre">{JSON.stringify(inspectModalVersion.configuration, null, 2)}</pre>
              </details>
            </div>

            <div className="evo-modal-foot">
              <button
                type="button"
                onClick={() => setInspectModalVersion(null)}
                className="evo-btn-secondary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
