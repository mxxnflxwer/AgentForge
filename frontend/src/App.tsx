import React, { useState, useEffect, useRef, Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import './App.css'
import { AgentEvoOptimizer } from './components/AgentEvoOptimizer'

interface DocumentItem {
  id: string
  filename?: string
  original_filename?: string
  status?: string
  processing_status?: string
  file_type?: string
  file_size?: number
  specialty?: string | null
  chunk_count?: number
  page_count?: number | null
  processing_error?: string | null
  created_at?: string
  processed_at?: string | null
}

interface RetrievedChunk {
  chunk_id: string
  document_id: string
  chunk_index: number
  content: string
  section_title?: string | null
  distance: number
  similarity_score: number
  relevance_score?: number | null
}

interface QuerySource {
  chunk_id: string
  document_id: string
  chunk_index: number
  section_title?: string | null
  similarity_score?: number
  relevance_score?: number | null
}

interface RAGSearchResponse {
  query: string
  total_results: number
  results: RetrievedChunk[]
}

export interface EvaluationMetrics {
  model: string
  accuracy: number | null
  groundedness: number | null
  hallucination_rate: number | null
  supported_claims: string[]
  unsupported_claims: string[]
  supported_claim_count?: number
  unsupported_claim_count?: number
  total_claims: number
  input_tokens: number | null
  output_tokens: number | null
  total_tokens: number | null
  latency_ms: number | null
  execution_time_ms: number | null
  input_cost: number | null
  output_cost: number | null
  total_cost: number | null
  details?: Record<string, any>
}

interface QueryAnswerResponse {
  status: string
  query: string
  intent: string
  model: string
  model_id?: string
  provider?: string
  success?: boolean
  error?: string | null
  answer: string
  context?: string | null
  sources: QuerySource[]
  latency_ms: number | null
  disclaimer: string
  pipeline_type?: string
  evaluation?: EvaluationMetrics
  execution_id?: string | null
  workflow_version_id?: string | null
  workflow_name?: string | null
  task_category?: string | null
  selection_reason?: string | null
  groundedness?: number | null
  execution_trace?: Record<string, any>
}

interface ModelComparisonResult {
  model: string
  model_id?: string
  provider: string
  answer: string
  latency_ms: number | null
  success: boolean
  error?: string | null
  pipeline_type?: string
  workflow_version_id?: string | null
  workflow_name?: string | null
  context_type?: string
  context_hash?: string | null
  execution_trace?: Record<string, any>
  evaluation?: EvaluationMetrics
}

interface QueryCompareResponse {
  status: string
  query: string
  intent: string
  context?: string | null
  context_hash?: string | null
  workflow_version_id?: string | null
  workflow_name?: string | null
  comparison_mode?: string | null
  sources: QuerySource[]
  results: ModelComparisonResult[]
  disclaimer: string
}

const API_BASE = 'http://localhost:8000'

export interface AvailableModelItem {
  id: string
  model_id: string
  name: string
  provider: string
  badge: string
  is_configured?: boolean
}

const DEFAULT_AVAILABLE_MODELS: AvailableModelItem[] = [
  { id: 'gemini', model_id: 'gemini-3.5-flash-lite', name: 'Gemini 3.5 Flash-Lite', provider: 'Google', badge: 'Fast & Grounded', is_configured: true },
  { id: 'qwen', model_id: 'qwen/qwen3.6-27b', name: 'Qwen 3.6 27B', provider: 'OpenRouter', badge: 'High Accuracy', is_configured: true },
  { id: 'gemma', model_id: 'google/gemma-4-26b-a4b-it:free', name: 'Gemma 4 26B A4B (OpenRouter)', provider: 'OpenRouter', badge: 'Free Open-Weights', is_configured: true },
]

// Formatting helpers for Evaluation telemetry
const formatPercent = (val: number | null | undefined): string => {
  if (val === null || val === undefined) return 'N/A'
  return `${(val * 100).toFixed(1)}%`
}

const formatMs = (ms: number | null | undefined): string => {
  if (ms === null || ms === undefined) return 'N/A'
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(2)}s (${ms.toFixed(0)} ms)`
  }
  return `${ms.toFixed(1)} ms`
}

const formatCost = (cost: number | null | undefined): string => {
  if (cost === null || cost === undefined) return 'N/A'
  if (cost === 0) return '$0.00 (Free)'
  if (cost < 0.0001) return `$${cost.toFixed(6)}`
  return `$${cost.toFixed(4)}`
}

const formatTokens = (tokens: number | null | undefined): string => {
  if (tokens === null || tokens === undefined) return 'N/A'
  return tokens.toLocaleString()
}

// Safe helper getters
const getDocFilename = (doc?: DocumentItem | null): string => {
  if (!doc) return 'Document'
  return doc.filename || doc.original_filename || `Document-${doc.id?.slice(0, 8) || 'unknown'}`
}

const getDocStatus = (doc?: DocumentItem | null): string => {
  if (!doc) return 'COMPLETED'
  return String(doc.status || doc.processing_status || 'COMPLETED')
}

// React Error Boundary
interface ErrorBoundaryProps {
  children: ReactNode
}
interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo)
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rag-app-container">
          <div className="rag-alert rag-alert-error" style={{ margin: '40px auto', maxWidth: 600 }}>
            <h2>⚠️ Application Render Notice</h2>
            <p>{this.state.error?.message || 'A render error occurred.'}</p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null })
                window.location.reload()
              }}
              className="rag-btn rag-btn-primary"
              style={{ marginTop: 12 }}
            >
              Reload Interface
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

function MainApp() {
  // Auth state
  const [userEmail, setUserEmail] = useState('demo@agentforge.dev')
  const [password, setPassword] = useState('StrongPassword123!')
  const [userName, setUserName] = useState('Clinical Evaluator')
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [currentUser, setCurrentUser] = useState<string | null>(null)
  const [userRole, setUserRole] = useState<string | null>(null)

  // Document state
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [loadingDocs, setLoadingDocs] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadSpecialty, setUploadSpecialty] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Active Main Tab: 'llm' | 'vector' | 'optimizer'
  const [activeTab, setActiveTab] = useState<'llm' | 'vector' | 'optimizer'>('llm')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  // Query & Model settings
  const [availableModels, setAvailableModels] = useState(DEFAULT_AVAILABLE_MODELS)
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [selectedModel, setSelectedModel] = useState<string>('gemini')
  const [queryMode, setQueryMode] = useState<'single' | 'compare'>('single')
  const [compareMode, setCompareMode] = useState<'active_workflow' | 'baseline'>('active_workflow')
  const [query, setQuery] = useState('What is the clinical diagnosis?')
  const [expectedAnswer, setExpectedAnswer] = useState('')
  const [topK, setTopK] = useState<number>(5)

  // Execution states
  const [generating, setGenerating] = useState(false)
  const [singleAnswer, setSingleAnswer] = useState<QueryAnswerResponse | null>(() => {
    try {
      const saved = sessionStorage.getItem('agentforge_single_answer')
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  })
  const [compareResults, setCompareResults] = useState<QueryCompareResponse | null>(() => {
    try {
      const saved = sessionStorage.getItem('agentforge_compare_results')
      return saved ? JSON.parse(saved) : null
    } catch {
      return null
    }
  })

  // Synchronize execution results with sessionStorage so they survive page refresh
  useEffect(() => {
    try {
      if (singleAnswer) {
        sessionStorage.setItem('agentforge_single_answer', JSON.stringify(singleAnswer))
      } else {
        sessionStorage.removeItem('agentforge_single_answer')
      }
    } catch (e) {
      console.warn('Could not save singleAnswer to sessionStorage', e)
    }
  }, [singleAnswer])

  useEffect(() => {
    try {
      if (compareResults) {
        sessionStorage.setItem('agentforge_compare_results', JSON.stringify(compareResults))
      } else {
        sessionStorage.removeItem('agentforge_compare_results')
      }
    } catch (e) {
      console.warn('Could not save compareResults to sessionStorage', e)
    }
  }, [compareResults])

  // Vector Search state
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<RAGSearchResponse | null>(null)

  // Feedback / Error state
  const [authError, setAuthError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)

  useEffect(() => {
    checkCurrentUser()
    fetchModels()
  }, [])

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/query/models`)
      if (res.ok) {
        const data = await res.json()
        if (data && Array.isArray(data.models) && data.models.length > 0) {
          setAvailableModels(
            data.models.map((m: any) => {
              const lowerName = (m.name || '').toLowerCase()
              const lowerId = (m.model_id || '').toLowerCase()
              let id = 'gemini'
              let badge = 'Fast & Grounded'
              if (lowerName.includes('qwen') || lowerId.includes('qwen')) {
                id = 'qwen'
                badge = 'High Accuracy'
              } else if (
                lowerName.includes('gemma') ||
                lowerId.includes('gemma') ||
                lowerName.includes('gpt') ||
                lowerId.includes('gpt')
              ) {
                id = 'gemma'
                badge = 'Free Open-Weights'
              }
              return {
                id,
                model_id: m.model_id,
                name: m.name,
                provider: m.provider,
                badge,
                is_configured: m.is_configured ?? true,
              }
            })
          )
        }
      }
    } catch (err) {
      console.warn('Could not fetch backend model metadata:', err)
    }
  }

  const checkCurrentUser = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/me`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        setIsLoggedIn(true)
        setCurrentUser(data?.name ? `${data.name} (${data.email})` : data?.email || 'Authenticated User')
        setUserRole(data?.role || 'DEVELOPER')
        fetchDocuments()
      } else {
        setIsLoggedIn(false)
        setUserRole(null)
      }
    } catch (err) {
      console.warn('Backend connection notice:', err)
    }
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError(null)
    try {
      if (authMode === 'register') {
        const regRes = await fetch(`${API_BASE}/api/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ email: userEmail, password, name: userName || 'Developer User' }),
        })
        if (!regRes.ok) {
          const errData = await regRes.json().catch(() => ({}))
          throw new Error(errData.detail || 'Registration failed')
        }
      }

      let res = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email: userEmail, password }),
      })

      if (!res.ok && authMode === 'login' && userEmail === 'demo@agentforge.dev') {
        // Fallback convenience registration for demo account
        await fetch(`${API_BASE}/api/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ email: userEmail, password, name: 'Demo Developer' }),
        })
        res = await fetch(`${API_BASE}/api/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ email: userEmail, password }),
        })
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || 'Authentication failed. Please check credentials.')
      }

      const userData = await res.json()
      setIsLoggedIn(true)
      setCurrentUser(userData?.name ? `${userData.name} (${userData.email})` : userData.email)
      setUserRole(userData?.role || 'DEVELOPER')
      fetchDocuments()
    } catch (err: any) {
      setAuthError(err.message || 'Authentication error')
    }
  }

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      })
    } catch {}
    setIsLoggedIn(false)
    setCurrentUser(null)
    setUserRole(null)
    setDocuments([])
    setSingleAnswer(null)
    setCompareResults(null)
    setSearchResults(null)
  }

  const fetchDocuments = async () => {
    setLoadingDocs(true)
    try {
      const res = await fetch(`${API_BASE}/api/documents`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json().catch(() => [])
        if (Array.isArray(data)) {
          setDocuments(data)
        }
      }
    } catch (err) {
      console.warn('Failed to fetch documents', err)
    } finally {
      setLoadingDocs(false)
    }
  }

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedFile) {
      setUploadError('Please select a file to upload (.pdf, .docx, .txt).')
      return
    }

    setUploadError(null)
    setUploadSuccess(null)
    setUploading(true)

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)
      if (uploadSpecialty.trim()) {
        formData.append('specialty', uploadSpecialty.trim())
      }

      const res = await fetch(`${API_BASE}/api/documents`, {
        method: 'POST',
        credentials: 'include',
        body: formData,
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Upload failed with status ${res.status}`)
      }

      const uploadResult = await res.json().catch(() => ({}))
      const uploadedName = uploadResult.filename || uploadResult.original_filename || selectedFile.name
      setUploadSuccess(`Successfully uploaded "${uploadedName}"! Indexing pipeline active.`)
      setSelectedFile(null)
      setUploadSpecialty('')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      await fetchDocuments()
      setTimeout(() => fetchDocuments(), 2000)
      setTimeout(() => fetchDocuments(), 4000)
    } catch (err: any) {
      setUploadError(err.message || 'Error uploading document')
    } finally {
      setUploading(false)
    }
  }

  // Handle LLM Query / Comparison
  const handleLLMQuery = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) {
      setQueryError('Please enter a clinical or document question.')
      return
    }

    setQueryError(null)
    setGenerating(true)
    setSingleAnswer(null)
    setCompareResults(null)

    try {
      if (queryMode === 'single') {
        const payload: {
          query: string
          document_id?: string
          model: string
          top_k: number
          expected_answer?: string
        } = {
          query: query.trim(),
          model: selectedModel,
          top_k: topK,
        }
        if (selectedDocId) {
          payload.document_id = selectedDocId
        }
        if (expectedAnswer.trim()) {
          payload.expected_answer = expectedAnswer.trim()
        }

        const res = await fetch(`${API_BASE}/api/query/answer`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(payload),
        })

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}))
          throw new Error(errData.detail || `Generation failed with status ${res.status}`)
        }

        const data: QueryAnswerResponse = await res.json()
        setSingleAnswer(data)
      } else {
        // Multi-model comparison
        const payload: {
          query: string
          document_id?: string
          top_k: number
          expected_answer?: string
          comparison_mode?: string
        } = {
          query: query.trim(),
          top_k: topK,
          comparison_mode: compareMode,
        }
        if (selectedDocId) {
          payload.document_id = selectedDocId
        }
        if (expectedAnswer.trim()) {
          payload.expected_answer = expectedAnswer.trim()
        }

        const res = await fetch(`${API_BASE}/api/query/compare`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify(payload),
        })

        if (!res.ok) {
          const errData = await res.json().catch(() => ({}))
          throw new Error(errData.detail || `Comparison failed with status ${res.status}`)
        }

        const data: QueryCompareResponse = await res.json()
        setCompareResults(data)
      }
    } catch (err: any) {
      setQueryError(err.message || 'Error processing LLM request')
    } finally {
      setGenerating(false)
    }
  }

  // Handle Raw Vector Retrieval
  const handleVectorSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) {
      setQueryError('Please enter a search query.')
      return
    }

    setQueryError(null)
    setSearching(true)

    try {
      const payload: { query: string; document_id?: string; top_k: number } = {
        query: query.trim(),
        top_k: topK,
      }
      if (selectedDocId) {
        payload.document_id = selectedDocId
      }

      const res = await fetch(`${API_BASE}/api/rag/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}))
        throw new Error(errData.detail || `Search failed with status ${res.status}`)
      }

      const data: RAGSearchResponse = await res.json()
      setSearchResults(data)
    } catch (err: any) {
      setQueryError(err.message || 'Error performing vector search')
      setSearchResults(null)
    } finally {
      setSearching(false)
    }
  }

  const getDocNameById = (docId: string): string => {
    const doc = documents.find((d) => d.id === docId)
    return getDocFilename(doc)
  }

  return (
    <div className="rag-app-container">
      {/* Header */}
      <header className="rag-header">
        <div className="rag-header-left">
          <div className="rag-logo-badge">⚡ AgentForge</div>
          <h1>Medical Document Analyzer & Model Comparison</h1>
          <p className="rag-subtitle">
            RAG Ingestion Pipeline ➔ 3-LLM Integration (Gemini 3.5 Flash-Lite, Qwen 3.6 27B, Gemma 4 26B A4B) ➔ AgentEvo Foundation
          </p>
        </div>
        <div className="rag-header-right">
          <span className="rag-tag">Gemini 3.5 Flash-Lite</span>
          <span className="rag-tag">Qwen 3.6 27B</span>
          <span className="rag-tag">Gemma 4 26B A4B (OpenRouter)</span>
          {isLoggedIn ? (
            <div className="rag-user-controls">
              <span className="rag-user-badge">🟢 {currentUser || 'Authenticated'}</span>
              {userRole && (
                <span
                  className={`rag-role-badge ${userRole === 'DEVELOPER' ? 'role-dev' : 'role-user'}`}
                  style={{
                    fontSize: '0.72rem',
                    padding: '2px 8px',
                    borderRadius: 4,
                    fontWeight: 700,
                    background: userRole === 'DEVELOPER' ? '#ede9fe' : '#e0f2fe',
                    color: userRole === 'DEVELOPER' ? '#6d28d9' : '#0369a1',
                    border: '1px solid currentColor',
                  }}
                >
                  {userRole}
                </span>
              )}
              <button onClick={handleLogout} className="rag-btn-secondary rag-btn-sm">
                Logout
              </button>
            </div>
          ) : (
            <span className="rag-user-badge-off">Not Authenticated</span>
          )}
          <button
            type="button"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="rag-btn-secondary rag-btn-sm"
            title="Toggle Documents Ingestion Sidebar"
          >
            {sidebarCollapsed ? 'Show Ingestion' : 'Hide Ingestion'}
          </button>
        </div>
      </header>

      {/* Auth Card */}
      {!isLoggedIn && (
        <div className="rag-card rag-auth-card">
          <div className="rag-card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h2>{authMode === 'login' ? 'User & Developer Sign In' : 'Create New Account'}</h2>
            <div className="rag-auth-tabs" style={{ display: 'flex', gap: 6 }}>
              <button
                type="button"
                className={`rag-btn-sm ${authMode === 'login' ? 'rag-btn-primary' : 'rag-btn-secondary'}`}
                onClick={() => { setAuthMode('login'); setAuthError(null); }}
              >
                Sign In
              </button>
              <button
                type="button"
                className={`rag-btn-sm ${authMode === 'register' ? 'rag-btn-primary' : 'rag-btn-secondary'}`}
                onClick={() => { setAuthMode('register'); setAuthError(null); }}
              >
                Register
              </button>
            </div>
          </div>
          <p className="rag-card-desc">
            {authMode === 'login'
              ? 'Connect session with authenticated credentials to execute queries, upload documents, and manage evolutionary optimization.'
              : 'Create an authorized user account for clinical document evaluation.'}
          </p>
          {authError && <div className="rag-alert rag-alert-error">⚠️ {authError}</div>}
          <form onSubmit={handleLogin} className="rag-auth-form">
            {authMode === 'register' && (
              <input
                type="text"
                placeholder="Full Name"
                value={userName}
                onChange={(e) => setUserName(e.target.value)}
                required
              />
            )}
            <input
              type="email"
              placeholder="Email address"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
              required
            />
            <input
              type="password"
              placeholder="Password (min 8 chars)"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button type="submit" className="rag-btn rag-btn-primary">
              {authMode === 'login' ? 'Sign In' : 'Register & Connect'}
            </button>
          </form>
        </div>
      )}

      {/* Main Grid */}
      <div className={`rag-grid-main ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
        {/* Left Column: Documents & Ingestion */}
        {!sidebarCollapsed && (
          <div className="rag-col-left">
            {/* Ingest Card */}
            <div className="rag-card">
              <div className="rag-card-header">
                <h2>Ingest Document</h2>
              </div>
              <p className="rag-card-desc">Upload clinical records (.pdf, .docx, .txt) for chunking and BGE-M3 vector indexing.</p>

              {uploadError && <div className="rag-alert rag-alert-error">⚠️ {uploadError}</div>}
              {uploadSuccess && <div className="rag-alert rag-alert-success">✓ {uploadSuccess}</div>}

              <form onSubmit={handleFileUpload} className="rag-upload-form">
                <div className="rag-form-group">
                  <label>Select File:</label>
                  <input
                    type="file"
                    ref={fileInputRef}
                    accept=".pdf,.docx,.txt"
                    onChange={(e) => setSelectedFile(e.target.files ? e.target.files[0] : null)}
                    className="rag-file-input"
                    required
                  />
                </div>

                <div className="rag-form-group">
                  <label>Specialty / Tag (Optional):</label>
                  <input
                    type="text"
                    placeholder="e.g. Cardiology, Oncology, General"
                    value={uploadSpecialty}
                    onChange={(e) => setUploadSpecialty(e.target.value)}
                    className="rag-text-input"
                  />
                </div>

                <button
                  type="submit"
                  disabled={uploading || !isLoggedIn}
                  className="rag-btn rag-btn-primary rag-btn-block"
                >
                  {uploading ? 'Uploading & Indexing...' : 'Upload & Index Document'}
                </button>
              </form>
            </div>

            {/* Ingested Documents Card */}
            <div className="rag-card">
              <div className="rag-card-header">
                <h2>Ingested Documents ({documents.length})</h2>
                <button
                  onClick={fetchDocuments}
                  className="rag-btn-secondary rag-btn-sm"
                  disabled={loadingDocs || !isLoggedIn}
                >
                  {loadingDocs ? '...' : 'Refresh'}
                </button>
              </div>

              {documents.length === 0 ? (
                <div className="rag-placeholder-sm">
                  <p>No documents indexed yet. Upload a sample document to start.</p>
                </div>
              ) : (
                <div className="rag-doc-list">
                  {documents.map((doc) => {
                    const statusStr = getDocStatus(doc)
                    const isSelected = selectedDocId === doc.id
                    const fileSizeKb = doc.file_size ? (doc.file_size / 1024).toFixed(1) : '0'

                    return (
                      <div
                        key={doc.id}
                        className={`rag-doc-item ${isSelected ? 'rag-doc-selected' : ''}`}
                        onClick={() => setSelectedDocId(isSelected ? '' : doc.id)}
                      >
                        <div className="rag-doc-item-header">
                          <span className="rag-doc-title">{getDocFilename(doc)}</span>
                          <span className={`rag-status-badge rag-status-${statusStr.toLowerCase()}`}>
                            {statusStr}
                          </span>
                        </div>
                        <div className="rag-doc-item-meta">
                          <span>{fileSizeKb} KB</span>
                          <span>•</span>
                          <span>{doc.chunk_count ?? 0} Chunks</span>
                          {doc.specialty && (
                            <>
                              <span>•</span>
                              <span className="rag-doc-tag">{doc.specialty}</span>
                            </>
                          )}
                        </div>
                        {doc.processing_error && (
                          <div className="rag-doc-error">Error: {doc.processing_error}</div>
                        )}
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Right Column: LLM Generation & Multi-Model Comparison */}
        <div className="rag-col-right">
          {/* Main Mode Navigation Bar */}
          <div className="rag-tab-nav">
            <button
              className={`rag-tab-btn ${activeTab === 'llm' ? 'rag-tab-active' : ''}`}
              onClick={() => setActiveTab('llm')}
            >
              LLM Analysis & Comparison
            </button>
            <button
              className={`rag-tab-btn ${activeTab === 'vector' ? 'rag-tab-active' : ''}`}
              onClick={() => setActiveTab('vector')}
            >
              Vector Search (RAG Chunks)
            </button>
            <button
              className={`rag-tab-btn ${activeTab === 'optimizer' ? 'rag-tab-active' : ''}`}
              onClick={() => setActiveTab('optimizer')}
            >
              AgentEvo Optimizer
            </button>
          </div>

          {activeTab === 'llm' && (
            <>
              {/* Query & Model Selection Card */}
              <div className="rag-card">
                <div className="rag-card-header">
                  <h2>🤖 Ask Grounded Document Questions</h2>
                  {selectedDocId && (
                    <span className="rag-filter-indicator">
                      Scoped: {getDocNameById(selectedDocId)} (
                      <button onClick={() => setSelectedDocId('')} className="rag-link-btn">
                        Clear
                      </button>
                      )
                    </span>
                  )}
                </div>

                {queryError && <div className="rag-alert rag-alert-error">⚠️ {queryError}</div>}

                <form onSubmit={handleLLMQuery} className="rag-search-form">
                  {/* Execution Mode Switcher */}
                  <div className="rag-mode-switcher">
                    <label
                      className={`rag-mode-opt ${queryMode === 'single' ? 'rag-mode-opt-active' : ''}`}
                      onClick={() => setQueryMode('single')}
                    >
                      <input
                        type="radio"
                        name="queryMode"
                        checked={queryMode === 'single'}
                        onChange={() => setQueryMode('single')}
                      />
                      <span>Single Model Answering</span>
                    </label>

                    <label
                      className={`rag-mode-opt ${queryMode === 'compare' ? 'rag-mode-opt-active' : ''}`}
                      onClick={() => setQueryMode('compare')}
                    >
                      <input
                        type="radio"
                        name="queryMode"
                        checked={queryMode === 'compare'}
                        onChange={() => setQueryMode('compare')}
                      />
                      <span>⚖️ Compare All 3 Models Concurrently</span>
                    </label>
                  </div>

                  {/* Comparison Workflow Selector (Visible in Compare Mode) */}
                  {queryMode === 'compare' && (
                    <div className="rag-form-group">
                      <label className="rag-label-with-hint">
                        <span>Comparison Execution Workflow:</span>
                        <span className="rag-input-hint">Select workflow configuration applied for shared retrieval and prompt generation</span>
                      </label>
                      <div className="rag-mode-switcher" style={{ marginTop: '4px' }}>
                        <label
                          className={`rag-mode-opt ${compareMode === 'active_workflow' ? 'rag-mode-opt-active' : ''}`}
                          onClick={() => {
                            setCompareMode('active_workflow')
                            setCompareResults(null)
                          }}
                        >
                          <input
                            type="radio"
                            name="compareMode"
                            checked={compareMode === 'active_workflow'}
                            onChange={() => {
                              setCompareMode('active_workflow')
                              setCompareResults(null)
                            }}
                          />
                          <span>🚀 Active AgentEvo Workflow (Optimized RAG & Prompt)</span>
                        </label>

                        <label
                          className={`rag-mode-opt ${compareMode === 'baseline' ? 'rag-mode-opt-active' : ''}`}
                          onClick={() => {
                            setCompareMode('baseline')
                            setCompareResults(null)
                          }}
                        >
                          <input
                            type="radio"
                            name="compareMode"
                            checked={compareMode === 'baseline'}
                            onChange={() => {
                              setCompareMode('baseline')
                              setCompareResults(null)
                            }}
                          />
                          <span>⚖️ Multi-LLM Comparative Baseline (Default RAG)</span>
                        </label>
                      </div>
                    </div>
                  )}

                  {/* Model Selector (Visible in Single Mode) */}
                  {queryMode === 'single' && (
                    <div className="rag-form-group">
                      <label>Select LLM Engine:</label>
                      <div className="rag-model-grid">
                        {availableModels.map((m) => (
                          <div
                            key={m.id}
                            className={`rag-model-card ${selectedModel === m.id ? 'rag-model-selected' : ''}`}
                            onClick={() => {
                              if (selectedModel !== m.id) {
                                setSelectedModel(m.id)
                                setSingleAnswer(null)
                                setQueryError(null)
                              }
                            }}
                          >
                            <div className="rag-model-header">
                              <span className="rag-model-name">{m.name}</span>
                              <span className="rag-model-badge">{m.badge}</span>
                            </div>
                            <span className="rag-model-provider">Provider: {m.provider}</span>
                            <span className="rag-model-id-sub"><code>{m.model_id}</code></span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Document Scope Selection */}
                  <div className="rag-form-group">
                    <label>Document Scope:</label>
                    <select
                      value={selectedDocId}
                      onChange={(e) => {
                        setSelectedDocId(e.target.value)
                        setSingleAnswer(null)
                        setCompareResults(null)
                        setQueryError(null)
                      }}
                      className="rag-select"
                    >
                      <option value="">🌐 All Uploaded Documents</option>
                      {documents.map((doc) => (
                        <option key={doc.id} value={doc.id}>
                          📄 {getDocFilename(doc)} ({doc.chunk_count ?? 0} chunks)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Question Textarea */}
                  <div className="rag-form-group">
                    <label>Question:</label>
                    <textarea
                      rows={3}
                      placeholder="e.g. What is the clinical diagnosis? What medications are prescribed?"
                      value={query}
                      onChange={(e) => {
                        setQuery(e.target.value)
                        if (singleAnswer) setSingleAnswer(null)
                        if (compareResults) setCompareResults(null)
                        if (queryError) setQueryError(null)
                      }}
                      className="rag-textarea"
                      required
                    />

                    {/* Quick Suggestion Chips */}
                    <div className="rag-query-chips">
                      <button
                        type="button"
                        className="rag-chip"
                        onClick={() => {
                          setQuery('What is the clinical diagnosis?')
                          setSingleAnswer(null)
                          setCompareResults(null)
                          setQueryError(null)
                        }}
                      >
                        💡 Clinical Diagnosis
                      </button>
                      <button
                        type="button"
                        className="rag-chip"
                        onClick={() => {
                          setQuery('What medications are prescribed and what is the treatment plan?')
                          setSingleAnswer(null)
                          setCompareResults(null)
                          setQueryError(null)
                        }}
                      >
                        💊 Medications & Plan
                      </button>
                      <button
                        type="button"
                        className="rag-chip"
                        onClick={() => {
                          setQuery('What is the patient\'s phone number?')
                          setSingleAnswer(null)
                          setCompareResults(null)
                          setQueryError(null)
                        }}
                      >
                        🛡️ Phone Number (Out-of-Domain Test)
                      </button>
                    </div>
                    {/* Optional Reference Answer for Accuracy Metric */}
                    <div className="rag-form-group">
                      <label className="rag-label-with-hint">
                        <span>Expected / Reference Answer:</span>
                        <span className="rag-input-hint">(Optional — calculates accuracy metric)</span>
                      </label>
                      <input
                        type="text"
                        placeholder="e.g. Stable angina pectoris (leave empty if evaluating open-ended retrieval)"
                        value={expectedAnswer}
                        onChange={(e) => {
                          setExpectedAnswer(e.target.value)
                          if (singleAnswer) setSingleAnswer(null)
                          if (compareResults) setCompareResults(null)
                        }}
                        className="rag-text-input"
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={generating || !isLoggedIn}
                    className="rag-btn rag-btn-primary rag-btn-block rag-btn-lg"
                  >
                    {generating
                      ? (queryMode === 'compare' ? '⚡ Querying 3 Models Concurrently & Evaluating...' : `⚡ Generating with ${availableModels.find((m) => m.id === selectedModel)?.name || 'Model'} & Evaluating...`)
                      : (queryMode === 'compare' ? '⚖️ Run 3-Model Comparative Evaluation' : '✨ Generate Grounded Answer')}
                  </button>
                </form>
              </div>

              {/* Single Model Answer Output */}
              {queryMode === 'single' && (
                <div className="rag-card rag-results-panel">
                  <div className="rag-card-header">
                    <h2>🎯 Generated Grounded Answer & Evaluation</h2>
                    {singleAnswer && (
                      <div className="rag-header-metrics">
                        {singleAnswer.latency_ms !== null && singleAnswer.latency_ms !== undefined && (
                          <span className="rag-metric-tag">⚡ {formatMs(singleAnswer.latency_ms)}</span>
                        )}
                        <span className={`rag-status-chip ${
                          singleAnswer.status === 'insufficient_credits'
                            ? 'rag-status-insufficient_credits'
                            : (singleAnswer.error && (singleAnswer.error.toLowerCase().includes('429') || singleAnswer.error.toLowerCase().includes('rate limit')))
                            ? 'rag-status-warning'
                            : (singleAnswer.error && (singleAnswer.error.toLowerCase().includes('401') || singleAnswer.error.toLowerCase().includes('unauthorized')))
                            ? 'rag-status-error'
                            : singleAnswer.status === 'error'
                            ? 'rag-status-error'
                            : singleAnswer.status === 'not_found'
                            ? 'rag-status-not_found'
                            : singleAnswer.success === false
                            ? 'rag-status-error'
                            : 'rag-status-success'
                        }`}>
                          {singleAnswer.status === 'insufficient_credits'
                            ? '💳 Insufficient Provider Credits'
                            : (singleAnswer.error && (singleAnswer.error.toLowerCase().includes('429') || singleAnswer.error.toLowerCase().includes('rate limit')))
                            ? '⏳ Rate Limited (HTTP 429)'
                            : (singleAnswer.error && (singleAnswer.error.toLowerCase().includes('401') || singleAnswer.error.toLowerCase().includes('unauthorized')))
                            ? '🔑 Auth Error (HTTP 401)'
                            : singleAnswer.status === 'error'
                            ? '❌ Provider Error'
                            : singleAnswer.status === 'not_found'
                            ? '⚠️ Not Found in Document'
                            : singleAnswer.success === false
                            ? '❌ Generation Failed'
                            : '🟢 Grounded'}
                        </span>
                      </div>
                    )}
                  </div>

                  {generating && (
                    <div className="rag-placeholder">
                      <div className="rag-spinner"></div>
                      <p>Retrieving context via BGE-M3 + Reranker, generating answer, and running AgentEvo Evaluation Engine...</p>
                    </div>
                  )}

                  {!generating && !singleAnswer && (
                    <div className="rag-placeholder">
                      <span className="rag-placeholder-icon">💬</span>
                      <p>Ask a question above and click <strong>Generate Grounded Answer</strong>.</p>
                    </div>
                  )}

                  {!generating && singleAnswer && (
                    <div className="rag-answer-container">
                      <div className="rag-answer-box">
                        <div className="rag-answer-model-tag">
                          <span>🤖 {singleAnswer.model} {singleAnswer.provider ? `(${singleAnswer.provider})` : ''}</span>
                          {singleAnswer.model_id && (
                            <span className="rag-model-id-badge" title="Model Identifier"><code>{singleAnswer.model_id}</code></span>
                          )}
                          <span className="rag-intent-badge">Intent: {singleAnswer.intent}</span>
                          <span
                            className="rag-intent-badge"
                            style={{
                              background: singleAnswer.pipeline_type === 'baseline' ? 'rgba(100, 116, 139, 0.2)' : 'rgba(14, 165, 233, 0.2)',
                              color: singleAnswer.pipeline_type === 'baseline' ? '#94a3b8' : '#38bdf8',
                              borderColor: singleAnswer.pipeline_type === 'baseline' ? '#475569' : '#0284c7',
                            }}
                          >
                            {singleAnswer.pipeline_type === 'baseline'
                              ? 'Pipeline: Baseline (Isolated)'
                              : `Active Workflow: ${singleAnswer.workflow_version_id || 'v10'}`}
                          </span>
                        </div>
                        {singleAnswer.status === 'insufficient_credits' && (
                          <div className="rag-eval-status-banner credits" style={{ marginBottom: 12 }}>
                            <strong>💳 Insufficient Provider Credits:</strong> {singleAnswer.error || 'The model provider returned HTTP 402: Insufficient credits. Please fund credits to continue.'}
                          </div>
                        )}
                        {singleAnswer.error && (singleAnswer.error.toLowerCase().includes('429') || singleAnswer.error.toLowerCase().includes('rate limit')) && (
                          <div className="rag-eval-status-banner warning" style={{ marginBottom: 12 }}>
                            <strong>⏳ OpenRouter Rate Limit Exceeded (HTTP 429):</strong> The free-tier model is currently throttled by the provider. Please wait a moment and retry.
                          </div>
                        )}
                        {singleAnswer.error && (singleAnswer.error.toLowerCase().includes('401') || singleAnswer.error.toLowerCase().includes('unauthorized')) && (
                          <div className="rag-eval-status-banner error" style={{ marginBottom: 12 }}>
                            <strong>🔑 Authentication Error (HTTP 401):</strong> Invalid or missing API key. Please check OPENROUTER_API_KEY in backend configuration.
                          </div>
                        )}
                        {singleAnswer.status === 'error' && !singleAnswer.error?.toLowerCase().includes('429') && !singleAnswer.error?.toLowerCase().includes('rate limit') && !singleAnswer.error?.toLowerCase().includes('401') && !singleAnswer.error?.toLowerCase().includes('unauthorized') && (
                          <div className="rag-eval-status-banner error" style={{ marginBottom: 12 }}>
                            <strong>❌ Provider Error:</strong> {singleAnswer.error || 'The model provider returned an error.'}
                          </div>
                        )}
                        <div className="rag-answer-text">
                          <p className={!singleAnswer.success ? 'text-error' : ''}>{singleAnswer.answer}</p>
                        </div>
                        {singleAnswer.disclaimer && (
                          <div className="rag-disclaimer-box">
                            ℹ️ {singleAnswer.disclaimer}
                          </div>
                        )}
                      </div>

                      {/* Single Model Context Preview */}
                      {singleAnswer.context && (
                        <details className="rag-context-accordion" style={{ marginTop: 12 }}>
                          <summary>📋 Retrieved Document Context (Sent to model)</summary>
                          <pre className="rag-context-pre">{singleAnswer.context}</pre>
                        </details>
                      )}

                      {/* AgentEvo Evaluation Scorecard */}
                      {singleAnswer.evaluation ? (
                        <div className="rag-eval-section">
                          <div className="rag-eval-header">
                            <div className="rag-eval-title-group">
                              <span className="rag-eval-pill">AgentEvo</span>
                              <h3>Evaluation Metrics & Telemetry</h3>
                            </div>
                            <span className="rag-eval-model-badge">{singleAnswer.evaluation.model}</span>
                          </div>

                          <div className="rag-eval-grid">
                            {/* Groundedness */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">Groundedness</span>
                              <span className={`rag-eval-metric-value ${singleAnswer.evaluation.groundedness !== null && singleAnswer.evaluation.groundedness !== undefined ? (singleAnswer.evaluation.groundedness >= 0.8 ? 'text-green' : singleAnswer.evaluation.groundedness >= 0.5 ? 'text-yellow' : 'text-red') : 'text-muted'}`}>
                                {formatPercent(singleAnswer.evaluation.groundedness)}
                              </span>
                              <span className="rag-eval-metric-sub">
                                {singleAnswer.evaluation.groundedness !== null && singleAnswer.evaluation.groundedness !== undefined
                                  ? `${singleAnswer.evaluation.supported_claims?.length || singleAnswer.evaluation.supported_claim_count || 0} / ${singleAnswer.evaluation.total_claims} verified claims`
                                  : 'Evaluation unavailable'}
                              </span>
                            </div>

                            {/* Hallucination Rate */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">Hallucination Rate</span>
                              <span className={`rag-eval-metric-value ${singleAnswer.evaluation.hallucination_rate !== null && singleAnswer.evaluation.hallucination_rate !== undefined ? (singleAnswer.evaluation.hallucination_rate === 0 ? 'text-green' : singleAnswer.evaluation.hallucination_rate <= 0.2 ? 'text-yellow' : 'text-red') : 'text-muted'}`}>
                                {formatPercent(singleAnswer.evaluation.hallucination_rate)}
                              </span>
                              <span className="rag-eval-metric-sub">
                                {singleAnswer.evaluation.hallucination_rate !== null && singleAnswer.evaluation.hallucination_rate !== undefined
                                  ? `${singleAnswer.evaluation.unsupported_claims?.length || singleAnswer.evaluation.unsupported_claim_count || 0} unsupported claim(s)`
                                  : 'Evaluation unavailable'}
                              </span>
                            </div>

                            {/* Accuracy */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">Accuracy</span>
                              <span className={`rag-eval-metric-value ${singleAnswer.evaluation.accuracy !== null && singleAnswer.evaluation.accuracy !== undefined ? (singleAnswer.evaluation.accuracy >= 0.8 ? 'text-green' : singleAnswer.evaluation.accuracy >= 0.5 ? 'text-yellow' : 'text-red') : 'text-muted'}`}>
                                {singleAnswer.evaluation.accuracy !== null && singleAnswer.evaluation.accuracy !== undefined
                                  ? formatPercent(singleAnswer.evaluation.accuracy)
                                  : 'N/A'}
                              </span>
                              <span className="rag-eval-metric-sub">
                                {singleAnswer.evaluation.accuracy !== null && singleAnswer.evaluation.accuracy !== undefined
                                  ? 'Reference match score'
                                  : 'No reference answer provided'}
                              </span>
                            </div>

                            {/* Token Usage */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">Token Usage</span>
                              <span className="rag-eval-metric-value text-purple">
                                {formatTokens(singleAnswer.evaluation.total_tokens)}
                              </span>
                              <span className="rag-eval-metric-sub">
                                In: {formatTokens(singleAnswer.evaluation.input_tokens)} | Out: {formatTokens(singleAnswer.evaluation.output_tokens)}
                              </span>
                            </div>

                            {/* LLM Latency */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">LLM Latency</span>
                              <span className="rag-eval-metric-value text-yellow">
                                {formatMs(singleAnswer.evaluation.latency_ms)}
                              </span>
                              <span className="rag-eval-metric-sub">Model Inference Latency</span>
                            </div>

                            {/* Execution Time */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">Execution Time</span>
                              <span className="rag-eval-metric-value text-cyan">
                                {formatMs(singleAnswer.evaluation.execution_time_ms)}
                              </span>
                              <span className="rag-eval-metric-sub">RAG + LLM + Eval Workflow</span>
                            </div>

                            {/* Estimated Cost */}
                            <div className="rag-eval-metric-card">
                              <span className="rag-eval-metric-label">Estimated Cost</span>
                              <span className="rag-eval-metric-value text-emerald">
                                {formatCost(singleAnswer.evaluation.total_cost)}
                              </span>
                              <span className="rag-eval-metric-sub">
                                In: {formatCost(singleAnswer.evaluation.input_cost)} | Out: {formatCost(singleAnswer.evaluation.output_cost)}
                              </span>
                            </div>
                          </div>

                          {/* Supported Claims Details */}
                          {singleAnswer.evaluation.supported_claims && singleAnswer.evaluation.supported_claims.length > 0 && (
                            <details className="rag-compare-claims-details" style={{ marginTop: 10 }}>
                              <summary>✓ {singleAnswer.evaluation.supported_claims.length} Supported Claim(s) Verified</summary>
                              <ul className="rag-claims-list supported">
                                {singleAnswer.evaluation.supported_claims.map((claim, cIdx) => (
                                  <li key={cIdx}>{claim}</li>
                                ))}
                              </ul>
                            </details>
                          )}

                          {/* Zero Unsupported Claims Banner */}
                          {singleAnswer.evaluation.total_claims > 0 && (!singleAnswer.evaluation.unsupported_claims || singleAnswer.evaluation.unsupported_claims.length === 0) && (
                            <div className="rag-compare-eval-supported-all" style={{ marginTop: 10 }}>
                              ✓ All factual claims supported by retrieved evidence.
                            </div>
                          )}

                          {/* Unsupported Claims Alert with Claim / Evidence / Reason */}
                          {singleAnswer.evaluation.unsupported_claims && singleAnswer.evaluation.unsupported_claims.length > 0 && (
                            <div className="rag-eval-claims-alert">
                              <strong style={{ display: 'block', marginBottom: 6 }}>⚠️ Unsupported Claims Detected ({singleAnswer.evaluation.unsupported_claims.length}):</strong>
                              <div className="rag-claims-breakdown-list">
                                {singleAnswer.evaluation.unsupported_claims.map((claimText, cIdx) => {
                                  const detail = (singleAnswer.evaluation?.details?.claim_breakdown || []).find((c: any) => c.claim === claimText)
                                  return (
                                    <div key={cIdx} className="rag-claim-card unsupported">
                                      <div className="rag-claim-row"><strong>Claim:</strong> {claimText}</div>
                                      <div className="rag-claim-row"><strong>Evidence:</strong> <span className="text-muted">{detail?.matched_context_snippet || "None found in retrieved document"}</span></div>
                                      <div className="rag-claim-row"><strong>Reason:</strong> <span className="text-red">{detail?.reason || "Factual claim not substantiated by retrieved context"}</span></div>
                                    </div>
                                  )
                                })}
                              </div>
                            </div>
                          )}

                          {/* Status banners */}
                          {singleAnswer.evaluation.details?.status === "insufficient_credits" && (
                            <div className="rag-eval-status-banner credits">
                              💳 Insufficient provider credits: {singleAnswer.evaluation.details?.error_message || "Account credits exhausted."}
                            </div>
                          )}
                          {singleAnswer.evaluation.details?.status === "empty_response" && (
                            <div className="rag-eval-status-banner warning">
                              ⚠️ Empty model response.
                            </div>
                          )}
                          {(singleAnswer.evaluation.details?.status === "model_error" || singleAnswer.evaluation.details?.status === "provider_error") && (
                            <div className="rag-eval-status-banner error">
                              ❌ Provider/API error: {singleAnswer.evaluation.details?.error_message || "API generation failed."}
                            </div>
                          )}

                          {/* Structured Runtime Stage Execution Trace & Provenance */}
                          <details className="rag-compare-claims-details" style={{ marginTop: 12 }} open>
                            <summary>
                              🔍 Execution Workflow & Provenance ({singleAnswer.pipeline_type === 'baseline' ? 'Baseline Pipeline' : `Active Workflow: ${singleAnswer.workflow_version_id || 'v10'}`})
                            </summary>
                            <div style={{ padding: '10px 12px', fontSize: '0.82rem', background: '#f8fafc', borderRadius: 6, marginTop: 8, border: '1px solid #e2e8f0', color: '#1e293b' }}>
                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, marginBottom: 10, paddingBottom: 8, borderBottom: '1px solid #cbd5e1' }}>
                                <div><strong>Pipeline:</strong> <span style={{ color: '#0369a1', fontWeight: 600 }}>{singleAnswer.pipeline_type || 'production'}</span></div>
                                <div><strong>Workflow ID:</strong> <code>{singleAnswer.workflow_version_id || (singleAnswer.pipeline_type === 'baseline' ? 'baseline' : 'v10')}</code></div>
                                <div><strong>Strategy / Name:</strong> <span>{singleAnswer.workflow_name || (singleAnswer.pipeline_type === 'baseline' ? 'Baseline RAG' : 'Retrieval Reduction')}</span></div>
                                <div><strong>Configured Top-K:</strong> <span style={{ color: '#16a34a', fontWeight: 700 }}>{singleAnswer.execution_trace?.stages?.retrieval?.configured_top_k ?? (singleAnswer.pipeline_type === 'baseline' ? 5 : 3)}</span></div>
                                <div><strong>Retrieved Chunks:</strong> <span>{singleAnswer.sources?.length ?? 0}</span></div>
                                <div><strong>Task Category:</strong> <span>{singleAnswer.task_category || 'Clinical Query'}</span></div>
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 8 }}>
                                <div style={{ padding: 8, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 4 }}>
                                  <div style={{ fontWeight: 700, color: '#0369a1', marginBottom: 4 }}>1. Retrieval Stage</div>
                                  <div>Latency: <strong>{singleAnswer.execution_trace?.stages?.retrieval?.latency_ms ?? 0} ms</strong></div>
                                  <div>Chunks: <strong>{singleAnswer.sources?.length ?? singleAnswer.execution_trace?.stages?.retrieval?.chunks_retrieved ?? 0}</strong></div>
                                  <div>Reranking: <strong>{singleAnswer.execution_trace?.stages?.retrieval?.reranking_enabled ? 'Active' : 'Bypassed'}</strong></div>
                                  <div>Threshold: <strong>{singleAnswer.execution_trace?.stages?.retrieval?.similarity_threshold ?? '0.35'}</strong></div>
                                </div>
                                <div style={{ padding: 8, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 4 }}>
                                  <div style={{ fontWeight: 700, color: '#0369a1', marginBottom: 4 }}>2. Context Compression</div>
                                  <div>Status: <strong>{singleAnswer.execution_trace?.stages?.compression?.enabled ? 'Active' : 'Bypassed'}</strong></div>
                                  <div>Chars: <strong>{singleAnswer.execution_trace?.stages?.compression?.effective_context_length ?? singleAnswer.context?.length ?? 0}</strong></div>
                                </div>
                                <div style={{ padding: 8, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 4 }}>
                                  <div style={{ fontWeight: 700, color: '#0369a1', marginBottom: 4 }}>3. Model Generation</div>
                                  <div>Model: <strong>{singleAnswer.execution_trace?.stages?.llm_generation?.model || singleAnswer.model}</strong></div>
                                  <div>Latency: <strong>{singleAnswer.latency_ms !== null ? formatMs(singleAnswer.latency_ms) : 'N/A'}</strong></div>
                                  <div>Tokens: <strong>{formatTokens(singleAnswer.evaluation?.total_tokens)}</strong></div>
                                </div>
                                <div style={{ padding: 8, background: '#fff', border: '1px solid #cbd5e1', borderRadius: 4 }}>
                                  <div style={{ fontWeight: 700, color: '#0369a1', marginBottom: 4 }}>4. Evidence Verification</div>
                                  <div>Groundedness: <strong>{formatPercent(singleAnswer.evaluation?.groundedness)}</strong></div>
                                  <div>Claims: <strong>{singleAnswer.evaluation?.total_claims ?? 0}</strong></div>
                                  <div>Supported: <strong>{singleAnswer.evaluation?.supported_claims?.length ?? 0}</strong></div>
                                </div>
                              </div>
                            </div>
                          </details>
                        </div>
                      ) : (
                        <div className="rag-eval-unavailable">
                          <span>ℹ️ Evaluation metrics not available for this response.</span>
                        </div>
                      )}

                      {/* Source Chunks */}
                      {singleAnswer.sources.length > 0 && (
                        <div className="rag-sources-section">
                          <h3>📑 Grounding Source Chunks ({singleAnswer.sources.length})</h3>
                          <div className="rag-sources-grid">
                            {singleAnswer.sources.map((src, idx) => (
                              <div key={src.chunk_id || idx} className="rag-source-card">
                                <div className="rag-source-header">
                                  <span className="rag-badge rag-badge-index">Chunk #{src.chunk_index}</span>
                                  {src.section_title && (
                                    <span className="rag-badge rag-badge-section">📌 {src.section_title}</span>
                                  )}
                                  {typeof src.relevance_score === 'number' && (
                                    <span className="rag-metric-score">
                                      Relevance: {(src.relevance_score * 100).toFixed(1)}%
                                    </span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* 3-Model Side-by-Side Comparison Output */}
              {queryMode === 'compare' && (
                <div className="rag-card rag-results-panel">
                  <div className="rag-card-header">
                    <h2>⚖️ 3-Model Comparative Evaluation</h2>
                    {compareResults && (
                      <span className="rag-results-count">
                        Query Intent: <strong>{compareResults.intent}</strong>
                      </span>
                    )}
                  </div>

                  {generating && (
                    <div className="rag-placeholder">
                      <div className="rag-spinner"></div>
                      <p>Executing RAG retrieval once & querying Gemini 3.5 Flash-Lite, Qwen 3.6 27B, and Gemma 4 26B A4B (OpenRouter) concurrently with AgentEvo evaluation...</p>
                    </div>
                  )}

                  {!generating && !compareResults && (
                    <div className="rag-placeholder">
                      <span className="rag-placeholder-icon">⚖️</span>
                      <p>Click <strong>Run 3-Model Comparative Evaluation</strong> to compare answers, groundedness, tokens, cost, and latency side-by-side.</p>
                    </div>
                  )}

                  {!generating && compareResults && (
                    <div className="rag-comparison-container">
                      {/* Common Context Preview */}
                      {compareResults.context && (
                        <details className="rag-context-accordion" open>
                          <summary style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                            <span>📋 Shared Retrieved Document Context (Sent to all 3 models)</span>
                            {compareResults.workflow_name && (
                              <span style={{ fontSize: '0.75rem', background: '#e0f2fe', color: '#0369a1', padding: '2px 8px', borderRadius: 4, fontWeight: 600 }}>
                                {compareResults.workflow_name}
                              </span>
                            )}
                            {compareResults.context_hash && (
                              <span style={{ fontSize: '0.72rem', color: '#64748b', background: '#f1f5f9', padding: '2px 6px', borderRadius: 4, fontFamily: 'monospace' }} title={`Full SHA-256: ${compareResults.context_hash}`}>
                                Hash: {compareResults.context_hash.slice(0, 12)}...
                              </span>
                            )}
                          </summary>
                          <pre className="rag-context-pre">{compareResults.context}</pre>
                        </details>
                      )}

                      {/* 3 Model Cards Side-by-Side */}
                      <div className="rag-compare-grid">
                        {compareResults.results.map((res, idx) => (
                          <div key={res.model || idx} className="rag-compare-col">
                            <div className="rag-compare-card-header">
                              <div>
                                <h3 className="rag-compare-model-name">{res.model}</h3>
                                <span className="rag-compare-provider">{res.provider}</span>
                                {res.model_id && (
                                  <span className="rag-model-id-sub"><code>{res.model_id}</code></span>
                                )}
                              </div>
                              <div className="rag-compare-badges">
                                {res.latency_ms !== null && res.latency_ms !== undefined && (
                                  <span className="rag-metric-tag" title="Model API Network & Inference Latency">⚡ {formatMs(res.latency_ms)}</span>
                                )}
                                <span className={`rag-status-chip ${
                                  !res.success && (res.evaluation?.details?.status === 'insufficient_credits' || (res.error && res.error.toLowerCase().includes('credit')))
                                    ? 'rag-status-insufficient_credits'
                                    : !res.success && (res.error && (res.error.toLowerCase().includes('429') || res.error.toLowerCase().includes('rate limit')))
                                    ? 'rag-status-warning'
                                    : !res.success && (res.error && (res.error.toLowerCase().includes('401') || res.error.toLowerCase().includes('unauthorized')))
                                    ? 'rag-status-error'
                                    : res.success
                                    ? 'rag-status-success'
                                    : 'rag-status-error'
                                }`}>
                                  {!res.success && (res.evaluation?.details?.status === 'insufficient_credits' || (res.error && res.error.toLowerCase().includes('credit')))
                                    ? '💳 Insufficient Credits'
                                    : !res.success && (res.error && (res.error.toLowerCase().includes('429') || res.error.toLowerCase().includes('rate limit')))
                                    ? '⏳ Rate Limited (429)'
                                    : !res.success && (res.error && (res.error.toLowerCase().includes('401') || res.error.toLowerCase().includes('unauthorized')))
                                    ? '🔑 Auth Error (401)'
                                    : res.success
                                    ? 'Success'
                                    : 'Provider Error'}
                                </span>
                              </div>
                            </div>

                            <div className="rag-compare-body">
                              <p className={`rag-compare-answer ${!res.success ? 'text-error' : ''}`}>{res.answer}</p>
                              {res.error && (
                                <div className="rag-compare-error">
                                  <span>Error / Notice:</span> {res.error}
                                </div>
                              )}

                              {/* Per-Model AgentEvo Evaluation Metrics */}
                              {res.evaluation ? (
                                <div className="rag-compare-eval-box">
                                  <div className="rag-compare-eval-title">
                                    <span>AgentEvo Evaluation</span>
                                    <span className="rag-compare-eval-cost">{formatCost(res.evaluation.total_cost)}</span>
                                  </div>
                                  <div className="rag-compare-eval-metrics">
                                    <div className="rag-compare-eval-item">
                                      <span className="rag-compare-eval-label">Groundedness</span>
                                      <span className={`rag-compare-eval-val ${res.evaluation.groundedness !== null && res.evaluation.groundedness !== undefined ? (res.evaluation.groundedness >= 0.8 ? 'text-green' : res.evaluation.groundedness >= 0.5 ? 'text-yellow' : 'text-red') : 'text-muted'}`}>
                                        {formatPercent(res.evaluation.groundedness)}
                                      </span>
                                    </div>
                                    <div className="rag-compare-eval-item">
                                      <span className="rag-compare-eval-label">Hallucination</span>
                                      <span className={`rag-compare-eval-val ${res.evaluation.hallucination_rate !== null && res.evaluation.hallucination_rate !== undefined ? (res.evaluation.hallucination_rate === 0 ? 'text-green' : res.evaluation.hallucination_rate <= 0.2 ? 'text-yellow' : 'text-red') : 'text-muted'}`}>
                                        {formatPercent(res.evaluation.hallucination_rate)}
                                      </span>
                                    </div>
                                    <div className="rag-compare-eval-item">
                                      <span className="rag-compare-eval-label">Accuracy</span>
                                      <span className={`rag-compare-eval-val ${res.evaluation.accuracy !== null && res.evaluation.accuracy !== undefined ? (res.evaluation.accuracy >= 0.8 ? 'text-green' : res.evaluation.accuracy >= 0.5 ? 'text-yellow' : 'text-red') : 'text-muted'}`}>
                                        {res.evaluation.accuracy !== null && res.evaluation.accuracy !== undefined ? formatPercent(res.evaluation.accuracy) : 'N/A'}
                                      </span>
                                    </div>
                                    <div className="rag-compare-eval-item">
                                      <span className="rag-compare-eval-label">Tokens (Total)</span>
                                      <span className="rag-compare-eval-val text-purple">{formatTokens(res.evaluation.total_tokens)}</span>
                                    </div>
                                    <div className="rag-compare-eval-item" title="Model API Network & Inference Latency">
                                      <span className="rag-compare-eval-label">LLM Latency</span>
                                      <span className="rag-compare-eval-val text-yellow">{formatMs(res.evaluation.latency_ms)}</span>
                                    </div>
                                    <div className="rag-compare-eval-item" title="Dedicated Model Pipeline Time (RAG Retrieval + Model Latency + Eval)">
                                      <span className="rag-compare-eval-label">Exec Time</span>
                                      <span className="rag-compare-eval-val text-cyan">{formatMs(res.evaluation.execution_time_ms)}</span>
                                    </div>
                                  </div>

                                  {/* Supported Claims Details */}
                                  {res.evaluation.supported_claims && res.evaluation.supported_claims.length > 0 && (
                                    <details className="rag-compare-claims-details">
                                      <summary>✓ {res.evaluation.supported_claims.length} Supported Claim(s)</summary>
                                      <ul className="rag-claims-list supported">
                                        {res.evaluation.supported_claims.map((claim, cIdx) => (
                                          <li key={cIdx}>{claim}</li>
                                        ))}
                                      </ul>
                                    </details>
                                  )}

                                  {/* Zero Unsupported Claims Banner */}
                                  {res.evaluation.total_claims > 0 && (!res.evaluation.unsupported_claims || res.evaluation.unsupported_claims.length === 0) && (
                                    <div className="rag-compare-eval-supported-all">
                                      ✓ All factual claims supported by retrieved evidence.
                                    </div>
                                  )}

                                  {/* Unsupported Claims Alert with Claim / Evidence / Reason */}
                                  {res.evaluation.unsupported_claims && res.evaluation.unsupported_claims.length > 0 && (
                                    <div className="rag-compare-eval-unsupported">
                                      <div className="rag-unsupported-header">⚠️ {res.evaluation.unsupported_claims.length} Unsupported Claim(s):</div>
                                      <div className="rag-claims-breakdown-list">
                                        {res.evaluation.unsupported_claims.map((claimText, cIdx) => {
                                          const detail = (res.evaluation?.details?.claim_breakdown || []).find((c: any) => c.claim === claimText)
                                          return (
                                            <div key={cIdx} className="rag-claim-card unsupported">
                                              <div className="rag-claim-row"><strong>Claim:</strong> {claimText}</div>
                                              <div className="rag-claim-row"><strong>Evidence:</strong> <span className="text-muted">{detail?.matched_context_snippet || "None found in retrieved document"}</span></div>
                                              <div className="rag-claim-row"><strong>Reason:</strong> <span className="text-red">{detail?.reason || "Factual claim not substantiated by retrieved context"}</span></div>
                                            </div>
                                          )
                                        })}
                                      </div>
                                    </div>
                                  )}

                                  {/* Status banners */}
                                  {res.evaluation.details?.status === "insufficient_credits" && (
                                    <div className="rag-eval-status-banner credits">
                                      💳 Insufficient provider credits: {res.evaluation.details?.error_message || "Account credits exhausted."}
                                    </div>
                                  )}
                                  {res.evaluation.details?.status === "empty_response" && (
                                    <div className="rag-eval-status-banner warning">
                                      ⚠️ Empty model response.
                                    </div>
                                  )}
                                  {(res.evaluation.details?.status === "model_error" || res.evaluation.details?.status === "provider_error") && (
                                    <div className="rag-eval-status-banner error">
                                      ❌ Provider/API error: {res.evaluation.details?.error_message || "API generation failed."}
                                    </div>
                                  )}

                                  {/* Model Workflow Provenance & Execution Trace */}
                                  <details className="rag-compare-claims-details" style={{ marginTop: 10 }}>
                                    <summary>🔍 Provenance & Trace ({res.pipeline_type || 'comparison_shared_retrieval'})</summary>
                                    <div style={{ padding: '8px 10px', fontSize: '0.78rem', background: '#f8fafc', borderRadius: 4, marginTop: 6, border: '1px solid #e2e8f0', color: '#1e293b' }}>
                                      <div style={{ marginBottom: 4 }}><strong>Pipeline:</strong> <span style={{ color: '#0369a1', fontWeight: 600 }}>{res.pipeline_type || 'comparison_shared_retrieval'}</span></div>
                                      <div style={{ marginBottom: 4 }}><strong>Workflow:</strong> {res.workflow_name || 'Multi-LLM Comparative Baseline'} {res.workflow_version_id ? `(${res.workflow_version_id})` : ''}</div>
                                      <div style={{ marginBottom: 4 }}><strong>Context Type:</strong> {res.context_type || 'shared'} ({compareResults.sources?.length ?? 0} shared chunks)</div>
                                      {(res.context_hash || compareResults.context_hash) && (
                                        <div style={{ marginBottom: 4 }}><strong>Context Hash:</strong> <code style={{ fontSize: '0.72rem' }}>{(res.context_hash || compareResults.context_hash)?.slice(0, 16)}...</code></div>
                                      )}
                                      <div style={{ marginBottom: 4 }}><strong>Model ID:</strong> <code>{res.model_id || res.model}</code></div>
                                      <div style={{ marginBottom: 4 }}><strong>Latency:</strong> {formatMs(res.latency_ms)} | <strong>Tokens:</strong> {formatTokens(res.evaluation?.total_tokens)}</div>
                                      <div style={{ marginBottom: 4 }}><strong>Cost:</strong> {formatCost(res.evaluation?.total_cost)}</div>
                                      <div><strong>Groundedness:</strong> {formatPercent(res.evaluation?.groundedness)}</div>
                                    </div>
                                  </details>
                                </div>
                              ) : (
                                <div className="rag-compare-eval-unavailable">
                                  <span>Evaluation metrics unavailable</span>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>

                      {/* Common Sources */}
                      {compareResults.sources.length > 0 && (
                        <div className="rag-sources-section">
                          <h3>📑 Grounding Source Sections ({compareResults.sources.length})</h3>
                          <div className="rag-sources-grid">
                            {compareResults.sources.map((src, idx) => (
                              <div key={src.chunk_id || idx} className="rag-source-card">
                                <div className="rag-source-header">
                                  <span className="rag-badge rag-badge-index">Chunk #{src.chunk_index}</span>
                                  {src.section_title && (
                                    <span className="rag-badge rag-badge-section">📌 {src.section_title}</span>
                                  )}
                                  {typeof src.relevance_score === 'number' && (
                                    <span className="rag-metric-score">
                                      Relevance: {(src.relevance_score * 100).toFixed(1)}%
                                    </span>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {compareResults.disclaimer && (
                        <div className="rag-disclaimer-box" style={{ marginTop: 20 }}>
                          ℹ️ {compareResults.disclaimer}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </>
          )}

          {/* Raw Vector Retrieval Tab */}
          {activeTab === 'vector' && (
            <div className="rag-card">
              <div className="rag-card-header">
                <h2>🔍 Direct Semantic Vector Retrieval (RAG Chunks)</h2>
                {selectedDocId && (
                  <span className="rag-filter-indicator">
                    Scoped: {getDocNameById(selectedDocId)} (
                    <button onClick={() => setSelectedDocId('')} className="rag-link-btn">
                      Clear
                    </button>
                    )
                  </span>
                )}
              </div>

              {queryError && <div className="rag-alert rag-alert-error">⚠️ {queryError}</div>}

              <form onSubmit={handleVectorSearch} className="rag-search-form">
                <div className="rag-form-group">
                  <label>Document Scope:</label>
                  <select
                    value={selectedDocId}
                    onChange={(e) => setSelectedDocId(e.target.value)}
                    className="rag-select"
                  >
                    <option value="">🌐 All Documents</option>
                    {documents.map((doc) => (
                      <option key={doc.id} value={doc.id}>
                        📄 {getDocFilename(doc)} ({doc.chunk_count ?? 0} chunks)
                      </option>
                    ))}
                  </select>
                </div>

                <div className="rag-form-group">
                  <label>Search Query:</label>
                  <textarea
                    rows={3}
                    placeholder="e.g. What is the clinical diagnosis?"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="rag-textarea"
                    required
                  />
                </div>

                <div className="rag-form-group">
                  <div className="rag-slider-header">
                    <label>Top-K Retrieval Limit:</label>
                    <span className="rag-slider-value">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="15"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value, 10))}
                    className="rag-slider"
                  />
                </div>

                <button
                  type="submit"
                  disabled={searching || !isLoggedIn}
                  className="rag-btn rag-btn-primary rag-btn-block rag-btn-lg"
                >
                  {searching ? '⚡ Searching Vector Store...' : '🚀 Retrieve Relevant Chunks'}
                </button>
              </form>

              {/* Vector Results */}
              <div className="rag-results-subpanel" style={{ marginTop: 24 }}>
                {searching && (
                  <div className="rag-placeholder">
                    <div className="rag-spinner"></div>
                    <p>Generating 1024-dim BGE-M3 query embeddings & running multi-signal reranked search...</p>
                  </div>
                )}

                {!searching && !searchResults && (
                  <div className="rag-placeholder">
                    <span className="rag-placeholder-icon">🎯</span>
                    <p>Enter a query above and click <strong>Retrieve Relevant Chunks</strong>.</p>
                  </div>
                )}

                {!searching && searchResults && searchResults.results.length === 0 && (
                  <div className="rag-placeholder">
                    <span className="rag-placeholder-icon">📭</span>
                    <p>No matching chunks found for query: "<em>{searchResults.query}</em>"</p>
                  </div>
                )}

                {!searching && searchResults && searchResults.results.length > 0 && (
                  <div className="rag-chunks-list">
                    {searchResults.results.map((chunk, index) => {
                      const relevance = typeof chunk.relevance_score === 'number' ? chunk.relevance_score : chunk.similarity_score
                      const denseSim = typeof chunk.similarity_score === 'number' ? chunk.similarity_score : 0
                      const dist = typeof chunk.distance === 'number' ? chunk.distance : 0

                      return (
                        <div key={chunk.chunk_id || index} className="rag-chunk-card">
                          <div className="rag-chunk-header">
                            <div className="rag-chunk-badges">
                              <span className="rag-badge rag-badge-index">Chunk #{chunk.chunk_index ?? index}</span>
                              {chunk.section_title && (
                                <span className="rag-badge rag-badge-section">📌 {chunk.section_title}</span>
                              )}
                              <span className="rag-badge rag-badge-doc">📄 {getDocNameById(chunk.document_id)}</span>
                            </div>
                            <div className="rag-chunk-metrics">
                              <span className="rag-metric-score" title="Multi-Signal Reranked Relevance Score">
                                Relevance: {(relevance * 100).toFixed(1)}%
                              </span>
                              <span className="rag-metric-dense" title="ChromaDB Dense Vector Cosine Similarity">
                                Dense: {(denseSim * 100).toFixed(1)}%
                              </span>
                              <span className="rag-metric-dist" title="ChromaDB Cosine Distance">
                                Dist: {dist.toFixed(4)}
                              </span>
                            </div>
                          </div>

                          <div className="rag-chunk-content">
                            <pre>{chunk.content || ''}</pre>
                          </div>

                          <div className="rag-chunk-footer">
                            <span>Doc ID: {chunk.document_id}</span>
                            <span>Chunk ID: {chunk.chunk_id}</span>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'optimizer' && (
            <AgentEvoOptimizer
              apiBase={API_BASE}
              documents={documents}
              selectedDocId={selectedDocId}
              setSelectedDocId={setSelectedDocId}
              isLoggedIn={isLoggedIn}
            />
          )}
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <MainApp />
    </ErrorBoundary>
  )
}
