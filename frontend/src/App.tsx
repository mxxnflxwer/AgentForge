import React, { useState, useEffect, useRef, Component } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import './App.css'

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


interface RAGSearchResponse {
  query: string
  total_results: number
  results: RetrievedChunk[]
}

const API_BASE = 'http://localhost:8000'

// Safe helper getters
const getDocFilename = (doc?: DocumentItem | null): string => {
  if (!doc) return 'Document'
  return doc.filename || doc.original_filename || `Document-${doc.id?.slice(0, 8) || 'unknown'}`
}

const getDocStatus = (doc?: DocumentItem | null): string => {
  if (!doc) return 'COMPLETED'
  return String(doc.status || doc.processing_status || 'COMPLETED')
}

// React Error Boundary to protect against any unexpected rendering errors
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
  const [isLoggedIn, setIsLoggedIn] = useState(false)
  const [currentUser, setCurrentUser] = useState<string | null>(null)

  // Document state
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [loadingDocs, setLoadingDocs] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadSpecialty, setUploadSpecialty] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Search state
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState<number>(5)
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<RAGSearchResponse | null>(null)

  // Feedback / Error state
  const [authError, setAuthError] = useState<string | null>(null)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [searchError, setSearchError] = useState<string | null>(null)

  useEffect(() => {
    checkCurrentUser()
  }, [])

  const checkCurrentUser = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/auth/me`, { credentials: 'include' })
      if (res.ok) {
        const data = await res.json()
        setIsLoggedIn(true)
        setCurrentUser(data?.email || 'Authenticated User')
        fetchDocuments()
      } else {
        setIsLoggedIn(false)
      }
    } catch (err) {
      console.warn('Backend connection notice:', err)
    }
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setAuthError(null)
    try {
      let res = await fetch(`${API_BASE}/api/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ email: userEmail, password }),
      })

      // Auto-register then login if user does not exist
      if (res.status === 401 || res.status === 404) {
        await fetch(`${API_BASE}/api/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ email: userEmail, password, name: 'Developer User' }),
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
        throw new Error(errData.detail || 'Authentication failed')
      }

      setIsLoggedIn(true)
      setCurrentUser(userEmail)
      fetchDocuments()
    } catch (err: any) {
      setAuthError(err.message || 'Login failed')
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
    setDocuments([])
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
      setUploadSuccess(`Successfully uploaded "${uploadedName}"! Ingestion pipeline running.`)
      setSelectedFile(null)
      setUploadSpecialty('')
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      // Refresh list immediately and poll for background processing
      await fetchDocuments()
      setTimeout(() => fetchDocuments(), 2000)
      setTimeout(() => fetchDocuments(), 4000)
    } catch (err: any) {
      setUploadError(err.message || 'Error uploading document')
    } finally {
      setUploading(false)
    }
  }

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) {
      setSearchError('Please enter a semantic search query.')
      return
    }

    setSearchError(null)
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
      setSearchError(err.message || 'Error performing RAG search')
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
          <h1>Phase 3: Embeddings + ChromaDB + RAG</h1>
          <p className="rag-subtitle">PostgreSQL Source of Truth ➔ BGE-M3 1024-dim Embeddings ➔ ChromaDB Vector Store ➔ Multi-Signal Reranking</p>
        </div>
        <div className="rag-header-right">
          <span className="rag-tag">BAAI BGE-M3 (1024-dim)</span>
          <span className="rag-tag-highlight">ChromaDB Vector Store</span>
          {isLoggedIn ? (
            <div className="rag-user-controls">
              <span className="rag-user-badge">🟢 {currentUser || 'Authenticated'}</span>
              <button onClick={handleLogout} className="rag-btn-secondary rag-btn-sm">
                Logout
              </button>
            </div>
          ) : (
            <span className="rag-user-badge-off">🔴 Not Authenticated</span>
          )}
        </div>
      </header>

      {/* Auth Box if not logged in */}
      {!isLoggedIn && (
        <div className="rag-card rag-auth-card">
          <div className="rag-card-header">
            <h2>🔐 Developer Authentication</h2>
          </div>
          <p className="rag-card-desc">Log in with your developer account to access document ingestion and semantic search.</p>
          {authError && <div className="rag-alert rag-alert-error">⚠️ {authError}</div>}
          <form onSubmit={handleLogin} className="rag-auth-form">
            <input
              type="email"
              placeholder="Email address"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
              required
            />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            <button type="submit" className="rag-btn rag-btn-primary">
              Connect Session
            </button>
          </form>
        </div>
      )}

      {/* Main Grid */}
      <div className="rag-grid-main">
        {/* Left Column: Document Ingestion & Document List */}
        <div className="rag-col-left">
          {/* Ingest Card */}
          <div className="rag-card">
            <div className="rag-card-header">
              <h2>📤 Ingest Document</h2>
            </div>
            <p className="rag-card-desc">Upload clinical or technical files (.pdf, .docx, .txt) to extract, chunk, and index vectors.</p>

            {uploadError && <div className="rag-alert rag-alert-error">⚠️ {uploadError}</div>}
            {uploadSuccess && <div className="rag-alert rag-alert-success">✅ {uploadSuccess}</div>}

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
                  placeholder="e.g. Cardiology, Pulmonology, AI"
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
                {uploading ? '⏳ Uploading & Processing...' : '📤 Upload & Index Chunks'}
              </button>
            </form>
          </div>

          {/* Ingested Documents Card */}
          <div className="rag-card">
            <div className="rag-card-header">
              <h2>📚 Ingested Documents ({documents.length})</h2>
              <button
                onClick={fetchDocuments}
                className="rag-btn-secondary rag-btn-sm"
                disabled={loadingDocs || !isLoggedIn}
              >
                {loadingDocs ? '...' : '🔄 Refresh'}
              </button>
            </div>

            {documents.length === 0 ? (
              <div className="rag-placeholder-sm">
                <p>No documents uploaded yet. Upload your first document above!</p>
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
                        <span className="rag-doc-title">📄 {getDocFilename(doc)}</span>
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

        {/* Right Column: Semantic Vector Search & Results */}
        <div className="rag-col-right">
          {/* Search Controls Card */}
          <div className="rag-card">
            <div className="rag-card-header">
              <h2>🔍 Semantic Vector Retrieval</h2>
              {selectedDocId && (
                <span className="rag-filter-indicator">
                  Filter: {getDocNameById(selectedDocId)} (
                  <button onClick={() => setSelectedDocId('')} className="rag-link-btn">
                    Clear
                  </button>
                  )
                </span>
              )}
            </div>

            {searchError && <div className="rag-alert rag-alert-error">⚠️ {searchError}</div>}

            <form onSubmit={handleSearch} className="rag-search-form">
              {/* Document Filter Dropdown */}
              <div className="rag-form-group">
                <label>Document Scope:</label>
                <select
                  value={selectedDocId}
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  className="rag-select"
                >
                  <option value="">🌐 All Documents (Global User Scope)</option>
                  {documents.map((doc) => (
                    <option key={doc.id} value={doc.id}>
                      📄 {getDocFilename(doc)} ({doc.chunk_count ?? 0} chunks)
                    </option>
                  ))}
                </select>
              </div>

              {/* Query Input */}
              <div className="rag-form-group">
                <label>Semantic Search Query:</label>
                <textarea
                  rows={3}
                  placeholder="e.g. What is the disease? What medications are prescribed?"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="rag-textarea"
                  required
                />
              </div>

              {/* Top-K Control */}
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
                {searching ? '⚡ Searching ChromaDB...' : '🚀 Retrieve Relevant Chunks'}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="rag-card rag-results-panel">
            <div className="rag-card-header">
              <h2>📑 Retrieved Chunks</h2>
              {searchResults && (
                <span className="rag-results-count">
                  {searchResults.total_results} chunk{searchResults.total_results === 1 ? '' : 's'} retrieved
                </span>
              )}
            </div>

            {searching && (
              <div className="rag-placeholder">
                <div className="rag-spinner"></div>
                <p>Generating 1024-dim BGE-M3 query embeddings & running multi-signal reranked search...</p>
              </div>
            )}

            {!searching && !searchResults && (
              <div className="rag-placeholder">
                <span className="rag-placeholder-icon">🎯</span>
                <p>Enter a clinical or technical query above and click <strong>Retrieve Relevant Chunks</strong> to perform semantic search.</p>
              </div>
            )}

            {!searching && searchResults && searchResults.results.length === 0 && (
              <div className="rag-placeholder">
                <span className="rag-placeholder-icon">📭</span>
                <p>No matching chunks found in ChromaDB for query: "<em>{searchResults.query}</em>"</p>
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
