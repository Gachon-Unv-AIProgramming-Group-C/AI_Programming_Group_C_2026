import { useState } from 'react'
import axios from 'axios'
import './App.css'

const MCP_URL = (import.meta as any).env?.VITE_MCP_URL ?? 'http://localhost:8000/mcp'

type Verdict = 'HALLUCINATION' | 'NO_HALLUCINATION' | 'UNCERTAIN'

interface ClusterInfo {
  label: string
  size: number
  members: string[]
  similarityToOriginal: number
}

interface HallucinationResult {
  verdict: Verdict
  is_hallucination: boolean
  confidence: number
  reason: string
  details?: {
    stage?: number
    layersRun: number[]
    mode?: string
    score1?: number
    score2?: number
    score3?: number
    final_score: number
    clusters?: ClusterInfo[]
    paraphrasedQuestions?: string[]
    thresholdsUsed: { t1: number; tStar: number; s2_threshold: number; t2: number; m: number }
    weightsUsed: { w1: number; w2: number; w3: number }
  }
}

const VERDICT_LABEL: Record<Verdict, string> = {
  HALLUCINATION: 'HALLUCINATION',
  NO_HALLUCINATION: 'FACTUAL',
  UNCERTAIN: 'UNCERTAIN',
}

const MODEL_OPTIONS = [
  'gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo',
  'claude-3-5-haiku-20241022', 'claude-3-5-sonnet-20241022',
]

function ScoreBar({ value }: { value: number }) {
  const pct = Math.min(100, value * 100)
  const color = value > 0.7 ? 'var(--red)' : value > 0.4 ? 'var(--amber)' : 'var(--green)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: 'var(--surface-2)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 2, transition: 'width 0.5s' }} />
      </div>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', minWidth: 36, textAlign: 'right' }}>
        {value.toFixed(3)}
      </span>
    </div>
  )
}

export default function App() {
  const [question, setQuestion] = useState('')
  const [response, setResponse] = useState('')
  const [model, setModel] = useState('gpt-4o-mini')
  const [mSamples, setMSamples] = useState(5)
  const [t1, setT1] = useState(0.3)
  const [tStar, setTStar] = useState(0.7)
  const [t2, setT2] = useState(0.5)
  const [useHf, setUseHf] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<HallucinationResult | null>(null)

  async function handleAnalyze() {
    if (!question.trim() || !response.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const rpcRes = await axios.post(MCP_URL, {
        jsonrpc: '2.0',
        id: Date.now(),
        method: 'tools/call',
        params: {
          name: 'check_hallucination',
          arguments: {
            question: question.trim(), response: response.trim(),
            targetModel: model, verifierModel: model,
            m: mSamples, t1, tStar, t2, useHuggingFaceNli: useHf,
          },
        },
      })
      const raw = rpcRes.data?.result?.content?.[0]?.text
      if (!raw) throw new Error('서버로부터 응답을 받지 못했습니다.')
      setResult(JSON.parse(raw) as HallucinationResult)
    } catch (err: unknown) {
      const e = err as any
      setError(e?.response?.data?.error?.message ?? e?.message ?? '알 수 없는 오류가 발생했습니다.')
    } finally {
      setLoading(false)
    }
  }

  const confPct = result ? Math.round(result.confidence * 100) : 0
  const layers = result?.details?.layersRun ?? []

  return (
    <>
      <header className="header">
        <div className="header-logo">
          <div className="header-dot" />
          <span className="header-title">H3llucination Detector</span>
        </div>
        <span className="header-sub">Verify when Uncertain · 3-Layer Cascade</span>
      </header>

      <main className="main">
        <div className="card">
          <div className="card-header">
            <span className="card-label">Input</span>
            <span className="card-tag">JSON-RPC 2.0</span>
          </div>
          <div className="input-grid">
            <div className="input-col">
              <label className="input-label">Question</label>
              <textarea className="textarea"
                placeholder={'LLM에게 던진 질문을 입력하세요\n예) 세종대왕은 언제 태어났나요?'}
                value={question} onChange={e => setQuestion(e.target.value)} />
            </div>
            <div className="input-col">
              <label className="input-label">LLM Response</label>
              <textarea className="textarea"
                placeholder={'LLM의 응답을 붙여넣으세요\n예) 세종대왕은 1397년 5월 15일에 태어났습니다.'}
                value={response} onChange={e => setResponse(e.target.value)} />
            </div>
          </div>

          <button className="advanced-toggle" onClick={() => setShowAdvanced(v => !v)}>
            Advanced Settings
            <span className={`toggle-arrow ${showAdvanced ? 'open' : ''}`}>▼</span>
          </button>

          {showAdvanced && (
            <div className="advanced-body">
              <div className="field">
                <label className="field-label">Target Model</label>
                <select className="field-select" value={model} onChange={e => setModel(e.target.value)}>
                  {MODEL_OPTIONS.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
              </div>
              <div className="field">
                <label className="field-label">Samples (m)</label>
                <input className="field-input" type="number" min={1} max={10} value={mSamples}
                  onChange={e => setMSamples(Number(e.target.value))} />
              </div>
              <div className="field">
                <label className="field-label">t1 — L1 pass</label>
                <input className="field-input" type="number" step={0.05} min={0} max={1} value={t1}
                  onChange={e => setT1(Number(e.target.value))} />
              </div>
              <div className="field">
                <label className="field-label">t* — L1 force</label>
                <input className="field-input" type="number" step={0.05} min={0} max={1} value={tStar}
                  onChange={e => setTStar(Number(e.target.value))} />
              </div>
              <div className="field">
                <label className="field-label">t2 — final</label>
                <input className="field-input" type="number" step={0.05} min={0} max={1} value={t2}
                  onChange={e => setT2(Number(e.target.value))} />
              </div>
              <div className="field">
                <label className="field-label">HF NLI</label>
                <select className="field-select" value={useHf ? '1' : '0'}
                  onChange={e => setUseHf(e.target.value === '1')}>
                  <option value="0">Off</option>
                  <option value="1">On (HF API Key 필요)</option>
                </select>
              </div>
            </div>
          )}

          <div className="submit-row">
            <button className="btn-analyze" onClick={handleAnalyze}
              disabled={loading || !question.trim() || !response.trim()}>
              {loading ? <><div className="spinner" /> Analyzing…</> : '▶ Analyze'}
            </button>
            <span className="hint-text">
              {loading ? `Running ${mSamples} samples via cascade…` : 'question + response 입력 후 분석'}
            </span>
          </div>
        </div>

        {error && (
          <div className="error-box"><span>⚠</span><span>{error}</span></div>
        )}

        <div className="card result-card">
          <div className="card-header">
            <span className="card-label">Result</span>
            {result && (
              <span className="card-tag" style={{
                color: result.verdict === 'HALLUCINATION' ? 'var(--red)'
                  : result.verdict === 'UNCERTAIN' ? 'var(--amber)' : 'var(--green)'
              }}>Stage {result.details?.stage ?? '—'}</span>
            )}
          </div>

          {!result ? (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <div className="empty-title">분석 결과가 여기에 표시됩니다</div>
              <div className="empty-desc">질문과 LLM 응답을 입력하고 Analyze를 클릭하세요</div>
            </div>
          ) : (
            <>
              <div className="verdict-row">
                <span className={`verdict-badge ${result.verdict}`}>{VERDICT_LABEL[result.verdict]}</span>
                <div className="verdict-confidence">
                  <div className="confidence-label">
                    <span>Confidence</span>
                    <span className="confidence-pct">{confPct}%</span>
                  </div>
                  <div className="conf-bar-track">
                    <div className={`conf-bar-fill ${result.verdict}`} style={{ width: `${confPct}%` }} />
                  </div>
                </div>
              </div>

              <div className="reason-box">{result.reason}</div>

              <div className="details-grid">
                <div className="detail-cell">
                  <span className="detail-key">Layers Run</span>
                  <div className="layers-row">
                    {[1, 2, 3].map(n => (
                      <span key={n} className={`layer-pip ${layers.includes(n) ? 'active' : 'inactive'}`}>L{n}</span>
                    ))}
                  </div>
                </div>
                <div className="detail-cell">
                  <span className="detail-key">Final Score</span>
                  <span className="detail-val">{result.details?.final_score?.toFixed(4) ?? '—'}</span>
                </div>
                <div className="detail-cell">
                  <span className="detail-key">Mode</span>
                  <span className="detail-val small">{result.details?.mode ?? '—'}</span>
                </div>
                {result.details?.score1 !== undefined && (
                  <div className="detail-cell">
                    <span className="detail-key">L1 · LSC Risk</span>
                    <ScoreBar value={result.details.score1} />
                  </div>
                )}
                {result.details?.score2 !== undefined && (
                  <div className="detail-cell">
                    <span className="detail-key">L2 · SINdex</span>
                    <ScoreBar value={result.details.score2} />
                  </div>
                )}
                {result.details?.score3 !== undefined && (
                  <div className="detail-cell">
                    <span className="detail-key">L3 · SAC³ Risk</span>
                    <ScoreBar value={result.details.score3} />
                  </div>
                )}
              </div>

              {result.details?.clusters && result.details.clusters.length > 0 && (
                <div className="clusters-section">
                  <div className="section-label">
                    L2 Clusters · {result.details.clusters.length} group
                    {result.details.clusters.length > 1 ? 's' : ''} from{' '}
                    {result.details.clusters.reduce((s, c) => s + c.size, 0)} samples
                  </div>
                  <div className="cluster-list">
                    {result.details.clusters.map((c, i) => (
                      <div key={i} className="cluster-item">
                        <span className="cluster-name">{c.label}</span>
                        <span className="cluster-size">×{c.size}</span>
                        <div className="cluster-bar-wrap">
                          <div className="cluster-bar-fill" style={{ width: `${c.similarityToOriginal * 100}%` }} />
                        </div>
                        <span className="cluster-sim">sim {(c.similarityToOriginal * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {result.details?.paraphrasedQuestions && result.details.paraphrasedQuestions.length > 0 && (
                <div className="clusters-section">
                  <div className="section-label">L3 Paraphrased Questions</div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {result.details.paraphrasedQuestions.map((q, i) => (
                      <div key={i} style={{
                        fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)',
                        background: 'var(--surface-2)', border: '1px solid var(--border)',
                        borderRadius: 'var(--r)', padding: '6px 12px',
                        display: 'flex', gap: 8, alignItems: 'flex-start',
                      }}>
                        <span style={{ color: 'var(--text-dim)', flexShrink: 0 }}>Q{i + 1}</span>
                        <span>{q}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </main>
    </>
  )
}
