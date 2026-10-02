import { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'

const FILTERS = [
  ['all', 'All'],
  ['strong', 'Strong'],
  ['fading', 'Fading'],
  ['at_risk', 'At risk'],
]
const SORTS = [
  ['recent', 'Newest first'],
  ['retention', 'Weakest first'],
  ['priority', 'Highest priority'],
  ['retrieved', 'Most recalled'],
]

// The memory list. Each memory's text is drawn fainter as its
// retention drops — forgetting is visible at a glance.
// `user` is set when a caregiver views a patient.
export default function Memories({ user, ownerName, compact = false, version = 0, onChange }) {
  const [data, setData] = useState(null)
  const [filter, setFilter] = useState('all')
  const [sort, setSort] = useState('recent')
  const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)

  const load = () => api.memories(user?.username ?? user)
    .then(setData).catch((e) => setError(e.message))

  useEffect(() => { load() }, [user, version]) // eslint-disable-line react-hooks/exhaustive-deps

  const items = useMemo(() => {
    if (!data) return []
    const list = data.memories.filter((m) => filter === 'all' || m.health === filter)
    const by = {
      recent: (a, b) => b.created_at.localeCompare(a.created_at),
      retention: (a, b) => a.retention - b.retention,
      priority: (a, b) => b.priority - a.priority,
      retrieved: (a, b) => b.times_retrieved - a.times_retrieved,
    }[sort]
    return [...list].sort(by)
  }, [data, filter, sort])

  async function reinforce(id) {
    try { await api.reinforce(id); await load(); onChange?.() } catch (e) { setError(e.message) }
  }

  const s = data?.summary
  const target = user?.username ?? user

  return (
    <section className={compact ? 'panel' : 'memories-page'}>
      <div className="panel-head">
        {compact
          ? <h2 className="panel-title">Memories</h2>
          : <h1 className="page-title">{ownerName ? `${ownerName}'s memories` : 'Your memories'}</h1>}
        {!adding && (
          <button className="btn ghost small" onClick={() => setAdding(true)}>Add memory</button>
        )}
      </div>

      {adding && (
        <AddMemory target={target} ownerName={ownerName}
          onDone={() => { setAdding(false); load(); onChange?.() }} />
      )}

      {s && s.memory_count > 0 && !compact && (
        <div className="health">
          <HealthCount label="Strong" value={s.strong} tone="strong" />
          <HealthCount label="Fading" value={s.fading} tone="fading" />
          <HealthCount label="At risk" value={s.at_risk} tone="at_risk" />
          <div className="health-avg">
            <span className="health-num">{Math.round((s.avg_retention ?? 0) * 100)}%</span>
            <span className="muted small">average retention</span>
          </div>
        </div>
      )}

      {error && <p className="form-error">{error}</p>}

      {data && data.memories.length === 0 && !adding && (
        <p className="empty">
          {ownerName
            ? `Nothing remembered for ${ownerName} yet.`
            : 'Nothing remembered yet. Chat on the Home page, or add a memory here.'}
        </p>
      )}

      {data && data.memories.length > 0 && (
        <>
          <div className="toolbar">
            <div className="segmented" role="group" aria-label="Filter">
              {FILTERS.map(([id, label]) => (
                <button key={id} className={filter === id ? 'active' : ''}
                  aria-pressed={filter === id} onClick={() => setFilter(id)}>{label}</button>
              ))}
            </div>
            <label className="sort">
              <span className="sr-only">Sort</span>
              <select value={sort} onChange={(e) => setSort(e.target.value)}>
                {SORTS.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
              </select>
            </label>
          </div>

          <ol className="memory-list">
            {items.map((m) => (
              <li key={m.id} className={`memory ${m.health}`}>
                <p className="memory-text" style={{ opacity: 0.38 + 0.62 * m.retention }}>
                  {m.content}
                </p>
                <div className="memory-meter" aria-hidden="true">
                  <span style={{ width: `${Math.max(2, m.retention * 100)}%` }} />
                </div>
                <dl className="memory-stats">
                  <div><dt>Retention</dt><dd>{Math.round(m.retention * 100)}%</dd></div>
                  <div><dt>Importance</dt><dd>{m.importance.toFixed(2)}</dd></div>
                  <div><dt>Priority</dt><dd>{m.priority.toFixed(2)}</dd></div>
                  <div><dt>Recalled</dt><dd>{m.times_retrieved}×</dd></div>
                  <div><dt>Stability</dt><dd>{Math.round(m.stability_hours)} h</dd></div>
                </dl>
                <div className="memory-foot">
                  <span className="muted small">
                    {m.important && <span className="tag">Important</span>}
                    {m.source === 'caregiver' && <span className="tag">Added by caregiver</span>}
                    Saved {new Date(m.created_at).toLocaleDateString()}
                  </span>
                  <button className="link-btn small" onClick={() => reinforce(m.id)}>
                    Refresh memory
                  </button>
                </div>
              </li>
            ))}
          </ol>
          {items.length === 0 && <p className="empty">No memories in this group.</p>}
        </>
      )}
    </section>
  )
}

function HealthCount({ label, value, tone }) {
  return (
    <div className={`health-count ${tone}`}>
      <span className="health-num">{value}</span>
      <span className="small">{label}</span>
    </div>
  )
}

function AddMemory({ target, ownerName, onDone }) {
  const [text, setText] = useState('')
  const [important, setImportant] = useState(false)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function save(e) {
    e.preventDefault()
    setBusy(true)
    try {
      await api.addMemory(text, important, target)
      onDone()
    } catch (err) {
      setError(err.message)
    } finally { setBusy(false) }
  }

  return (
    <form className="stack tight add-memory" onSubmit={save}>
      <label className="field">
        <span>{ownerName ? `Something ${ownerName} should remember` : 'Something to remember'}</span>
        <textarea rows={2} value={text} autoFocus onChange={(e) => setText(e.target.value)}
          placeholder="Doctor's appointment on Friday at 10 am" />
      </label>
      <label className="check">
        <input type="checkbox" checked={important} onChange={(e) => setImportant(e.target.checked)} />
        Important (fades more slowly)
      </label>
      {error && <p className="form-error">{error}</p>}
      <div className="actions">
        <button className="btn primary small" disabled={busy || !text.trim()}>Save memory</button>
        <button type="button" className="btn ghost small" onClick={onDone}>Cancel</button>
      </div>
    </form>
  )
}
