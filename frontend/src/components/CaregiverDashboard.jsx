import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import Memories from './Memories.jsx'
import Reminders from './Reminders.jsx'
import { TrendChart } from './Charts.jsx'

const STAGE = { mild: 'Mild', moderate: 'Moderate', severe: 'Severe' }

export default function CaregiverDashboard() {
  const [patients, setPatients] = useState(null)
  const [selected, setSelected] = useState('')
  const [version, setVersion] = useState(0)
  const [error, setError] = useState('')

  const refresh = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    api.patients()
      .then((rows) => {
        setPatients(rows)
        setSelected((cur) => cur || rows[0]?.username || '')
      })
      .catch((e) => setError(e.message))
  }, [version])

  if (error) return <p className="form-error">{error}</p>
  if (!patients) return <p className="muted">Loading…</p>
  if (!patients.length) {
    return (
      <p className="empty">
        No one is linked to you yet. When a person with Alzheimer's creates an
        account and chooses you as their caregiver, they appear here.
      </p>
    )
  }

  const patient = patients.find((p) => p.username === selected) || patients[0]

  return (
    <div className="cg-grid">
      <nav className="patient-rail" aria-label="People you look after">
        <h2 className="panel-title">People you look after</h2>
        <ul>
          {patients.map((p) => (
            <li key={p.username}>
              <button className={p.username === patient.username ? 'active' : ''}
                aria-current={p.username === patient.username ? 'true' : undefined}
                onClick={() => setSelected(p.username)}>
                <span className="rail-name">{p.name}</span>
                <span className="rail-meta">
                  {STAGE[p.severity]}
                  {p.avg_retention != null && `, ${Math.round(p.avg_retention * 100)}% retention`}
                </span>
                {p.open_alerts > 0 && (
                  <span className="badge" aria-label={`${p.open_alerts} open alerts`}>
                    {p.open_alerts}
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      </nav>

      <div className="cg-main">
        <header className="patient-head">
          <div>
            <h1 className="page-title">{patient.name}</h1>
            <p className="muted">
              {STAGE[patient.severity]} stage. Memories fade {patient.decay_multiplier}×
              faster than typical.
            </p>
          </div>
          <div className="health inline">
            <Count label="Strong" value={patient.strong} tone="strong" />
            <Count label="Fading" value={patient.fading} tone="fading" />
            <Count label="At risk" value={patient.at_risk} tone="at_risk" />
          </div>
        </header>

        <Alerts patient={patient} version={version} onChange={refresh} />

        <div className="cg-two">
          <section className="panel">
            <h2 className="panel-title">Average retention</h2>
            <Trend user={patient.username} version={version} />
          </section>
          <Reminders key={`${patient.username}-${version}`} user={patient.username}
            ownerName={patient.name} onChange={refresh} />
        </div>

        <Memories compact user={patient.username} ownerName={patient.name}
          version={version} onChange={refresh} />
      </div>
    </div>
  )
}

function Count({ label, value, tone }) {
  return (
    <div className={`health-count ${tone}`}>
      <span className="health-num">{value}</span>
      <span className="small">{label}</span>
    </div>
  )
}

function Trend({ user, version }) {
  const [points, setPoints] = useState(null)
  useEffect(() => {
    api.trend(user).then(setPoints).catch(() => setPoints([]))
  }, [user, version])
  return points ? <TrendChart points={points} /> : null
}

function Alerts({ patient, version, onChange }) {
  const [rows, setRows] = useState([])
  const [showHistory, setShowHistory] = useState(false)

  useEffect(() => {
    api.alerts(patient.username).then(setRows).catch(() => setRows([]))
  }, [patient.username, version])

  const open = rows.filter((a) => a.status === 'open')
  const past = rows.filter((a) => a.status !== 'open')

  async function handle(id) {
    await api.handleAlert(id)
    onChange()
  }

  if (!rows.length) return null

  return (
    <section className={`panel alerts ${open.length ? 'has-open' : ''}`}>
      <div className="panel-head">
        <h2 className="panel-title">
          {open.length
            ? `${open.length} important ${open.length === 1 ? 'memory is' : 'memories are'} fading`
            : 'No open alerts'}
        </h2>
        {past.length > 0 && (
          <button className="link-btn small" onClick={() => setShowHistory(!showHistory)}
            aria-expanded={showHistory}>
            {showHistory ? 'Hide history' : `History (${past.length})`}
          </button>
        )}
      </div>
      {open.length > 0 && (
        <ul className="alert-list">
          {open.map((a) => (
            <li key={a.id} className={a.level}>
              <span className={`level ${a.level}`}>
                {a.level === 'critical' ? 'Critical' : 'Warning'}
              </span>
              <span className="alert-text">{a.memory}</span>
              <span className="muted small">{Math.round(a.retention * 100)}% retained</span>
              <button className="btn ghost small" onClick={() => handle(a.id)}>
                Mark as handled
              </button>
            </li>
          ))}
        </ul>
      )}
      {showHistory && (
        <ul className="alert-list history">
          {past.map((a) => (
            <li key={a.id}>
              <span className="level muted-level">
                {a.status === 'handled' ? 'Handled' : 'Resolved'}
              </span>
              <span className="alert-text">{a.memory}</span>
              <span className="muted small">
                {a.status === 'handled' && a.handled_at
                  ? `by ${a.handled_by} on ${new Date(a.handled_at).toLocaleDateString()}`
                  : 'memory was refreshed'}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
