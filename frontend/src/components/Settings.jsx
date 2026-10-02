import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { CompareChart } from './Charts.jsx'

export default function Settings({ user, theme, onTheme, onLogout, onLock }) {
  const isCaregiver = user.role === 'caregiver'
  return (
    <div className="settings">
      <h1 className="page-title">Settings</h1>

      <Section title="Account">
        <dl className="kv">
          <div><dt>Name</dt><dd>{user.name}</dd></div>
          <div><dt>Username</dt><dd>{user.username}</dd></div>
          {user.caregiver && <div><dt>Caregiver</dt><dd>{user.caregiver.name}</dd></div>}
          {isCaregiver && (
            <div><dt>Looking after</dt>
              <dd>{user.patients?.map((p) => p.name).join(', ') || 'No one yet'}</dd></div>
          )}
        </dl>
        <ChangePassword />
        <div className="actions">
          <button className="btn ghost" onClick={onLogout}>Log out</button>
          <button className="btn ghost" onClick={onLock}>Lock app</button>
        </div>
      </Section>

      <Section title="Appearance">
        <div className="segmented" role="group" aria-label="Theme">
          {['light', 'dark'].map((t) => (
            <button key={t} className={theme === t ? 'active' : ''} aria-pressed={theme === t}
              onClick={() => onTheme(t)}>{t === 'light' ? 'Light' : 'Dark'}</button>
          ))}
        </div>
      </Section>

      {isCaregiver && (
        <Section title="Email alerts">
          <Notifications />
        </Section>
      )}

      <Section title="Memory tools"
        hint="For testing how memories fade without waiting days.">
        <Simulate user={user} />
        <Compare />
      </Section>

      <Section title="Start fresh">
        <Reset user={user} />
      </Section>
    </div>
  )
}

function Section({ title, hint, children }) {
  return (
    <section className="settings-section">
      <div className="settings-label">
        <h2>{title}</h2>
        {hint && <p className="muted small">{hint}</p>}
      </div>
      <div className="settings-body">{children}</div>
    </section>
  )
}

function ChangePassword() {
  const [open, setOpen] = useState(false)
  const [cur, setCur] = useState('')
  const [next, setNext] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')

  async function save(e) {
    e.preventDefault()
    setError(''); setMsg('')
    try {
      await api.changePassword(cur, next)
      setMsg('Password changed.')
      setCur(''); setNext(''); setOpen(false)
    } catch (err) { setError(err.message) }
  }

  if (!open) {
    return (
      <>
        {msg && <p className="form-ok">{msg}</p>}
        <button className="link-btn" onClick={() => setOpen(true)}>Change password</button>
      </>
    )
  }
  return (
    <form className="stack tight narrow" onSubmit={save}>
      <label className="field"><span>Current password</span>
        <input type="password" value={cur} onChange={(e) => setCur(e.target.value)}
          autoComplete="current-password" /></label>
      <label className="field"><span>New password</span>
        <input type="password" value={next} onChange={(e) => setNext(e.target.value)}
          autoComplete="new-password" /></label>
      {error && <p className="form-error">{error}</p>}
      <div className="actions">
        <button className="btn primary small" disabled={!cur || next.length < 6}>Change password</button>
        <button type="button" className="btn ghost small" onClick={() => setOpen(false)}>Cancel</button>
      </div>
    </form>
  )
}

function Notifications() {
  const [s, setS] = useState(null)
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')

  useEffect(() => { api.notifications().then(setS).catch((e) => setError(e.message)) }, [])

  async function save(e) {
    e.preventDefault()
    setError(''); setMsg('')
    try {
      const saved = await api.saveNotifications(s.email_enabled, s.email)
      setS({ ...s, ...saved })
      setMsg('Saved.')
    } catch (err) { setError(err.message) }
  }

  if (!s) return error ? <p className="form-error">{error}</p> : null
  return (
    <form className="stack tight narrow" onSubmit={save}>
      <label className="check">
        <input type="checkbox" checked={s.email_enabled}
          onChange={(e) => setS({ ...s, email_enabled: e.target.checked })} />
        Email me when an important memory starts to fade
      </label>
      <label className="field"><span>Email address</span>
        <input type="email" value={s.email || ''} placeholder="you@example.com"
          onChange={(e) => setS({ ...s, email: e.target.value })} /></label>
      {!s.smtp_configured && (
        <p className="muted small">
          Email sending isn't configured on the server yet (SMTP settings in .env),
          so alerts will only show on the dashboard for now.
        </p>
      )}
      {error && <p className="form-error">{error}</p>}
      {msg && <p className="form-ok">{msg}</p>}
      <div><button className="btn primary small">Save</button></div>
    </form>
  )
}

function Simulate({ user }) {
  const isCaregiver = user.role === 'caregiver'
  const [hours, setHours] = useState(12)
  const [target, setTarget] = useState(isCaregiver ? user.patients?.[0]?.username || '' : '')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function run() {
    setBusy(true); setError(''); setMsg('')
    try {
      const r = await api.simulate(hours, target || undefined)
      const avg = r.summary.avg_retention
      setMsg(`Moved ${r.memories_aged} ${r.memories_aged === 1 ? 'memory' : 'memories'} ${hours} hours forward.` +
        (avg != null ? ` Average retention is now ${Math.round(avg * 100)}%.` : ''))
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  return (
    <div className="tool">
      <h3 className="sub-title">Simulate time passing</h3>
      <div className="inline-form">
        {isCaregiver && (
          <select value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Person">
            {user.patients?.map((p) => <option key={p.username} value={p.username}>{p.name}</option>)}
          </select>
        )}
        <select value={hours} onChange={(e) => setHours(Number(e.target.value))} aria-label="Hours">
          {[6, 12, 24, 48, 72].map((h) => <option key={h} value={h}>{h} hours</option>)}
        </select>
        <button className="btn ghost" disabled={busy || (isCaregiver && !target)} onClick={run}>
          {busy ? 'Working…' : 'Simulate'}
        </button>
      </div>
      {msg && <p className="form-ok">{msg}</p>}
      {error && <p className="form-error">{error}</p>}
    </div>
  )
}

function Compare() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const load = () => api.compare().then(setData).catch((e) => setError(e.message))

  return (
    <div className="tool">
      <h3 className="sub-title">Compare forgetting</h3>
      {!data && (
        <button className="btn ghost" onClick={load}>Show comparison</button>
      )}
      {error && <p className="form-error">{error}</p>}
      {data && (
        <>
          <p className="muted small">
            How long an ordinary memory lasts with typical memory and at each
            Alzheimer's stage, if it is never recalled.
          </p>
          <CompareChart curves={data.curves} />
          <table className="table">
            <thead><tr><th>Person</th><th>Profile</th><th>Memories</th><th>Avg retention</th></tr></thead>
            <tbody>
              {data.accounts.map((a) => (
                <tr key={a.username}>
                  <td>{a.name}</td>
                  <td>{a.account_type === 'alzheimers'
                    ? `${a.severity[0].toUpperCase()}${a.severity.slice(1)} (${a.decay_multiplier}×)`
                    : 'Standard'}</td>
                  <td>{a.memory_count}</td>
                  <td>{a.avg_retention != null ? `${Math.round(a.avg_retention * 100)}%` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

function Reset({ user }) {
  const isCaregiver = user.role === 'caregiver'
  const [target, setTarget] = useState(isCaregiver ? user.patients?.[0]?.username || '' : '')
  const [confirm, setConfirm] = useState(false)
  const [all, setAll] = useState(false)
  const [key, setKey] = useState('')
  const [msg, setMsg] = useState('')
  const [error, setError] = useState('')

  const targetName = isCaregiver
    ? user.patients?.find((p) => p.username === target)?.name
    : 'your'

  async function resetOne() {
    setError(''); setMsg('')
    try {
      const r = await api.reset(target || undefined)
      setMsg(`Removed ${r.memories_removed} memories and all reminders.`)
      setConfirm(false)
    } catch (err) { setError(err.message) }
  }

  async function resetEverything(e) {
    e.preventDefault()
    setError(''); setMsg('')
    try {
      const r = await api.resetAll(key)
      setMsg(`Removed ${r.memories_removed} memories for every account.`)
      setAll(false); setKey('')
    } catch (err) { setError(err.message) }
  }

  return (
    <div className="stack tight">
      <p className="muted small">
        Deletes memories, reminders and alert history. Accounts and passwords stay.
      </p>
      <div className="inline-form">
        {isCaregiver && (
          <select value={target} onChange={(e) => setTarget(e.target.value)} aria-label="Person">
            {user.patients?.map((p) => <option key={p.username} value={p.username}>{p.name}</option>)}
          </select>
        )}
        {!confirm
          ? <button className="btn danger-ghost" disabled={isCaregiver && !target}
            onClick={() => setConfirm(true)}>
            {isCaregiver ? 'Delete their memories' : 'Delete my memories'}
          </button>
          : (
            <span className="confirm">
              Delete all of {isCaregiver ? `${targetName}'s` : 'your'} memories?
              <button className="btn danger small" onClick={resetOne}>Delete</button>
              <button className="btn ghost small" onClick={() => setConfirm(false)}>Cancel</button>
            </span>
          )}
      </div>

      {!all
        ? <button className="link-btn small" onClick={() => setAll(true)}>
          Reset the whole app (every account)
        </button>
        : (
          <form className="inline-form" onSubmit={resetEverything}>
            <input type="password" placeholder="Access key" value={key}
              onChange={(e) => setKey(e.target.value)} aria-label="Access key" />
            <button className="btn danger" disabled={!key}>Reset everything</button>
            <button type="button" className="btn ghost" onClick={() => setAll(false)}>Cancel</button>
          </form>
        )}
      {msg && <p className="form-ok">{msg}</p>}
      {error && <p className="form-error">{error}</p>}
    </div>
  )
}
