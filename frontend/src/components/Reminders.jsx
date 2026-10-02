import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { formatTime } from './UserHome.jsx'

// Reminders for one person. `user` is set when a caregiver manages a
// patient; omitted for the signed-in user's own reminders.
export default function Reminders({ user, ownerName, onChange }) {
  const [items, setItems] = useState([])
  const [adding, setAdding] = useState(false)
  const [title, setTitle] = useState('')
  const [due, setDue] = useState('')
  const [error, setError] = useState('')

  const load = () => api.reminders(user).then(setItems).catch((e) => setError(e.message))
  useEffect(() => { load() }, [user]) // eslint-disable-line react-hooks/exhaustive-deps

  async function add(e) {
    e.preventDefault()
    setError('')
    try {
      await api.addReminder(title, due || null, user)
      setTitle(''); setDue(''); setAdding(false)
      await load()
      onChange?.()
    } catch (err) { setError(err.message) }
  }

  async function toggle(r) {
    await api.setReminderDone(r.id, !r.done).catch((e) => setError(e.message))
    await load()
    onChange?.()
  }

  async function remove(r) {
    await api.deleteReminder(r.id).catch((e) => setError(e.message))
    await load()
    onChange?.()
  }

  const open = items.filter((r) => !r.done)
  const done = items.filter((r) => r.done)

  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="panel-title">Reminders</h2>
        {!adding && (
          <button className="btn ghost small" onClick={() => setAdding(true)}>Add</button>
        )}
      </div>

      {adding && (
        <form className="stack tight" onSubmit={add}>
          <label className="field">
            <span>What to remember</span>
            <input value={title} autoFocus onChange={(e) => setTitle(e.target.value)}
              placeholder={ownerName ? `For ${ownerName}` : 'Call the pharmacy'} />
          </label>
          <label className="field">
            <span>When <span className="muted">(optional)</span></span>
            <input type="datetime-local" value={due} onChange={(e) => setDue(e.target.value)} />
          </label>
          <div className="actions">
            <button className="btn primary small" disabled={!title.trim()}>Save reminder</button>
            <button type="button" className="btn ghost small"
              onClick={() => { setAdding(false); setTitle(''); setDue('') }}>Cancel</button>
          </div>
        </form>
      )}

      {error && <p className="form-error">{error}</p>}

      {items.length === 0 && !adding && (
        <p className="muted small">Nothing yet.</p>
      )}

      <ul className="reminder-list">
        {[...open, ...done].map((r) => (
          <li key={r.id} className={r.done ? 'done' : ''}>
            <label>
              <input type="checkbox" checked={r.done} onChange={() => toggle(r)} />
              <span className="reminder-title">{r.title}</span>
            </label>
            <span className="reminder-meta">
              {r.due_at && formatTime(r.due_at)}
              {r.created_by !== r.username && (
                <span className="from"> from {capitalise(r.created_by)}</span>
              )}
            </span>
            <button className="icon-btn" aria-label={`Delete ${r.title}`}
              onClick={() => remove(r)}>&times;</button>
          </li>
        ))}
      </ul>
    </section>
  )
}

function capitalise(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s
}
