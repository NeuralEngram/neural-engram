import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import Chat from './Chat.jsx'
import Reminders from './Reminders.jsx'

// Home screen for standard and Alzheimer's users:
// chat on the left; today, gentle reminders and reminders on the right.
export default function UserHome({ user }) {
  const [today, setToday] = useState(null)
  const [version, setVersion] = useState(0)
  const isAlz = user.account_type === 'alzheimers'

  const refresh = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    api.today().then(setToday).catch(() => setToday(null))
  }, [version])

  return (
    <div className="home-grid">
      <section className="home-chat">
        <Chat user={user} onChange={refresh} />
      </section>
      <aside className="home-side">
        {isAlz && <GentleReminders items={today?.gentle_reminders} />}
        <TodayCard today={today} />
        <Reminders key={version} onChange={refresh} />
      </aside>
    </div>
  )
}

function GentleReminders({ items }) {
  if (!items?.length) return null
  return (
    <section className="panel gentle" aria-live="polite">
      <h2 className="panel-title">Worth remembering</h2>
      <ul className="plain-list">
        {items.map((g) => <li key={g.id}>{g.text}</li>)}
      </ul>
    </section>
  )
}

export function TodayCard({ today }) {
  if (!today) return null
  const { reminders = [], memories = [] } = today
  if (!reminders.length && !memories.length) return null
  const date = new Date(today.date + 'T00:00:00').toLocaleDateString(undefined, {
    weekday: 'long', day: 'numeric', month: 'long',
  })
  return (
    <section className="panel today">
      <h2 className="panel-title">{date}</h2>
      {reminders.length > 0 && (
        <ul className="plain-list">
          {reminders.map((r) => (
            <li key={r.id}>
              <strong>{r.title}</strong>
              {r.due_at && <span className="muted"> {formatTime(r.due_at)}</span>}
            </li>
          ))}
        </ul>
      )}
      {memories.length > 0 && (
        <>
          <h3 className="sub-title">From what you've told me</h3>
          <ul className="plain-list">
            {memories.map((m) => <li key={m.id}>{m.content}</li>)}
          </ul>
        </>
      )}
    </section>
  )
}

export function formatTime(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const sameDay = d.toDateString() === new Date().toDateString()
  return sameDay
    ? d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
    : d.toLocaleString(undefined, {
      day: 'numeric', month: 'short', hour: 'numeric', minute: '2-digit',
    })
}
