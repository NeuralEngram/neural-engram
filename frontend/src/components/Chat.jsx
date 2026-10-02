import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'

const QUICK = {
  standard: [
    'What do you remember about me?',
    'What did I tell you recently?',
    'What are my plans this week?',
  ],
  alzheimers: [
    'What do I have today?',
    'Who is visiting me?',
    'When do I take my medicine?',
  ],
}

export default function Chat({ user, onChange }) {
  const [messages, setMessages] = useState([])
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const endRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, busy])

  async function send(message) {
    const content = (message ?? text).trim()
    if (!content || busy) return
    setText('')
    setError('')
    setMessages((m) => [...m, { from: 'me', text: content }])
    setBusy(true)
    try {
      const res = await api.chat(content)
      setMessages((m) => [...m, { from: 'bot', text: res.reply, used: res.memories_used }])
      onChange?.()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      inputRef.current?.focus()
    }
  }

  async function remember() {
    const content = text.trim()
    if (!content || busy) return
    setBusy(true)
    setError('')
    try {
      await api.addMemory(content, true)
      setText('')
      setMessages((m) => [...m,
        { from: 'me', text: content },
        { from: 'note', text: 'Saved as an important memory.' }])
      onChange?.()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const quick = QUICK[user.account_type] || QUICK.standard
  const firstName = user.name.split(' ')[0]

  return (
    <div className="chat">
      <div className="chat-log" aria-live="polite">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h1 className="greeting">Hello, {firstName}.</h1>
            <p className="muted">
              Tell me about your day, people and plans. I'll remember what
              matters and help you recall it later.
            </p>
          </div>
        )}
        {messages.map((m, i) => <Message key={i} m={m} />)}
        {busy && (
          <div className="msg bot typing" aria-label="Thinking">
            <span /><span /><span />
          </div>
        )}
        <div ref={endRef} />
      </div>

      <div className="quick" aria-label="Quick questions">
        {quick.map((q) => (
          <button key={q} className="chip" disabled={busy} onClick={() => send(q)}>{q}</button>
        ))}
      </div>

      {error && <p className="form-error" role="alert">{error}</p>}

      <form className="composer" onSubmit={(e) => { e.preventDefault(); send() }}>
        <label className="sr-only" htmlFor="composer-input">Message</label>
        <textarea
          id="composer-input"
          ref={inputRef}
          rows={1}
          value={text}
          placeholder="Type a message"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
          }}
        />
        <button type="button" className="btn ghost" disabled={busy || !text.trim()}
          onClick={remember} title="Save this as an important memory">
          Remember this
        </button>
        <button className="btn primary" disabled={busy || !text.trim()}>Send</button>
      </form>
    </div>
  )
}

function Message({ m }) {
  const [open, setOpen] = useState(false)
  if (m.from === 'note') return <p className="msg-note">{m.text}</p>
  return (
    <div className={`msg ${m.from}`}>
      <RichText text={m.text} />
      {m.from === 'bot' && m.used?.length > 0 && (
        <div className="used">
          <button className="link-btn small" onClick={() => setOpen(!open)}
            aria-expanded={open}>
            {open ? 'Hide' : 'Show'} {m.used.length} {m.used.length === 1 ? 'memory' : 'memories'} used
          </button>
          {open && (
            <ul className="used-list">
              {m.used.map((u, i) => (
                <li key={i}>
                  {u.content}
                  {u.relevance != null && (
                    <span className="muted"> ({Math.round(u.relevance * 100)}% match)</span>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

// Minimal, safe formatting: paragraphs, "- " bullets and **bold**.
function RichText({ text }) {
  const lines = String(text).split('\n')
  const blocks = []
  let list = null
  lines.forEach((line, i) => {
    const bullet = line.match(/^\s*[-*•]\s+(.*)/)
    if (bullet) {
      list = list || []
      list.push(<li key={i}>{bold(bullet[1])}</li>)
      return
    }
    if (list) { blocks.push(<ul key={`l${i}`}>{list}</ul>); list = null }
    if (line.trim()) blocks.push(<p key={i}>{bold(line)}</p>)
  })
  if (list) blocks.push(<ul key="lend">{list}</ul>)
  return <div className="rich">{blocks}</div>
}

function bold(s) {
  return s.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**')
      ? <strong key={i}>{part.slice(2, -2)}</strong>
      : part)
}
