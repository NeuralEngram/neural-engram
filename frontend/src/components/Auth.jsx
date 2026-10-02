import { useEffect, useState } from 'react'
import { api, session } from '../api.js'
import { Brand } from './Gate.jsx'

const TYPES = [
  ['standard', 'Standard', 'Keeps track of what you tell it.'],
  ['alzheimers', "Alzheimer's support", 'Adds gentle reminders and a caregiver.'],
  ['caregiver', 'Caregiver', "Looks after people with Alzheimer's."],
]

export default function Auth({ onSignedIn, onLock, notice }) {
  const [mode, setMode] = useState('signin')

  return (
    <div className="center-screen">
      <div className="auth-card">
        <Brand />
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={mode === 'signin'}
            className={mode === 'signin' ? 'active' : ''}
            onClick={() => setMode('signin')}>Sign in</button>
          <button role="tab" aria-selected={mode === 'register'}
            className={mode === 'register' ? 'active' : ''}
            onClick={() => setMode('register')}>Create account</button>
        </div>
        {notice && <p className="form-note">{notice}</p>}
        {mode === 'signin'
          ? <SignIn onSignedIn={onSignedIn} />
          : <Register onSignedIn={onSignedIn} />}
        <button className="link-btn small" onClick={onLock}>
          Use a different access key
        </button>
      </div>
    </div>
  )
}

function SignIn({ onSignedIn }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError('')
    try {
      const { token, user } = await api.login(username, password)
      session.setToken(token)
      onSignedIn(user)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="stack">
      <label className="field">
        <span>Username</span>
        <input value={username} autoFocus autoComplete="username"
          onChange={(e) => setUsername(e.target.value)} />
      </label>
      <label className="field">
        <span>Password</span>
        <input type="password" value={password} autoComplete="current-password"
          onChange={(e) => setPassword(e.target.value)} />
      </label>
      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="btn primary wide" disabled={busy || !username || !password}>
        {busy ? 'Signing in…' : 'Sign in'}
      </button>
    </form>
  )
}

function Register({ onSignedIn }) {
  const [form, setForm] = useState({
    name: '', username: '', password: '',
    account_type: 'standard', severity: 'mild', caregiver: '',
  })
  const [caregivers, setCaregivers] = useState([])
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    api.caregivers().then(setCaregivers).catch(() => setCaregivers([]))
  }, [])

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })
  const isAlz = form.account_type === 'alzheimers'

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError('')
    const body = {
      name: form.name,
      username: form.username,
      password: form.password,
      account_type: form.account_type,
      ...(isAlz ? { severity: form.severity, caregiver: form.caregiver } : {}),
    }
    try {
      const { token, user } = await api.register(body)
      session.setToken(token)
      onSignedIn(user)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <form onSubmit={submit} className="stack">
      <fieldset className="type-picker">
        <legend>Account type</legend>
        {TYPES.map(([value, label, hint]) => (
          <label key={value} className={form.account_type === value ? 'selected' : ''}>
            <input type="radio" name="account_type" value={value}
              checked={form.account_type === value} onChange={set('account_type')} />
            <span className="type-label">{label}</span>
            <span className="type-hint">{hint}</span>
          </label>
        ))}
      </fieldset>

      <div className="row-2">
        <label className="field">
          <span>Name</span>
          <input value={form.name} onChange={set('name')} autoComplete="name" />
        </label>
        <label className="field">
          <span>Username</span>
          <input value={form.username} onChange={set('username')} autoComplete="username" />
        </label>
      </div>
      <label className="field">
        <span>Password</span>
        <input type="password" value={form.password} onChange={set('password')}
          autoComplete="new-password" />
        <small className="muted">At least 6 characters.</small>
      </label>

      {isAlz && (
        <div className="row-2">
          <label className="field">
            <span>Stage</span>
            <select value={form.severity} onChange={set('severity')}>
              <option value="mild">Mild</option>
              <option value="moderate">Moderate</option>
              <option value="severe">Severe</option>
            </select>
          </label>
          <label className="field">
            <span>Caregiver</span>
            <select value={form.caregiver} onChange={set('caregiver')}>
              <option value="">Choose…</option>
              {caregivers.map((c) => (
                <option key={c.username} value={c.username}>{c.name}</option>
              ))}
            </select>
          </label>
        </div>
      )}

      {error && <p className="form-error" role="alert">{error}</p>}
      <button className="btn primary wide"
        disabled={busy || !form.name || !form.username || !form.password ||
          (isAlz && !form.caregiver)}>
        {busy ? 'Creating account…' : 'Create account'}
      </button>
    </form>
  )
}
