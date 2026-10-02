import { useState } from 'react'
import { api, session } from '../api.js'

// First screen: the application is locked until the MAEM access key
// (MAEM_API_KEY from the server's .env) is entered.
export default function Gate({ onUnlock }) {
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(e) {
    e.preventDefault()
    setBusy(true)
    setError('')
    session.setAccessKey(key.trim())
    try {
      await api.checkKey()
      onUnlock()
    } catch (err) {
      session.forgetAccessKey()
      setError(err.status === 401
        ? 'That access key is not correct.'
        : err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="center-screen">
      <form className="auth-card" onSubmit={submit}>
        <Brand />
        <h1 className="auth-title">Unlock the app</h1>
        <p className="muted">Enter the access key set up for this installation.</p>
        <label className="field">
          <span>Access key</span>
          <input
            type="password"
            autoFocus
            value={key}
            onChange={(e) => setKey(e.target.value)}
            autoComplete="off"
          />
        </label>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="btn primary wide" disabled={busy || !key.trim()}>
          {busy ? 'Checking…' : 'Unlock'}
        </button>
      </form>
    </div>
  )
}

export function Brand() {
  return (
    <div className="brand">
      <span className="brand-mark" aria-hidden="true">
        <i /><i /><i />
      </span>
      <span className="brand-name">NeuralEngram</span>
    </div>
  )
}
