import { useCallback, useEffect, useState } from 'react'
import { api, session } from './api.js'
import Gate from './components/Gate.jsx'
import Auth from './components/Auth.jsx'
import TopBar from './components/TopBar.jsx'
import UserHome from './components/UserHome.jsx'
import Memories from './components/Memories.jsx'
import CaregiverDashboard from './components/CaregiverDashboard.jsx'
import Settings from './components/Settings.jsx'

const THEME_STORE = 'maem.theme'

function readTheme() {
  try { return localStorage.getItem(THEME_STORE) || 'light' } catch { return 'light' }
}

export default function App() {
  const [unlocked, setUnlocked] = useState(Boolean(session.accessKey))
  const [user, setUser] = useState(null)
  const [checking, setChecking] = useState(Boolean(session.token))
  const [page, setPage] = useState('home')
  const [notice, setNotice] = useState('')
  const [theme, setTheme] = useState(readTheme)

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    try { localStorage.setItem(THEME_STORE, theme) } catch { /* ignore */ }
  }, [theme])

  const logout = useCallback((message = '') => {
    session.clearToken()
    setUser(null)
    setPage('home')
    setNotice(message)
  }, [])

  useEffect(() => {
    session.onExpired((msg) => logout(msg || 'Your session ended. Please sign in again.'))
  }, [logout])

  // Restore a previous session
  useEffect(() => {
    if (!unlocked || !session.token) { setChecking(false); return }
    api.me()
      .then(setUser)
      .catch(() => session.clearToken())
      .finally(() => setChecking(false))
  }, [unlocked])

  const lock = () => {
    session.forgetAccessKey()
    logout()
    setUnlocked(false)
  }

  if (!unlocked) return <Gate onUnlock={() => setUnlocked(true)} />
  if (checking) return <div className="center-screen muted">Loading…</div>
  if (!user) {
    return (
      <Auth
        notice={notice}
        onSignedIn={(u) => { setNotice(''); setUser(u); setPage('home') }}
        onLock={lock}
      />
    )
  }

  const isCaregiver = user.role === 'caregiver'
  const pages = isCaregiver
    ? [['home', 'Dashboard'], ['settings', 'Settings']]
    : [['home', 'Home'], ['memories', 'Memories'], ['settings', 'Settings']]

  return (
    <div className={`app ${user.account_type === 'alzheimers' ? 'large-text' : ''}`}>
      <TopBar
        user={user}
        pages={pages}
        page={page}
        onNavigate={setPage}
        onLogout={() => logout()}
      />
      <main className="page">
        {page === 'home' && !isCaregiver && <UserHome user={user} />}
        {page === 'home' && isCaregiver && <CaregiverDashboard user={user} />}
        {page === 'memories' && !isCaregiver && <Memories user={user} />}
        {page === 'settings' && (
          <Settings
            user={user}
            theme={theme}
            onTheme={setTheme}
            onLogout={() => logout()}
            onLock={lock}
          />
        )}
      </main>
    </div>
  )
}
