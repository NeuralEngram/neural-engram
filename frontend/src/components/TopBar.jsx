import { Brand } from './Gate.jsx'

const ROLE_LABEL = {
  standard: 'Standard',
  alzheimers: "Alzheimer's support",
}

export default function TopBar({ user, pages, page, onNavigate, onLogout }) {
  const role = user.role === 'caregiver' ? 'Caregiver' : ROLE_LABEL[user.account_type]
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <Brand />
        <nav className="nav" aria-label="Main">
          {pages.map(([id, label]) => (
            <button
              key={id}
              className={page === id ? 'active' : ''}
              aria-current={page === id ? 'page' : undefined}
              onClick={() => onNavigate(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="who">
          <span className="who-name">{user.name}</span>
          <span className="who-role">{role}</span>
          <button className="btn ghost small" onClick={onLogout}>Log out</button>
        </div>
      </div>
    </header>
  )
}
