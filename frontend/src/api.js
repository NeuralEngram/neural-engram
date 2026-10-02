// All calls to the FastAPI backend live here.
// Every request carries the access key (X-API-Key); requests made after
// login also carry the session token (Authorization: Bearer …).

const KEY_STORE = 'maem.accessKey'
const TOKEN_STORE = 'maem.token'

function read(name) {
  try { return localStorage.getItem(name) || '' } catch { return '' }
}
function write(name, value) {
  try {
    if (value) localStorage.setItem(name, value)
    else localStorage.removeItem(name)
  } catch { /* storage unavailable — keep in memory only */ }
}

let accessKey = read(KEY_STORE)
let token = read(TOKEN_STORE)
let onSessionExpired = () => {}

export const session = {
  get accessKey() { return accessKey },
  get token() { return token },
  setAccessKey(k) { accessKey = k; write(KEY_STORE, k) },
  setToken(t) { token = t; write(TOKEN_STORE, t) },
  clearToken() { token = ''; write(TOKEN_STORE, '') },
  forgetAccessKey() { accessKey = ''; write(KEY_STORE, '') },
  onExpired(fn) { onSessionExpired = fn },
}

export class ApiError extends Error {
  constructor(status, message) {
    super(message)
    this.status = status
  }
}

async function request(method, path, { body, params } = {}) {
  const url = new URL(path, window.location.origin)
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v)
  })
  const headers = { 'X-API-Key': accessKey }
  if (token) headers.Authorization = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  let res
  try {
    res = await fetch(url, {
      method,
      headers,
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
  } catch {
    throw new ApiError(0, "Can't reach the server. Check that it is running.")
  }

  let data = null
  try { data = await res.json() } catch { /* empty body */ }

  if (!res.ok) {
    let message = data?.detail
    if (Array.isArray(message)) message = message.map((d) => d.msg).join('; ')
    if (!message) {
      message = res.status === 429
        ? 'Too many requests. Wait a minute and try again.'
        : `Request failed (${res.status}).`
    }
    if (res.status === 401 && token && path !== '/app/auth/login') {
      onSessionExpired(message)
    }
    throw new ApiError(res.status, message)
  }
  return data
}

const get = (path, params) => request('GET', path, { params })
const post = (path, body = {}) => request('POST', path, { body })
const put = (path, body) => request('PUT', path, { body })
const patch = (path, body) => request('PATCH', path, { body })
const del = (path) => request('DELETE', path)

export const api = {
  // access check: any authenticated-by-key call works; this one is cheap
  checkKey: () => get('/app/auth/caregivers'),

  // accounts
  login: (username, password) => post('/app/auth/login', { username, password }),
  register: (form) => post('/app/auth/register', form),
  caregivers: () => get('/app/auth/caregivers'),
  me: () => get('/app/auth/me'),
  changePassword: (current_password, new_password) =>
    post('/app/auth/password', { current_password, new_password }),

  // chat & memories
  chat: (message) => post('/app/chat', { message }),
  memories: (user) => get('/app/memories', { user }),
  addMemory: (text, important, user) => post('/app/memories', { text, important, user }),
  reinforce: (id) => post(`/app/memories/${id}/reinforce`),

  // reminders
  reminders: (user) => get('/app/reminders', { user }),
  addReminder: (title, due_at, user) => post('/app/reminders', { title, due_at, user }),
  setReminderDone: (id, done) => patch(`/app/reminders/${id}`, { done }),
  deleteReminder: (id) => del(`/app/reminders/${id}`),

  today: (user) => get('/app/today', { user }),

  // caregiver
  patients: () => get('/app/patients'),
  alerts: (user, status) => get('/app/alerts', { user, status }),
  handleAlert: (id) => post(`/app/alerts/${id}/handle`),
  trend: (user, days = 14) => get('/app/trend', { user, days }),
  notifications: () => get('/app/settings/notifications'),
  saveNotifications: (email_enabled, email) =>
    put('/app/settings/notifications', { email_enabled, email }),

  // memory tools
  simulate: (hours, user) => post('/app/tools/simulate', { hours, user }),
  compare: () => get('/app/tools/compare'),

  // reset
  reset: (user) => post('/app/reset', { user }),
  resetAll: (api_key) => post('/app/reset-all', { api_key }),
}
