/* ============================================
   RepoChat — API Client
   ============================================
   Centralized API calls. In dev, Vite proxies /api
   to the backend. In production, set VITE_API_BASE.
   ============================================ */

const API_BASE = import.meta.env.VITE_API_BASE || '';
const TOKEN_KEY = 'repochat_jwt_token';

export function clearAuthToken() {
  localStorage.removeItem(TOKEN_KEY);
}

/** Decode JWT exp (seconds). Returns null if missing or malformed. */
function getTokenExpiry(token) {
  try {
    const payload = token.split('.')[1];
    const json = atob(payload.replace(/-/g, '+').replace(/_/g, '/'));
    const data = JSON.parse(json);
    return typeof data.exp === 'number' ? data.exp : null;
  } catch {
    return null;
  }
}

export function isTokenExpired(token) {
  const exp = getTokenExpiry(token);
  if (exp === null) return true;
  return Date.now() / 1000 >= exp;
}

let authRedirectPending = false;

function handleUnauthorized() {
  if (authRedirectPending) return;
  authRedirectPending = true;
  clearAuthToken();
  window.location.reload();
}

/**
 * Fetch wrapper that injects the JWT and clears stale sessions on 401.
 */
function apiFetch(path, options = {}) {
  const token = localStorage.getItem(TOKEN_KEY);
  const headers = { ...options.headers };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  return fetch(`${API_BASE}${path}`, { ...options, headers }).then((res) => {
    if (res.status === 401) {
      handleUnauthorized();
    }
    return res;
  });
}

/** Confirm token is valid and the user still exists in the database. */
export async function validateSession() {
  const token = localStorage.getItem(TOKEN_KEY);
  if (!token || isTokenExpired(token)) {
    throw new Error('Session expired');
  }
  const res = await fetch(`${API_BASE}/api/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (res.status === 401) {
    throw new Error('Session invalid');
  }
  if (res.status === 404) {
    throw new Error('Auth endpoint not found — rebuild frontend (docker compose up --build frontend)');
  }
  if (!res.ok) {
    throw new Error('Could not validate session');
  }
  return res.json();
}

/** List all repos */
export async function fetchRepos() {
  const res = await apiFetch('/api/repos');
  if (!res.ok) throw new Error('Failed to fetch repos');
  return res.json();
}

/** Login and store JWT token */
export async function login(username, password) {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const res = await fetch(`${API_BASE}/api/auth/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: formData.toString(),
  });

  if (!res.ok) throw new Error('Login failed');
  const data = await res.json();
  localStorage.setItem(TOKEN_KEY, data.access_token);
  return data;
}

export async function register(username, password) {
  const res = await fetch(`${API_BASE}/api/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  if (!res.ok) throw new Error('Registration failed. Username might be taken.');
  const data = await res.json();
  localStorage.setItem(TOKEN_KEY, data.access_token);
  return data;
}

/** Add a repo */
export async function addRepo(githubUrl) {
  const res = await apiFetch('/api/repos', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ github_url: githubUrl }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to add repo');
  }
  return res.json();
}

/** Delete a repo */
export async function deleteRepo(repoId) {
  const res = await apiFetch(`/api/repos/${repoId}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to delete repo');
  }
  return res.json();
}

/** Get repo status */
export async function getRepoStatus(repoId) {
  const res = await apiFetch(`/api/repos/${repoId}/status`);
  if (!res.ok) throw new Error('Failed to fetch repo status');
  return res.json();
}

/**
 * Stream chat via SSE.
 * Returns a ReadableStream reader that yields SSE lines.
 */
export async function chatStream(repoId, query) {
  const params = new URLSearchParams({ repo_id: repoId, query });
  const token = localStorage.getItem(TOKEN_KEY);
  const headers = {};
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_BASE}/api/chat/stream?${params}`, { headers });
  if (res.status === 401) {
    handleUnauthorized();
    throw new Error('Session expired');
  }
  if (!res.ok) throw new Error(`Server error: ${res.status}`);
  return res.body.getReader();
}

export async function fetchMessages(repoId, skip = 0, limit = 20) {
  const res = await apiFetch(
    `/api/repos/${repoId}/messages?skip=${skip}&limit=${limit}`
  );
  if (!res.ok) throw new Error('Failed to fetch messages');
  return res.json();
}

export function getGoogleLoginUrl() {
  return `${API_BASE}/api/auth/google/login`;
}
