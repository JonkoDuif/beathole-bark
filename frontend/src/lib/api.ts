import axios from 'axios'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'

const api = axios.create({
  baseURL: `${API_URL}/api`,
  headers: { 'Content-Type': 'application/json' },
})

// Attach token on every request
api.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('bf_token')
    if (token) config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 - auto logout
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401 && typeof window !== 'undefined') {
      localStorage.removeItem('bf_token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default api

// ─── Auth ────────────────────────────────────────────────────────────────────
export const authApi = {
  register: (data: { email: string; password: string; username: string; displayName?: string }) =>
    api.post('/auth/register', data),
  login: (data: { email: string; password: string }) =>
    api.post('/auth/login', data),
  me: () => api.get('/auth/me'),
  updateProfile: (data: object) => api.put('/auth/profile', data),
  forgotPassword: (emailOrUsername: string) => api.post('/auth/forgot-password', { emailOrUsername }),
  resetPassword: (data: { token: string; newPassword: string }) => api.post('/auth/reset-password', data),
  confirmEmailChange: (token: string) => api.post(`/auth/confirm-email-change/${token}`, {}),
  confirmPasswordChange: (token: string) => api.post(`/auth/confirm-password-change/${token}`, {}),
  verify2fa: (data: { userId: string; code: string }) => api.post('/auth/2fa/verify', data),
}

// ─── Beats ───────────────────────────────────────────────────────────────────
export const beatsApi = {
  generate: (data: {
    genre?: string; mood?: string; bpm?: number; style?: string; title?: string
    key?: string; prompt?: string; beat_type?: 'audio' | 'midi'; duration?: number
    referenceAudio?: string; referenceStrength?: number
  }) => api.post('/beats/generate', data),
  status: (beatId: string) => api.get(`/beats/status/${beatId}`),
  generationStatus: () => api.get('/beats/generation-status'),
  list: (params?: object) => api.get('/beats', { params }),
  get: (id: string) => api.get(`/beats/${id}`),
  myBeats: (params?: object) => api.get('/beats/my-beats', { params }),
  update: (id: string, data: object) => api.put(`/beats/${id}`, data),
  delete: (id: string) => api.delete(`/beats/${id}`),
  publish: (id: string) => api.post(`/beats/${id}/publish`),
  unpublish: (id: string) => api.post(`/beats/${id}/unpublish`),
  create: (data: object) => api.post('/beats', data),
  uploadCoverArt: (id: string, imageBase64: string) =>
    api.post(`/beats/${id}/cover-art`, { imageBase64 }),
}

// ─── Orders / Beat purchases ─────────────────────────────────────────────────
export const ordersApi = {
  checkout: (data: { beatId: string; licenseId: string }) =>
    api.post('/orders/checkout', data),
  success: (sessionId: string) => api.get(`/orders/success?session_id=${sessionId}`),
  myPurchases: () => api.get('/orders/my-purchases'),
}

// ─── Subscriptions ────────────────────────────────────────────────────────────
export const subscriptionsApi = {
  status: () => api.get('/subscriptions/status'),
  checkout: () => api.post('/subscriptions/checkout'),
  cancel: () => api.post('/subscriptions/cancel'),
  portal: () => api.post('/subscriptions/portal'),
}

// ─── Downloads (My Library) ───────────────────────────────────────────────────
export const downloadsApi = {
  myLibrary: () => api.get('/downloads/my-library'),

  /** Returns a URL to trigger the file download */
  beatDownloadUrl: (orderId: string, format: 'mp3' | 'wav') =>
    `${API_URL}/api/downloads/beat/${orderId}/${format}`,

  licensePdfUrl: (orderId: string) =>
    `${API_URL}/api/downloads/license/${orderId}`,

  /** Trigger an authenticated file download in the browser */
  downloadFile: async (url: string, filename: string) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('bf_token') : null
    const res = await fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
    if (!res.ok) throw new Error('Download failed')
    const blob = await res.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(a.href)
  },
}

// ─── Dashboard ───────────────────────────────────────────────────────────────
export const dashboardApi = {
  overview: () => api.get('/dashboard/overview'),
  beats: (params?: object) => api.get('/dashboard/beats', { params }),
  sales: (params?: object) => api.get('/dashboard/sales', { params }),
  withdraw: (amountCents: number) => api.post('/dashboard/withdraw', { amountCents }),
  connectStripe: () => api.post('/dashboard/connect-stripe'),
}

// ─── Profile & Account ───────────────────────────────────────────────────────
export const profileApi = {
  getPublic: (username: string) => api.get(`/auth/profile/${username}`),
}

export const accountApi = {
  changeEmail: (data: { newEmail: string; currentPassword: string }) =>
    api.put('/auth/change-email', data),
  changeUsername: (data: { username: string; displayName?: string }) =>
    api.put('/auth/change-username', data),
  changePassword: (data: { currentPassword: string; newPassword: string; confirmPassword: string }) =>
    api.put('/auth/change-password', data),
  requestEmailChange: (data: { newEmail: string; currentPassword: string }) =>
    api.post('/auth/request-email-change', data),
  requestPasswordChange: (data: { currentPassword: string; newPassword: string; confirmPassword: string }) =>
    api.post('/auth/request-password-change', data),
  toggle2fa: (enabled: boolean) => api.post('/auth/2fa/toggle', { enabled }),
}

// ─── Support ─────────────────────────────────────────────────────────────────
export const supportApi = {
  createTicket: (data: { category: string; message: string }) =>
    api.post('/support/tickets', data),
  myTickets: () => api.get('/support/tickets'),
  getTicket: (id: string) => api.get(`/support/tickets/${id}`),
  reply: (id: string, message: string) =>
    api.post(`/support/tickets/${id}/reply`, { message }),
  updateStatus: (id: string, status: string) =>
    api.put(`/support/tickets/${id}/status`, { status }),
  adminTickets: (params?: object) => api.get('/support/admin/tickets', { params }),
}

// ─── Admin ───────────────────────────────────────────────────────────────────
export const adminApi = {
  analytics: (period?: number) => api.get(`/admin/analytics?period=${period || 30}`),
  beats: (params?: object) => api.get('/admin/beats', { params }),
  removeBeat: (id: string) => api.delete(`/admin/beats/${id}`),
  users: (params?: object) => api.get('/admin/users', { params }),
  setUserRole: (id: string, role: 'admin' | 'creator') => api.put(`/admin/users/${id}/role`, { role }),
  setUserSubscription: (id: string, plan: 'free' | 'pro') => api.put(`/admin/users/${id}/subscription`, { plan }),
  activateUser: (id: string) => api.put(`/admin/users/${id}/activate`),
  deactivateUser: (id: string) => api.put(`/admin/users/${id}/deactivate`),
  orders: (params?: object) => api.get('/admin/orders', { params }),
  getSettings: () => api.get('/admin/settings'),
  updateSettings: (data: object) => api.put('/admin/settings', data),
  giveCredits: (userId: string, amount: number) => api.post(`/admin/users/${userId}/give-credits`, { amount }),
}

// ─── Studio ──────────────────────────────────────────────────────────────────
export const studioApi = {
  getProject: (beatId: string) => api.get(`/studio/${beatId}/project`),
  saveVersion: (beatId: string, data: { project_data: object; label?: string }) =>
    api.post(`/studio/${beatId}/versions`, data),
  listVersions: (beatId: string) => api.get(`/studio/${beatId}/versions`),
  getVersion: (beatId: string, versionId: string) =>
    api.get(`/studio/${beatId}/versions/${versionId}`),
  invite: (beatId: string, username: string) =>
    api.post(`/studio/${beatId}/invite`, { username }),
  acceptInvite: (beatId: string) =>
    api.post(`/studio/${beatId}/invite/accept`),
  declineInvite: (beatId: string) =>
    api.post(`/studio/${beatId}/invite/decline`),
  pendingInvitations: () => api.get('/studio/invitations/pending'),
  collaborators: (beatId: string) => api.get(`/studio/${beatId}/collaborators`),
  myCollabs: () => api.get('/studio/my-collabs'),
  kickCollaborator: (beatId: string, userId: string) =>
    api.delete(`/studio/${beatId}/collaborators/${userId}`),
}

// ─── Presets ─────────────────────────────────────────────────────────────────
export const presetsApi = {
  list: () => api.get('/presets'),
  save: (data: { name: string; data: object }) => api.post('/presets', data),
  update: (id: string, data: object) => api.put(`/presets/${id}`, data),
  delete: (id: string) => api.delete(`/presets/${id}`),
  share: (id: string) => api.post(`/presets/${id}/share`),
  getShared: (token: string) => api.get(`/presets/share/${token}`),
  saveShared: (token: string) => api.post(`/presets/share/${token}/save`),
}

// ─── Notifications ────────────────────────────────────────────────────────────
export const notificationsApi = {
  list: () => api.get('/notifications'),
  markRead: (id: string) => api.put(`/notifications/${id}/read`),
  markAllRead: () => api.put('/notifications/read-all'),
  delete: (id: string) => api.delete(`/notifications/${id}`),
}

// ─── Credits ─────────────────────────────────────────────────────────────────
export const creditsApi = {
  packages: () => api.get('/orders/credit-packages'),
  buyCredits: (packageId: string) => api.post('/orders/buy-credits', { packageId }),
  confirmPurchase: (sessionId: string) => api.get(`/orders/credits-success?session_id=${sessionId}`),
}
