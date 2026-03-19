import { create } from 'zustand'

export interface User {
  id: string
  email: string
  username: string
  display_name: string
  avatar_url: string | null
  role: 'creator' | 'admin' | 'moderator'
  balance_cents: number
  stripe_account_id?: string
  subscription_plan: 'free' | 'pro'
  subscription_status: 'active' | 'canceled' | 'past_due' | 'trialing'
  subscription_expires_at?: string | null
  beats_generated_count: number
  is_active?: boolean
}

interface AuthState {
  user: User | null
  token: string | null
  isLoading: boolean
  setUser: (user: User | null) => void
  setToken: (token: string | null) => void
  logout: () => void
  initialize: () => void
  refresh: () => Promise<void>
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isLoading: true,

  setUser: (user) => set({ user }),
  setToken: (token) => {
    if (token) {
      localStorage.setItem('bf_token', token)
    } else {
      localStorage.removeItem('bf_token')
    }
    set({ token })
  },

  logout: () => {
    localStorage.removeItem('bf_token')
    set({ user: null, token: null })
  },

  initialize: async () => {
    const { user, isLoading } = useAuthStore.getState()
    const token = localStorage.getItem('bf_token')

    if (!token) {
      set({ user: null, token: null, isLoading: false })
      return
    }

    if (user) {
      if (isLoading) set({ isLoading: false })
      return
    }

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        set({ user: data.user, token, isLoading: false })
      } else {
        localStorage.removeItem('bf_token')
        set({ user: null, token: null, isLoading: false })
      }
    } catch {
      set({ isLoading: false })
    }
  },

  refresh: async () => {
    const token = localStorage.getItem('bf_token')
    if (!token) return
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/me`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (res.ok) {
        const data = await res.json()
        set({ user: data.user, token })
      }
    } catch {}
  },
}))
