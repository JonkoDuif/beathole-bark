'use client'
import { useState, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useAuthStore } from '@/store/auth'
import { authApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Music2, Loader2, Eye, EyeOff } from 'lucide-react'

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-forge-black" />}>
      <LoginContent />
    </Suspense>
  )
}

function LoginContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { setUser, setToken } = useAuthStore()
  const [form, setForm] = useState({ email: '', password: '' })
  const [loading, setLoading] = useState(false)
  const [showPw, setShowPw] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await authApi.login(form)
      if (res.data.requires2fa) {
        sessionStorage.setItem('2fa_userId', res.data.userId)
        router.push('/2fa')
        return
      }
      setUser(res.data.user)
      setToken(res.data.token)
      toast.success('Welcome back!')
      router.push(searchParams.get('redirect') || '/dashboard')
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-30" />
      <div className="relative w-full max-w-md">

        <div className="text-center mb-8">
          <Link href="/" className="inline-flex items-center gap-2 mb-6">
            <div className="w-10 h-10 bg-gradient-forge rounded-xl flex items-center justify-center">
              <Music2 size={20} className="text-white" />
            </div>
            <span className="font-display text-2xl">BEAT<span className="text-forge-accent">HOLE</span></span>
          </Link>
          <h1 className="font-display text-4xl text-white">SIGN IN</h1>
          <p className="text-forge-muted mt-2">Welcome back, producer</p>
        </div>

        <form onSubmit={handleSubmit} className="card p-8 space-y-4">
          <div>
            <label className="label-forge">Email</label>
            <input
              type="email"
              value={form.email}
              onChange={e => setForm(f => ({ ...f, email: e.target.value }))}
              className="input-forge"
              placeholder="you@example.com"
              required
            />
          </div>
          <div>
            <label className="label-forge">Password</label>
            <div className="relative">
              <input
                type={showPw ? 'text' : 'password'}
                value={form.password}
                onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                className="input-forge pr-10"
                placeholder="••••••••"
                required
              />
              <button type="button" onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          <div className="text-right">
            <Link href="/forgot-password" className="text-forge-muted text-sm hover:text-forge-accent transition-colors">
              Forgot password?
            </Link>
          </div>
          <button type="submit" disabled={loading} className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-50">
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            Sign In
          </button>
          <p className="text-center text-forge-muted text-sm">
            No account?{' '}
            <Link href="/register" className="text-forge-accent hover:underline">Create one</Link>
          </p>
        </form>
      </div>
    </div>
  )
}
