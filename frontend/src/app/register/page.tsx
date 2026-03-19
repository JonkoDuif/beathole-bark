'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { useAuthStore } from '@/store/auth'
import { authApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Music2, Loader2, Eye, EyeOff, Check } from 'lucide-react'

export default function RegisterPage() {
  const router = useRouter()
  const { setUser, setToken } = useAuthStore()
  const [form, setForm] = useState({ email: '', password: '', username: '', displayName: '' })
  const [loading, setLoading] = useState(false)
  const [showPw, setShowPw] = useState(false)

  const pwStrength = [
    form.password.length >= 8,
    /[A-Z]/.test(form.password),
    /[0-9]/.test(form.password),
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (form.password.length < 8) return toast.error('Password must be at least 8 characters')
    setLoading(true)
    try {
      const res = await authApi.register(form)
      setUser(res.data.user)
      setToken(res.data.token)
      toast.success('Account created! Start forging beats 🎵')
      router.push('/generate')
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Registration failed')
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
          <h1 className="font-display text-4xl text-white">CREATE ACCOUNT</h1>
          <p className="text-forge-muted mt-2">Start earning from your beats today</p>
        </div>

        <form onSubmit={handleSubmit} className="card p-8 space-y-4">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="label-forge">Username</label>
              <input
                type="text"
                value={form.username}
                onChange={e => setForm(f => ({ ...f, username: e.target.value.toLowerCase().replace(/\s/g, '') }))}
                className="input-forge"
                placeholder="producer_x"
                required
              />
            </div>
            <div>
              <label className="label-forge">Display Name</label>
              <input
                type="text"
                value={form.displayName}
                onChange={e => setForm(f => ({ ...f, displayName: e.target.value }))}
                className="input-forge"
                placeholder="Producer X"
              />
            </div>
          </div>
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
                placeholder="Min. 8 characters"
                required
                minLength={8}
              />
              <button type="button" onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {form.password && (
              <div className="flex gap-2 mt-2">
                {['8+ chars', 'Uppercase', 'Number'].map((label, i) => (
                  <span key={label} className={`text-xs flex items-center gap-1 ${pwStrength[i] ? 'text-forge-green' : 'text-forge-muted'}`}>
                    <Check size={10} />
                    {label}
                  </span>
                ))}
              </div>
            )}
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-50">
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            Create Account
          </button>

          <div className="text-center text-xs text-forge-muted">
            By signing up you agree to our Terms & Privacy Policy.<br />
            <span className="text-forge-green font-medium">80% of all beat sales go directly to you.</span>
          </div>

          <p className="text-center text-forge-muted text-sm">
            Have an account?{' '}
            <Link href="/login" className="text-forge-accent hover:underline">Sign in</Link>
          </p>
        </form>
      </div>
    </div>
  )
}
