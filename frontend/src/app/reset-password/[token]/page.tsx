'use client'
import { useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { authApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Music2, Loader2, Eye, EyeOff } from 'lucide-react'

export default function ResetPasswordPage() {
  const { token } = useParams<{ token: string }>()
  const router = useRouter()
  const [form, setForm] = useState({ newPassword: '', confirmPassword: '' })
  const [showPw, setShowPw] = useState({ new: false, confirm: false })
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (form.newPassword !== form.confirmPassword) return toast.error('Passwords do not match')
    if (form.newPassword.length < 8) return toast.error('Password must be at least 8 characters')
    setLoading(true)
    try {
      await authApi.resetPassword({ token, newPassword: form.newPassword })
      toast.success('Password reset! Please login.')
      router.push('/login')
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Reset failed — link may have expired')
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
          <h1 className="font-display text-4xl text-white">RESET PASSWORD</h1>
          <p className="text-forge-muted mt-2">Choose a new password</p>
        </div>

        <form onSubmit={handleSubmit} className="card p-8 space-y-4">
          <div>
            <label className="label-forge">New Password</label>
            <div className="relative">
              <input
                type={showPw.new ? 'text' : 'password'}
                value={form.newPassword}
                onChange={e => setForm(f => ({ ...f, newPassword: e.target.value }))}
                className="input-forge pr-10"
                placeholder="Min. 8 characters"
                minLength={8}
                required
                autoFocus
              />
              <button type="button" onClick={() => setShowPw(s => ({ ...s, new: !s.new }))}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                {showPw.new ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </div>
          <div>
            <label className="label-forge">Confirm New Password</label>
            <div className="relative">
              <input
                type={showPw.confirm ? 'text' : 'password'}
                value={form.confirmPassword}
                onChange={e => setForm(f => ({ ...f, confirmPassword: e.target.value }))}
                className="input-forge pr-10"
                placeholder="Repeat new password"
                required
              />
              <button type="button" onClick={() => setShowPw(s => ({ ...s, confirm: !s.confirm }))}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                {showPw.confirm ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
            {form.confirmPassword && form.newPassword !== form.confirmPassword && (
              <p className="text-forge-accent text-xs mt-1">Passwords do not match</p>
            )}
          </div>
          <button type="submit" disabled={loading} className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-50">
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            Reset Password
          </button>
        </form>
      </div>
    </div>
  )
}
