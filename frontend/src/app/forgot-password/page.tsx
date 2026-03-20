'use client'
import { useState } from 'react'
import Link from 'next/link'
import { authApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Music2, Loader2, ArrowLeft } from 'lucide-react'

export default function ForgotPasswordPage() {
  const [emailOrUsername, setEmailOrUsername] = useState('')
  const [loading, setLoading] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!emailOrUsername.trim()) return toast.error('Please enter your email or username')
    setLoading(true)
    try {
      await authApi.forgotPassword(emailOrUsername.trim())
      setSubmitted(true)
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Something went wrong')
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
          <h1 className="font-display text-4xl text-white">FORGOT PASSWORD</h1>
          <p className="text-forge-muted mt-2">We'll send you a reset link</p>
        </div>

        <div className="card p-8">
          {submitted ? (
            <div className="text-center space-y-4">
              <div className="w-14 h-14 rounded-full bg-green-500/20 border border-green-500/30 flex items-center justify-center mx-auto">
                <span className="text-2xl">✓</span>
              </div>
              <p className="text-forge-text">If an account with this email/username exists, you'll receive a reset link.</p>
              <Link href="/login" className="inline-flex items-center gap-2 text-forge-accent hover:underline text-sm">
                <ArrowLeft size={14} />
                Back to login
              </Link>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="label-forge">Email or Username</label>
                <input
                  type="text"
                  value={emailOrUsername}
                  onChange={e => setEmailOrUsername(e.target.value)}
                  className="input-forge"
                  placeholder="you@example.com or your_username"
                  required
                  autoFocus
                />
              </div>
              <button type="submit" disabled={loading} className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-50">
                {loading ? <Loader2 size={16} className="animate-spin" /> : null}
                Send Reset Link
              </button>
              <p className="text-center text-forge-muted text-sm">
                <Link href="/login" className="text-forge-accent hover:underline inline-flex items-center gap-1">
                  <ArrowLeft size={13} />
                  Back to login
                </Link>
              </p>
            </form>
          )}
        </div>
      </div>
    </div>
  )
}
