'use client'
import { useState, useEffect, Suspense } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useAuthStore } from '@/store/auth'
import { authApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Music2, Loader2 } from 'lucide-react'

export default function TwoFAPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-forge-black" />}>
      <TwoFAContent />
    </Suspense>
  )
}

function TwoFAContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { setUser, setToken } = useAuthStore()
  const [code, setCode] = useState('')
  const [loading, setLoading] = useState(false)
  const [userId, setUserId] = useState<string | null>(null)

  useEffect(() => {
    const stored = sessionStorage.getItem('2fa_userId')
    if (!stored) {
      router.push('/login')
      return
    }
    setUserId(stored)
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!userId || code.length !== 6) return toast.error('Please enter the 6-digit code')
    setLoading(true)
    try {
      const res = await authApi.verify2fa({ userId, code })
      sessionStorage.removeItem('2fa_userId')
      setUser(res.data.user)
      setToken(res.data.token)
      toast.success('Welcome back!')
      const returnUrl = searchParams.get('returnUrl') || '/generate'
      router.push(returnUrl)
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Invalid or expired code')
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
          <h1 className="font-display text-4xl text-white">VERIFY LOGIN</h1>
          <p className="text-forge-muted mt-2">Enter the 6-digit code sent to your email</p>
        </div>

        <form onSubmit={handleSubmit} className="card p-8 space-y-6">
          <div>
            <label className="label-forge text-center block mb-3">Verification Code</label>
            <input
              type="text"
              inputMode="numeric"
              pattern="[0-9]*"
              maxLength={6}
              value={code}
              onChange={e => setCode(e.target.value.replace(/\D/g, '').slice(0, 6))}
              className="input-forge text-center text-3xl font-mono tracking-widest"
              placeholder="000000"
              autoFocus
              required
            />
          </div>
          <button
            type="submit"
            disabled={loading || code.length !== 6}
            className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-50"
          >
            {loading ? <Loader2 size={16} className="animate-spin" /> : null}
            Verify Code
          </button>
          <p className="text-center text-forge-muted text-sm">
            <Link href="/login" className="text-forge-accent hover:underline">
              Back to login
            </Link>
          </p>
        </form>
      </div>
    </div>
  )
}
