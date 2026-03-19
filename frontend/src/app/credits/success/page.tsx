'use client'
import { useEffect, useState } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth'
import { Zap, Check, Loader2 } from 'lucide-react'
import Link from 'next/link'

export default function CreditsSuccessPage() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const { user } = useAuthStore()
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [credits, setCredits] = useState(0)

  useEffect(() => {
    const sessionId = searchParams.get('session_id')
    if (!sessionId) { router.push('/credits'); return }
    const token = localStorage.getItem('bf_token')
    if (!token) { router.push('/login'); return }
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'}/api/orders/credits-success?session_id=${sessionId}`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(r => r.json())
      .then(data => {
        if (data.credits) { setCredits(data.credits); setStatus('success') }
        else setStatus('error')
      })
      .catch(() => setStatus('error'))
  }, [])

  return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center px-6">
      <div className="max-w-md w-full text-center">
        {status === 'loading' && (
          <div className="space-y-4">
            <Loader2 size={48} className="text-forge-accent animate-spin mx-auto" />
            <p className="text-forge-muted">Confirming your purchase...</p>
          </div>
        )}
        {status === 'success' && (
          <div className="space-y-6">
            <div className="w-20 h-20 rounded-full bg-forge-accent/20 border-2 border-forge-accent flex items-center justify-center mx-auto">
              <Check size={40} className="text-forge-accent" />
            </div>
            <div>
              <h1 className="text-3xl font-display font-bold text-forge-text mb-2">Credits Added!</h1>
              <p className="text-forge-muted">{credits} beat credits have been added to your account.</p>
            </div>
            <div className="flex gap-3 justify-center">
              <Link href="/generate" className="btn-primary flex items-center gap-2"><Zap size={16} /> Generate Now</Link>
              <Link href="/credits" className="btn-secondary">Buy More</Link>
            </div>
          </div>
        )}
        {status === 'error' && (
          <div className="space-y-4">
            <p className="text-forge-accent text-lg font-bold">Something went wrong</p>
            <p className="text-forge-muted text-sm">Your payment may have succeeded. Please contact support if credits weren't added.</p>
            <Link href="/support" className="btn-secondary">Contact Support</Link>
          </div>
        )}
      </div>
    </div>
  )
}
