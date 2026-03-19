'use client'
import { useEffect, useState, Suspense } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { Crown, Zap, Check, Loader2 } from 'lucide-react'
import { useAuthStore } from '@/store/auth'

export default function SubscriptionSuccessPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-forge-black" />}>
      <SubscriptionSuccessContent />
    </Suspense>
  )
}

function SubscriptionSuccessContent() {
  const { setUser } = useAuthStore()
  const searchParams = useSearchParams()
  const [checking, setChecking] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const activate = async () => {
      const token = localStorage.getItem('bf_token')
      const sessionId = searchParams.get('session_id')

      if (!token) {
        setError('You are not signed in. Please sign in and check your dashboard.')
        setChecking(false)
        return
      }

      // Wait 2 seconds to give the webhook a chance to fire
      await new Promise(r => setTimeout(r, 2000))

      // Fallback: activeer subscription direct via session_id
      if (sessionId) {
        try {
          await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/subscriptions/verify-session`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({ sessionId }),
          })
        } catch {}
      }

      // Refresh user data
      try {
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          setUser(data.user)
        }
      } catch {}

      setChecking(false)
    }

    activate()
  }, [])

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-lg mx-auto px-4 py-24 pt-40 text-center">
        <div className="w-24 h-24 rounded-full bg-forge-gold/20 border-2 border-forge-gold flex items-center justify-center mx-auto mb-8">
          {checking ? (
            <Loader2 size={40} className="text-forge-gold animate-spin" />
          ) : (
            <Crown size={40} className="text-forge-gold" />
          )}
        </div>

        <h1 className="font-display text-4xl text-white mb-3 tracking-wider">
          {checking ? 'ACTIVATING...' : error ? 'PAYMENT RECEIVED' : 'WELCOME TO PRO!'}
        </h1>

        <p className="text-forge-muted mb-10">
          {checking
            ? 'Activating your subscription, just a moment...'
            : error
            ? error
            : 'Your subscription is active. You can now generate 500 beats per month, publish to the marketplace, and sell with full licensing.'}
        </p>

        {!checking && !error && (
          <div className="card p-6 mb-8 text-left space-y-3">
            {[
              '500 AI beats per month',
              'Publish to marketplace',
              'Sell with 4 license tiers',
              '80% revenue on every sale',
              'Radio, stage, sync rights',
              'PDF license documents',
            ].map((item) => (
              <div key={item} className="flex items-center gap-3 text-sm text-forge-text">
                <Check size={16} className="text-forge-green flex-shrink-0" />
                {item}
              </div>
            ))}
          </div>
        )}

        {!checking && (
          <div className="flex gap-3">
            <Link href="/generate" className="flex-1 btn-primary py-3 flex items-center justify-center gap-2">
              <Zap size={16} />
              Start Generating
            </Link>
            <Link href="/dashboard" className="flex-1 btn-secondary py-3 text-center">
              Dashboard
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
