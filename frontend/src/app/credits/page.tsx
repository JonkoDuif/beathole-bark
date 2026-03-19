'use client'
import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { useAuthStore } from '@/store/auth'
import Navbar from '@/components/Navbar'
import { Zap, Crown, Check, Loader2 } from 'lucide-react'
import toast from 'react-hot-toast'

interface CreditPackage {
  id: string
  credits: number
  price_cents: number
  label: string
}

export default function CreditsPage() {
  const { user, isLoading } = useAuthStore()
  const router = useRouter()
  const [packages, setPackages] = useState<CreditPackage[]>([])
  const [loading, setLoading] = useState(true)
  const [buying, setBuying] = useState<string | null>(null)

  useEffect(() => {
    if (!isLoading && !user) router.push('/login?redirect=/credits')
  }, [user, isLoading])

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'}/api/orders/credit-packages`)
      .then(r => r.json())
      .then(setPackages)
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [])

  const buyCredits = async (pkg: CreditPackage) => {
    setBuying(pkg.id)
    try {
      const token = localStorage.getItem('bf_token')
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'}/api/orders/buy-credits`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ packageId: pkg.id }),
      })
      const data = await res.json()
      if (data.url) {
        window.location.href = data.url
      } else {
        toast.error(data.error || 'Failed to start checkout')
      }
    } catch {
      toast.error('Something went wrong')
    } finally {
      setBuying(null)
    }
  }

  const isPro = user?.subscription_plan === 'pro' && user?.subscription_status === 'active'

  const PACKAGE_HIGHLIGHTS: Record<string, string> = {
    credits_50: '',
    credits_250: 'BEST VALUE',
    credits_1000: 'POWER USER',
  }

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-4xl mx-auto px-6 pt-28 pb-20">
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-forge-accent/10 border border-forge-accent/20 text-forge-accent text-xs font-bold uppercase tracking-widest mb-4">
            <Zap size={12} /> Beat Credits
          </div>
          <h1 className="text-4xl font-display font-bold text-forge-text mb-3">
            Buy Beat Credits
          </h1>
          <p className="text-forge-muted max-w-md mx-auto">
            One-time credit purchases that stack on top of your plan.
            {isPro ? ' As a Pro user, credits never expire.' : ' Upgrade to Pro for 500 monthly beats.'}
          </p>
          {user && (
            <div className="inline-flex items-center gap-2 mt-4 px-4 py-2 bg-forge-card border border-forge-border rounded-xl">
              <Zap size={14} className="text-forge-accent" />
              <span className="text-sm text-forge-muted">
                Remaining this cycle: <strong className="text-forge-text">{user.beats_generated_count !== undefined ? '–' : '–'}</strong>
              </span>
              {(user as any).extra_beat_credits > 0 && (
                <span className="text-sm text-forge-accent font-semibold ml-2">
                  + {(user as any).extra_beat_credits} bonus credits
                </span>
              )}
            </div>
          )}
        </div>

        {loading ? (
          <div className="flex justify-center py-12"><Loader2 size={24} className="text-forge-muted animate-spin" /></div>
        ) : (
          <div className="grid md:grid-cols-3 gap-6">
            {packages.map(pkg => {
              const highlight = PACKAGE_HIGHLIGHTS[pkg.id]
              const isPopular = pkg.id === 'credits_250'
              return (
                <div key={pkg.id} className={`relative rounded-2xl border p-6 flex flex-col gap-4 transition-all hover:scale-[1.02] ${isPopular ? 'border-forge-accent bg-forge-accent/5 shadow-forge' : 'border-forge-border bg-forge-card'}`}>
                  {highlight && (
                    <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                      <span className="px-3 py-1 bg-forge-accent text-white text-[10px] font-bold uppercase tracking-widest rounded-full">
                        {highlight}
                      </span>
                    </div>
                  )}
                  <div>
                    <div className="text-3xl font-display font-bold text-forge-text mb-1">
                      {pkg.credits}
                    </div>
                    <div className="text-sm text-forge-muted">beat credits</div>
                  </div>
                  <div>
                    <span className="text-2xl font-bold text-forge-text">€{(pkg.price_cents / 100).toFixed(2)}</span>
                    <span className="text-xs text-forge-muted ml-1">one-time</span>
                  </div>
                  <div className="text-xs text-forge-muted">
                    €{(pkg.price_cents / pkg.credits / 100 * 100).toFixed(1)} cents per beat
                  </div>
                  <ul className="space-y-1.5 flex-1">
                    <li className="flex items-center gap-2 text-xs text-forge-muted">
                      <Check size={12} className="text-forge-accent flex-shrink-0" />
                      Never expires
                    </li>
                    <li className="flex items-center gap-2 text-xs text-forge-muted">
                      <Check size={12} className="text-forge-accent flex-shrink-0" />
                      Stacks with your plan
                    </li>
                    <li className="flex items-center gap-2 text-xs text-forge-muted">
                      <Check size={12} className="text-forge-accent flex-shrink-0" />
                      All output modes (audio + MIDI)
                    </li>
                  </ul>
                  <button
                    onClick={() => buyCredits(pkg)}
                    disabled={!!buying}
                    className={`w-full py-2.5 rounded-xl font-semibold text-sm transition-all flex items-center justify-center gap-2 ${isPopular ? 'btn-primary' : 'btn-secondary'} disabled:opacity-50`}
                  >
                    {buying === pkg.id ? <Loader2 size={16} className="animate-spin" /> : <Zap size={16} />}
                    {buying === pkg.id ? 'Redirecting...' : `Buy ${pkg.credits} Credits`}
                  </button>
                </div>
              )
            })}
          </div>
        )}

        <div className="mt-12 p-6 rounded-2xl border border-forge-border bg-forge-card">
          <div className="flex items-center gap-3 mb-3">
            <Crown size={20} className="text-forge-gold" />
            <h2 className="text-lg font-bold text-forge-text">Want unlimited beats?</h2>
          </div>
          <p className="text-sm text-forge-muted mb-4">Pro plan gives you 500 beats per month, studio collaboration, presets, and more.</p>
          <a href="/pricing" className="btn-primary text-sm inline-flex items-center gap-2">
            <Crown size={14} /> View Pro Plan
          </a>
        </div>
      </div>
    </div>
  )
}
