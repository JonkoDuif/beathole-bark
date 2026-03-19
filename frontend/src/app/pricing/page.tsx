'use client'
import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { subscriptionsApi } from '@/lib/api'
import api from '@/lib/api'
import { loadStripe } from '@stripe/stripe-js'
import toast from 'react-hot-toast'
import {
  Check, X, Zap, Crown, Music2, Radio, Tv, Mic2,
  Download, Banknote, ShieldCheck, Star, Sliders, Scissors, Mic
} from 'lucide-react'
import clsx from 'clsx'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

interface PlanFeature {
  label: string
  free: string | boolean
  pro: string | boolean
  icon: React.ReactNode
}

const FEATURES: PlanFeature[] = [
  { label: 'Beat Generations', free: '10 lifetime', pro: '500 / month', icon: <Zap size={16} /> },
  { label: 'AI Studio (Multi-track DAW)', free: '1 track', pro: 'Unlimited tracks', icon: <Sliders size={16} /> },
  { label: 'Studio Recording (Mic input)', free: true, pro: true, icon: <Mic size={16} /> },
  { label: 'Audio Editing (Cut, Trim, Fade)', free: true, pro: true, icon: <Scissors size={16} /> },
  { label: 'MIDI Piano Roll Editor', free: true, pro: true, icon: <Music2 size={16} /> },
  { label: 'Mix Export (WAV/MP3)', free: true, pro: true, icon: <Download size={16} /> },
  { label: 'Publish to Marketplace', free: false, pro: true, icon: <Music2 size={16} /> },
  { label: 'Sell Beats with Licenses', free: false, pro: true, icon: <ShieldCheck size={16} /> },
  { label: 'Earn from Sales (80% cut)', free: false, pro: true, icon: <Star size={16} /> },
  { label: 'Basic License (MP3)', free: false, pro: true, icon: <Download size={16} /> },
  { label: 'Standard License (WAV)', free: false, pro: true, icon: <Download size={16} /> },
  { label: 'Premium License + Stems', free: false, pro: true, icon: <Download size={16} /> },
  { label: 'Exclusive License', free: false, pro: true, icon: <Crown size={16} /> },
  { label: 'Radio Broadcasting Rights', free: false, pro: true, icon: <Radio size={16} /> },
  { label: 'Stage Performance Rights', free: false, pro: true, icon: <Mic2 size={16} /> },
  { label: 'Sync / TV / Film Rights', free: false, pro: true, icon: <Tv size={16} /> },
  { label: 'License PDF Downloads', free: false, pro: true, icon: <Download size={16} /> },
  { label: 'Stripe Creator Payouts', free: false, pro: true, icon: <Banknote size={16} /> },
]

const LICENSE_TIERS = [
  {
    name: 'Basic',
    color: 'text-forge-muted',
    border: 'border-forge-border',
    price: '€5',
    features: [
      'MP3 Download',
      'Up to 50,000 streams',
      '1 music video',
      'Non-commercial only',
    ],
    no: ['Radio', 'Stage', 'Sync/TV'],
  },
  {
    name: 'Standard',
    color: 'text-forge-cyan',
    border: 'border-forge-cyan/40',
    price: '€20',
    features: [
      'WAV + MP3 Download',
      'Up to 500,000 streams',
      '2 music videos',
      'Commercial use',
      'Non-commercial radio',
    ],
    no: ['Stage performances', 'Sync/TV'],
  },
  {
    name: 'Premium',
    color: 'text-forge-gold',
    border: 'border-forge-gold/40',
    price: '€50',
    features: [
      'WAV + MP3 + Stems',
      'Unlimited streams',
      'Unlimited music videos',
      'Radio & TV broadcasting',
      'Up to 5 live shows/year',
      'Sync licensing included',
    ],
    no: [],
  },
  {
    name: 'Exclusive',
    color: 'text-forge-accent',
    border: 'border-forge-accent/40',
    price: '€200',
    features: [
      'All file formats + Project',
      'Unlimited everything',
      'Full exclusive ownership',
      'Unlimited performances',
      'Beat removed from market',
      'No credit required',
    ],
    no: [],
  },
]

export default function PricingPage() {
  const { user } = useAuthStore()
  const router = useRouter()
  const [loading, setLoading] = useState(false)

  const isPro = user?.subscription_plan === 'pro' && user?.subscription_status === 'active'

  const handleManageSubscription = async () => {
    setLoading(true)
    try {
      const res = await api.post('/subscriptions/portal')
      window.location.href = res.data.url
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Could not open subscription portal')
    } finally {
      setLoading(false)
    }
  }

  const handleUpgrade = async () => {
    if (!user) {
      router.push('/register?redirect=/pricing')
      return
    }
    if (isPro) return

    setLoading(true)
    try {
      const res = await subscriptionsApi.checkout()
      const stripe = await stripePromise
      if (!stripe) throw new Error('Stripe not loaded')
      if (res.data.url) {
        window.location.href = res.data.url
      } else {
        await stripe.redirectToCheckout({ sessionId: res.data.sessionId })
      }
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Could not start checkout')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-6xl mx-auto px-4 py-24 pt-32">

        {/* Header */}
        <div className="text-center mb-16">
          <span className="inline-block text-xs font-mono text-forge-accent border border-forge-accent/30 px-3 py-1 rounded-full mb-4 uppercase tracking-widest">
            Plans & Pricing
          </span>
          <h1 className="font-display text-5xl md:text-6xl text-white mb-4 tracking-wider">
            START FOR FREE.<br />
            <span className="text-forge-accent">SELL WITH PRO.</span>
          </h1>
          <p className="text-forge-muted text-lg max-w-xl mx-auto">
            Generate beats with AI. Upgrade to Pro to publish to the marketplace,
            sell licenses, and earn 80% on every sale.
          </p>
        </div>

        {/* Plan Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-3xl mx-auto mb-20">

          {/* Free Plan */}
          <div className="card p-8 flex flex-col">
            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <Music2 size={18} className="text-forge-muted" />
                <span className="font-semibold text-forge-text text-sm uppercase tracking-widest">Free</span>
              </div>
              <div className="text-5xl font-bold text-white mb-1">€0</div>
              <p className="text-forge-muted text-sm">Forever free</p>
            </div>

            <ul className="space-y-3 flex-1 mb-8">
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>10 AI beat generations (lifetime)</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Personal use beats</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>MP3 download of your own beats</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>AI Studio with recording & MIDI editor</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-muted line-through">
                <X size={14} className="text-forge-accent/50 flex-shrink-0" />
                <span>Marketplace publishing</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-muted line-through">
                <X size={14} className="text-forge-accent/50 flex-shrink-0" />
                <span>Sell beats with licenses</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-muted line-through">
                <X size={14} className="text-forge-accent/50 flex-shrink-0" />
                <span>Creator earnings</span>
              </li>
            </ul>

            <button
              disabled={!isPro}
              onClick={isPro ? handleManageSubscription : undefined}
              className={clsx(
                'w-full py-3 rounded-xl text-sm transition-all',
                isPro
                  ? 'border border-forge-border text-forge-muted hover:border-forge-text hover:text-forge-text cursor-pointer'
                  : 'border border-forge-border text-forge-muted cursor-not-allowed'
              )}
            >
              {isPro ? 'Downgrade to Free' : user ? 'Current Plan' : 'Get Started Free'}
            </button>
          </div>

          {/* Pro Plan */}
          <div className={clsx(
            'relative card p-8 flex flex-col border-2',
            isPro ? 'border-forge-green' : 'border-forge-accent'
          )}>
            {/* Badge */}
            <div className={clsx(
              'absolute -top-3 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full text-xs font-bold uppercase tracking-widest text-white',
              isPro ? 'bg-forge-green' : 'bg-gradient-forge'
            )}>
              {isPro ? '✓ Active' : 'Most Popular'}
            </div>

            <div className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <Crown size={18} className="text-forge-gold" />
                <span className="font-semibold text-white text-sm uppercase tracking-widest">Pro</span>
              </div>
              <div className="flex items-end gap-2 mb-1">
                <span className="text-5xl font-bold text-white">€19.99</span>
                <span className="text-forge-muted text-sm mb-2">/month</span>
              </div>
              <p className="text-forge-muted text-sm">Cancel anytime</p>
            </div>

            <ul className="space-y-3 flex-1 mb-8">
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span><strong className="text-white">500 beats/month</strong> (resets monthly)</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Full AI Studio — unlimited tracks, effects &amp; mixing</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Multi-track recording with live monitoring</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Publish to marketplace</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Sell with 4 license tiers</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span><strong className="text-white">80% revenue</strong> on every sale</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>WAV + MP3 + Stems + Exclusive licenses</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Radio, stage, sync rights on licenses</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>Stripe Connect payouts</span>
              </li>
              <li className="flex items-center gap-2 text-sm text-forge-text">
                <Check size={14} className="text-forge-green flex-shrink-0" />
                <span>PDF license documents</span>
              </li>
            </ul>

            {isPro ? (
              <div className="space-y-2">
                <div className="w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 bg-forge-green/20 text-forge-green border border-forge-green">
                  <Check size={16} /> Pro Active
                </div>
                <button
                  onClick={handleManageSubscription}
                  disabled={loading}
                  className="w-full py-2 rounded-xl text-sm text-forge-muted border border-forge-border hover:border-forge-text hover:text-forge-text transition-all"
                >
                  {loading ? '...' : 'Manage Subscription'}
                </button>
              </div>
            ) : (
              <button
                onClick={handleUpgrade}
                disabled={loading}
                className="w-full py-3 rounded-xl font-bold text-sm flex items-center justify-center gap-2 btn-primary shadow-forge"
              >
                {loading ? <span className="animate-spin">⟳</span> : <><Crown size={16} /> Upgrade to Pro — €19.99/month</>}
              </button>
            )}
          </div>
        </div>

        {/* Feature comparison table */}
        <div className="max-w-3xl mx-auto mb-20">
          <h2 className="font-display text-2xl text-white text-center mb-8 tracking-wider">FULL COMPARISON</h2>
          <div className="card overflow-hidden">
            <div className="grid grid-cols-3 bg-forge-dark px-6 py-3 border-b border-forge-border">
              <span className="text-xs text-forge-muted uppercase tracking-widest">Feature</span>
              <span className="text-xs text-forge-muted uppercase tracking-widest text-center">Free</span>
              <span className="text-xs text-forge-accent uppercase tracking-widest text-center">Pro</span>
            </div>
            {FEATURES.map((f, i) => (
              <div
                key={i}
                className={clsx(
                  'grid grid-cols-3 px-6 py-3 border-b border-forge-border/50',
                  i % 2 === 0 ? 'bg-forge-card/30' : ''
                )}
              >
                <div className="flex items-center gap-2 text-sm text-forge-text">
                  <span className="text-forge-muted">{f.icon}</span>
                  {f.label}
                </div>
                <div className="text-center">
                  {typeof f.free === 'boolean' ? (
                    f.free
                      ? <Check size={16} className="text-forge-green mx-auto" />
                      : <X size={16} className="text-forge-accent/50 mx-auto" />
                  ) : (
                    <span className="text-sm text-forge-muted">{f.free}</span>
                  )}
                </div>
                <div className="text-center">
                  {typeof f.pro === 'boolean' ? (
                    f.pro
                      ? <Check size={16} className="text-forge-green mx-auto" />
                      : <X size={16} className="text-forge-accent/50 mx-auto" />
                  ) : (
                    <span className="text-sm text-forge-text font-semibold">{f.pro}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* License tiers (what buyers get) */}
        <div className="mb-20">
          <div className="text-center mb-8">
            <h2 className="font-display text-2xl text-white tracking-wider">BEAT LICENSE TIERS</h2>
            <p className="text-forge-muted text-sm mt-2">What buyers can choose from when purchasing your beats</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {LICENSE_TIERS.map((tier) => (
              <div key={tier.name} className={`card p-5 border ${tier.border}`}>
                <div className={`font-display text-lg font-bold mb-1 ${tier.color}`}>{tier.name}</div>
                <div className={`text-2xl font-bold text-white mb-4`}>{tier.price}</div>
                <ul className="space-y-1.5 mb-3">
                  {tier.features.map((f) => (
                    <li key={f} className="flex items-start gap-1.5 text-xs text-forge-muted">
                      <Check size={10} className="text-forge-green mt-0.5 flex-shrink-0" />
                      {f}
                    </li>
                  ))}
                </ul>
                {tier.no.length > 0 && (
                  <ul className="space-y-1.5">
                    {tier.no.map((f) => (
                      <li key={f} className="flex items-start gap-1.5 text-xs text-forge-muted/50">
                        <X size={10} className="text-forge-accent/40 mt-0.5 flex-shrink-0" />
                        {f}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
          <p className="text-center text-xs text-forge-muted mt-4">
            Licenses are automatically attached to every beat you publish. You set your own prices.
          </p>
        </div>

        {/* CTA */}
        <div className="text-center">
          <button
            onClick={handleUpgrade}
            disabled={loading || isPro}
            className={clsx(
              'px-12 py-4 text-base font-bold rounded-xl',
              isPro ? 'bg-forge-green/20 text-forge-green border border-forge-green' : 'btn-primary shadow-forge'
            )}
          >
            {isPro ? '✓ You are on Pro' : 'Get Pro — €19.99/month'}
          </button>
          <p className="text-forge-muted text-xs mt-3">No contracts. Cancel anytime from your dashboard.</p>
        </div>

      </div>
    </div>
  )
}
