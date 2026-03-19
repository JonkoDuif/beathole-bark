'use client'
import Link from 'next/link'
import { useEffect, useState } from 'react'
import { Zap, Music2, DollarSign, Shield, ArrowRight, Play, TrendingUp, Users } from 'lucide-react'
import Navbar from '@/components/Navbar'
import { beatsApi } from '@/lib/api'
import BeatCard from '@/components/BeatCard'
import axios from 'axios'

const GENRES = ['Trap', 'Drill', 'Hip-Hop', 'R&B', 'Afrobeats', 'Pop', 'Lo-Fi', 'Dancehall']

const HOW_IT_WORKS = [
  {
    step: '01',
    title: 'Describe Your Beat',
    desc: 'Pick your genre, mood, BPM and style. Our AI understands your vision.',
    color: 'text-forge-accent',
  },
  {
    step: '02',
    title: 'AI Generates',
    desc: 'Advanced music AI creates a unique instrumental beat in seconds.',
    color: 'text-forge-gold',
  },
  {
    step: '03',
    title: 'Publish & Sell',
    desc: 'Your beat goes live on the marketplace with 4 license tiers ready to sell.',
    color: 'text-forge-cyan',
  },
  {
    step: '04',
    title: 'Earn 80%',
    desc: 'Keep 80% of every sale. Withdraw to your bank whenever you want.',
    color: 'text-forge-green',
  },
]

interface PlatformStats {
  beats_generated: number
  total_plays: number
  active_producers: number
  licenses_sold: number
  creators_paid_cents: number
}

function formatNumber(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M+`
  if (n >= 1000) return `${Math.floor(n / 1000)}K+`
  return String(n)
}

function formatEuros(cents: number): string {
  const euros = cents / 100
  if (euros >= 1000000) return `€${(euros / 1000000).toFixed(1)}M+`
  if (euros >= 1000) return `€${Math.floor(euros / 1000)}K+`
  return `€${euros.toFixed(0)}`
}

export default function HomePage() {
  const [beats, setBeats] = useState<any[]>([])
  const [activeGenre, setActiveGenre] = useState(0)
  const [stats, setStats] = useState<PlatformStats | null>(null)
  // floatWave is set client-side only to avoid SSR/hydration mismatch from Math.random()
  const [floatWave, setFloatWave] = useState<number[]>([])

  useEffect(() => {
    beatsApi.list({ limit: 6, sort: 'popular' }).then(res => setBeats(res.data.beats || []))

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'
    axios.get(`${apiUrl}/api/stats`).then(res => setStats(res.data)).catch(() => {})

    // Generate waveform heights client-side only (Math.random() causes SSR hydration mismatch)
    setFloatWave(Array.from({ length: 60 }, (_, i) =>
      Math.abs(Math.sin(i * 0.3) * 0.6 + (Math.random() - 0.5) * 0.4)
    ))
  }, [])

  useEffect(() => {
    const interval = setInterval(() => setActiveGenre(g => (g + 1) % GENRES.length), 1800)
    return () => clearInterval(interval)
  }, [])

  const STATS = stats ? [
    { label: 'Beats Generated', value: formatNumber(stats.beats_generated), icon: <Music2 size={20} className="text-forge-accent" /> },
    { label: 'Paid to Creators', value: formatEuros(stats.creators_paid_cents), icon: <DollarSign size={20} className="text-forge-gold" /> },
    { label: 'Active Producers', value: formatNumber(stats.active_producers), icon: <Users size={20} className="text-forge-cyan" /> },
    { label: 'Licenses Sold', value: formatNumber(stats.licenses_sold), icon: <TrendingUp size={20} className="text-forge-green" /> },
  ] : [
    { label: 'Beats Generated', value: '...', icon: <Music2 size={20} className="text-forge-accent" /> },
    { label: 'Paid to Creators', value: '...', icon: <DollarSign size={20} className="text-forge-gold" /> },
    { label: 'Active Producers', value: '...', icon: <Users size={20} className="text-forge-cyan" /> },
    { label: 'Licenses Sold', value: '...', icon: <TrendingUp size={20} className="text-forge-green" /> },
  ]

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-screen flex items-center pt-16 overflow-hidden">
        <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-50" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[600px] bg-forge-accent/5 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 right-1/4 w-[400px] h-[400px] bg-forge-purple/5 rounded-full blur-[100px]" />

        <div className="relative w-full max-w-[1600px] px-6 sm:px-10 lg:px-20 py-20">
          <div className="max-w-4xl">
            <div className="inline-flex items-center gap-2 bg-forge-card border border-forge-border rounded-full px-4 py-2 mb-8 animate-fade-up">
              <span className="glow-dot" />
              <span className="text-sm text-forge-muted">AI-Powered Beat Generation</span>
              <span className="badge-accent text-[10px]">BETA</span>
            </div>

            <h1 className="font-display text-6xl sm:text-7xl lg:text-9xl leading-none mb-6 animate-fade-up-delay-1">
              <span className="text-white">FORGE</span>
              <br />
              <span className="text-white">YOUR</span>
              {' '}
              <span className="gradient-text">NEXT</span>
              <br />
              <div className="overflow-hidden h-[1.1em]">
                <span
                  key={activeGenre}
                  className="block text-forge-accent"
                  style={{ animation: 'fadeUp 0.4s ease-out forwards' }}
                >
                  {GENRES[activeGenre].toUpperCase()}
                </span>
              </div>
              <span className="text-white">HIT.</span>
            </h1>

            <p className="text-xl text-forge-muted max-w-2xl mb-10 animate-fade-up-delay-2">
              Generate professional instrumental beats with AI. Publish instantly. Earn 80% on every license.
              No studio needed.
            </p>

            <div className="flex flex-wrap gap-4 animate-fade-up-delay-3">
              <Link href="/generate" className="btn-primary flex items-center gap-2 text-lg px-8 py-4">
                <Zap size={20} />
                Generate a Beat
              </Link>
              <Link href="/marketplace" className="btn-secondary flex items-center gap-2 text-lg px-8 py-4">
                <Play size={20} />
                Browse Marketplace
              </Link>
            </div>
          </div>
        </div>

        {/* Floating visual */}
        <div className="absolute right-[12%] top-1/2 -translate-y-1/2 w-[480px] hidden xl:block">
          <div className="relative">
            <div className="bg-forge-card border border-forge-border rounded-2xl p-6 shadow-card animate-float">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-gradient-forge flex items-center justify-center">
                  <Music2 size={18} className="text-white" />
                </div>
                <div>
                  <div className="font-semibold text-sm">Dark Trap Wave</div>
                  <div className="text-forge-muted text-xs">@producer_x • 140 BPM</div>
                </div>
                <div className="ml-auto">
                  <span className="badge-accent">LIVE</span>
                </div>
              </div>
              <div className="flex items-end gap-0.5 h-16 mb-4">
                {(floatWave.length > 0
                  ? floatWave
                  : Array.from({ length: 60 }, (_, i) => Math.abs(Math.sin(i * 0.3) * 0.6))
                ).map((h, i) => (
                  <div key={i} className="flex-1 bg-forge-accent rounded-sm opacity-80"
                    style={{ height: `${Math.max(10, h * 100)}%` }} />
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'MP3 Lease', price: '€5' },
                  { label: 'WAV Lease', price: '€20' },
                  { label: 'Premium', price: '€50' },
                  { label: 'Exclusive', price: '€200' },
                ].map(l => (
                  <div key={l.label} className="bg-forge-dark rounded-lg p-2.5 flex justify-between items-center">
                    <span className="text-xs text-forge-muted">{l.label}</span>
                    <span className="text-xs font-mono text-white">{l.price}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats — ECHTE DATA */}
      <section className="border-y border-forge-border bg-forge-dark/50 py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {STATS.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="flex justify-center mb-2">{stat.icon}</div>
                <div className="font-display text-4xl text-forge-text mb-1">{stat.value}</div>
                <div className="text-forge-muted text-sm">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="section-title mb-4">HOW IT WORKS</h2>
            <p className="text-forge-muted text-lg max-w-xl mx-auto">
              From idea to income in four steps
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {HOW_IT_WORKS.map((step) => (
              <div key={step.step} className="card p-6 relative overflow-hidden group hover:border-forge-accent/30 transition-colors">
                <div className="font-display text-6xl text-forge-border group-hover:text-forge-accent/20 transition-colors absolute top-4 right-4">
                  {step.step}
                </div>
                <div className={`font-display text-lg ${step.color} mb-3`}>{step.step}</div>
                <h3 className="font-semibold text-forge-text mb-2">{step.title}</h3>
                <p className="text-forge-muted text-sm leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Featured Beats */}
      {beats.length > 0 && (
        <section className="py-24 bg-forge-dark/30">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-end justify-between mb-12">
              <div>
                <h2 className="section-title">TRENDING BEATS</h2>
                <p className="text-forge-muted mt-2">Fresh AI-generated instrumentals</p>
              </div>
              <Link href="/marketplace" className="text-forge-accent flex items-center gap-1 hover:gap-2 transition-all text-sm font-medium">
                View All <ArrowRight size={14} />
              </Link>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {beats.map(beat => (
                <BeatCard key={beat.id} beat={beat} />
              ))}
            </div>
          </div>
        </section>
      )}

      {/* CTA */}
      <section className="py-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-forge-accent/5 to-transparent" />
        <div className="relative max-w-4xl mx-auto px-4 text-center">
          <h2 className="font-display text-6xl sm:text-7xl text-white mb-6">
            START FOR FREE<br />
            <span className="gradient-text">CREATE</span>
          </h2>
          <p className="text-forge-muted text-xl mb-10 max-w-2xl mx-auto">
            Join producers earning passive income with AI-generated beats. No equipment needed.
          </p>
          <Link href="/register" className="btn-primary inline-flex items-center gap-2 text-xl px-10 py-5">
            <Zap size={22} />
            Make Your First Beat Free
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-forge-border py-12 bg-forge-dark/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-2">
              <Music2 size={20} className="text-forge-accent" />
              <span className="font-display text-xl tracking-wider">
                BEAT<span className="text-forge-accent">HOLE</span>
                <span className="ml-2 text-forge-cyan text-sm">AI</span>
              </span>
            </div>
            <div className="flex flex-wrap gap-4 text-sm text-forge-muted justify-center">
              <Link href="/marketplace" className="hover:text-forge-text transition-colors">Marketplace</Link>
              <Link href="/generate" className="hover:text-forge-text transition-colors">Generate</Link>
              <Link href="/pricing" className="hover:text-forge-text transition-colors">Pricing</Link>
              <Link href="/faq" className="hover:text-forge-text transition-colors">FAQ</Link>
              <Link href="/support" className="hover:text-forge-text transition-colors">Support</Link>
              <Link href="/terms" className="hover:text-forge-text transition-colors">Terms & Privacy</Link>
            </div>
            <p className="text-forge-muted text-sm">© 2026 BeatHole AI. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
