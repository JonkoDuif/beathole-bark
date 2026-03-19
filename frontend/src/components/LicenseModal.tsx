'use client'
import { useState } from 'react'
import {
  X, Check, Loader2, Music, FileAudio, Star, Crown,
  Radio, Mic2, Tv2, ShieldCheck, Info
} from 'lucide-react'
import { ordersApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { loadStripe } from '@stripe/stripe-js'
import clsx from 'clsx'

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!)

export interface License {
  id: string
  name: string
  type: 'basic' | 'standard' | 'premium' | 'exclusive'
  price_cents: number
  description: string
  features: string[]
  max_streams: number | null
  max_music_videos: number | null
  radio_allowed: boolean
  stage_performance_allowed: boolean
  max_stage_shows: number | null
  commercial_use: boolean
  sync_licensing: boolean
  distribution_copies: number | null
  credit_required: boolean
  is_exclusive: boolean
}

interface LicenseModalProps {
  beat: { id: string; title: string; creator_username: string; creator_display_name?: string }
  licenses: License[]
  initialLicense?: License | null
  onClose: () => void
}

const CONFIG: Record<string, {
  icon: React.ReactNode
  color: string
  selectedBg: string
  border: string
  selectedBorder: string
  badge: string
}> = {
  basic: {
    icon: <Music size={18} className="text-forge-muted" />,
    color: 'text-forge-muted',
    selectedBg: 'bg-forge-muted/10',
    border: 'border-forge-border',
    selectedBorder: 'border-forge-muted',
    badge: 'bg-forge-muted/20 text-forge-muted',
  },
  standard: {
    icon: <FileAudio size={18} className="text-forge-cyan" />,
    color: 'text-forge-cyan',
    selectedBg: 'bg-forge-cyan/10',
    border: 'border-forge-border',
    selectedBorder: 'border-forge-cyan',
    badge: 'bg-forge-cyan/20 text-forge-cyan',
  },
  premium: {
    icon: <Star size={18} className="text-forge-gold" />,
    color: 'text-forge-gold',
    selectedBg: 'bg-forge-gold/10',
    border: 'border-forge-border',
    selectedBorder: 'border-forge-gold',
    badge: 'bg-forge-gold/20 text-forge-gold',
  },
  exclusive: {
    icon: <Crown size={18} className="text-forge-accent" />,
    color: 'text-forge-accent',
    selectedBg: 'bg-forge-accent/10',
    border: 'border-forge-border',
    selectedBorder: 'border-forge-accent',
    badge: 'bg-forge-accent/20 text-forge-accent',
  },
}

const formatStreams = (n: number | null) => (n === null ? 'Unlimited' : n >= 1_000_000 ? `${n / 1_000_000}M` : n >= 1000 ? `${n / 1000}K` : String(n))

export default function LicenseModal({ beat, licenses, initialLicense, onClose }: LicenseModalProps) {
  const [selected, setSelected] = useState<License | null>(initialLicense || licenses[0] || null)
  const [loading, setLoading] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  const cfg = selected ? CONFIG[selected.type] : null

  const handlePurchase = async () => {
    if (!selected) return
    setLoading(true)
    try {
      const res = await ordersApi.checkout({ beatId: beat.id, licenseId: selected.id })
      const stripe = await stripePromise
      if (!stripe) throw new Error('Stripe not loaded')
      if (res.data.url) {
        window.location.href = res.data.url
      } else {
        await stripe.redirectToCheckout({ sessionId: res.data.sessionId })
      }
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Purchase failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
      <div
        className="relative z-10 bg-forge-dark border border-forge-border rounded-2xl w-full max-w-2xl max-h-[92vh] overflow-y-auto shadow-card animate-fade-up"
        onClick={e => e.stopPropagation()}
      >

        {/* Header */}
        <div className="p-6 border-b border-forge-border flex items-start justify-between">
          <div>
            <h2 className="font-display text-2xl text-forge-text">Choose License</h2>
            <p className="text-forge-muted text-sm mt-1">
              <span className="text-forge-text font-semibold">{beat.title}</span>
              {' '}· by @{beat.creator_username}
            </p>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-forge-card rounded-lg text-forge-muted hover:text-forge-text transition-colors">
            <X size={20} />
          </button>
        </div>

        {/* License tier selector */}
        <div className="p-6 grid grid-cols-2 sm:grid-cols-4 gap-2">
          {licenses.map((license) => {
            const c = CONFIG[license.type]
            const isSelected = selected?.id === license.id
            return (
              <button
                key={license.id}
                onClick={() => { setSelected(license); setShowDetails(false) }}
                className={clsx(
                  'text-left p-3 rounded-xl border-2 transition-all duration-200',
                  isSelected ? `${c.selectedBorder} ${c.selectedBg}` : `${c.border} bg-forge-card hover:${c.selectedBorder}`
                )}
              >
                <div className="flex items-center justify-between mb-2">
                  {c.icon}
                  {isSelected && (
                    <div className="w-4 h-4 rounded-full bg-forge-accent flex items-center justify-center">
                      <Check size={10} className="text-white" />
                    </div>
                  )}
                </div>
                <div className={clsx('text-xs font-bold uppercase tracking-widest mb-1', c.color)}>
                  {license.name}
                </div>
                <div className="text-lg font-bold text-white">
                  €{(license.price_cents / 100).toFixed(0)}
                </div>
                {license.is_exclusive && (
                  <div className="text-[10px] text-forge-accent mt-1 font-mono">EXCL.</div>
                )}
              </button>
            )
          })}
        </div>

        {/* Selected license rights */}
        {selected && (
          <div className="px-6 pb-2">
            <div className={clsx('rounded-xl border p-4', cfg?.selectedBorder, cfg?.selectedBg)}>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  {cfg?.icon}
                  <span className={clsx('font-bold text-sm', cfg?.color)}>{selected.name}</span>
                </div>
                <button
                  onClick={() => setShowDetails(!showDetails)}
                  className="text-xs text-forge-muted flex items-center gap-1 hover:text-forge-text transition-colors"
                >
                  <Info size={12} />
                  {showDetails ? 'Hide details' : 'Full terms'}
                </button>
              </div>

              {/* Quick rights grid */}
              <div className="grid grid-cols-3 gap-2 mb-4">
                <QuickRight
                  icon={<Music size={12} />}
                  label="Streams"
                  value={formatStreams(selected.max_streams)}
                  ok
                />
                <QuickRight
                  icon={<Tv2 size={12} />}
                  label="Music Videos"
                  value={selected.max_music_videos === null ? 'Unlimited' : String(selected.max_music_videos)}
                  ok
                />
                <QuickRight
                  icon={<ShieldCheck size={12} />}
                  label="Commercial"
                  value={selected.commercial_use ? 'Yes' : 'No'}
                  ok={selected.commercial_use}
                />
                <QuickRight
                  icon={<Radio size={12} />}
                  label="Radio"
                  value={selected.radio_allowed ? 'Allowed' : 'Not allowed'}
                  ok={selected.radio_allowed}
                />
                <QuickRight
                  icon={<Mic2 size={12} />}
                  label="Stage"
                  value={selected.stage_performance_allowed
                    ? selected.max_stage_shows ? `${selected.max_stage_shows}/yr` : 'Unlimited'
                    : 'Not allowed'}
                  ok={selected.stage_performance_allowed}
                />
                <QuickRight
                  icon={<Crown size={12} />}
                  label="Exclusive"
                  value={selected.is_exclusive ? 'Yes' : 'Non-excl.'}
                  ok={selected.is_exclusive}
                />
              </div>

              {/* Full terms (toggle) */}
              {showDetails && (
                <div className="border-t border-forge-border/50 pt-3 mt-2 space-y-2">
                  <h4 className="text-xs font-bold text-forge-muted uppercase tracking-widest mb-2">Full License Terms</h4>
                  {TERM_ROWS(selected).map((row) => (
                    <div key={row.label} className="flex items-start gap-2 text-xs">
                      <span className={clsx(
                        'flex-shrink-0 mt-0.5',
                        row.ok === true ? 'text-forge-green' : row.ok === false ? 'text-forge-accent/60' : 'text-forge-muted'
                      )}>
                        {row.ok === true ? '✓' : row.ok === false ? '✗' : '–'}
                      </span>
                      <span className="text-forge-muted w-32 flex-shrink-0">{row.label}</span>
                      <span className="text-forge-text">{row.value}</span>
                    </div>
                  ))}
                  <p className="text-[10px] text-forge-muted mt-3 border-t border-forge-border/40 pt-2">
                    This is a {selected.is_exclusive ? 'exclusive' : 'non-exclusive'} license.
                    {selected.credit_required && ' Producer credit required.'}
                    {' '}A full license PDF is provided after purchase.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Features list from DB */}
        {selected && (() => { try { return typeof selected.features === 'string' ? JSON.parse(selected.features) : selected.features } catch { return [] } })()?.length > 0 && (
          <div className="px-6 py-3">
            <ul className="grid grid-cols-1 sm:grid-cols-2 gap-1">
              {(typeof selected.features === 'string' ? JSON.parse(selected.features) : selected.features ?? []).slice(0, 8).map((f: string, i: number) => (
                <li key={i} className="flex items-center gap-1.5 text-xs text-forge-muted">
                  <Check size={10} className="text-forge-green flex-shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Buy section */}
        <div className="p-6 border-t border-forge-border">
          {selected && (
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm text-forge-muted">
                {selected.is_exclusive && (
                  <p className="text-forge-accent text-xs mb-1 font-semibold">
                    ⚠ Beat will be removed from marketplace after purchase
                  </p>
                )}
                <span>Selected: </span>
                <span className={clsx('font-medium', cfg?.color)}>{selected.name}</span>
              </div>
              <span className={clsx('text-3xl font-bold', cfg?.color)}>
                €{(selected.price_cents / 100).toFixed(2)}
              </span>
            </div>
          )}
          <button
            onClick={handlePurchase}
            disabled={!selected || loading}
            className="w-full btn-primary py-4 text-base flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <><Loader2 size={18} className="animate-spin" /> Processing...</>
            ) : (
              <>Purchase License — €{selected ? (selected.price_cents / 100).toFixed(2) : '0'}</>
            )}
          </button>
          <p className="text-xs text-forge-muted text-center mt-3">
            Secured by Stripe · Instant download after purchase · License PDF included
          </p>
        </div>
      </div>
    </div>
  )
}

function QuickRight({ icon, label, value, ok }: { icon: React.ReactNode; label: string; value: string; ok?: boolean }) {
  return (
    <div className="bg-forge-dark/60 rounded-lg p-2 text-center">
      <div className="flex items-center justify-center gap-1 text-forge-muted mb-1 text-[10px]">
        {icon} {label}
      </div>
      <span className={clsx(
        'text-xs font-semibold',
        ok === true ? 'text-forge-green' : ok === false ? 'text-forge-accent/60' : 'text-forge-text'
      )}>
        {value}
      </span>
    </div>
  )
}

function TERM_ROWS(l: License) {
  return [
    { label: 'Audio Format', value: l.is_exclusive ? 'WAV+MP3+Stems+Project' : l.type === 'premium' ? 'WAV+MP3+Stems' : l.type === 'standard' ? 'WAV+MP3' : 'MP3 320kbps', ok: null },
    { label: 'Max Streams', value: l.max_streams ? l.max_streams.toLocaleString() : 'Unlimited', ok: true },
    { label: 'Music Videos', value: l.max_music_videos === null ? 'Unlimited' : String(l.max_music_videos), ok: l.max_music_videos !== 0 },
    { label: 'Commercial Use', value: l.commercial_use ? 'Allowed' : 'Non-commercial only', ok: l.commercial_use },
    { label: 'Radio / TV', value: l.radio_allowed ? 'FM/AM/Internet radio allowed' : 'Not allowed', ok: l.radio_allowed },
    { label: 'Live Performances', value: l.stage_performance_allowed ? (l.max_stage_shows ? `Up to ${l.max_stage_shows} shows/year` : 'Unlimited') : 'Not allowed', ok: l.stage_performance_allowed },
    { label: 'Sync (Film/TV/Ads)', value: l.sync_licensing ? 'Allowed' : 'Not allowed', ok: l.sync_licensing },
    { label: 'Distribution', value: l.distribution_copies ? `${l.distribution_copies.toLocaleString()} copies` : 'Unlimited', ok: true },
    { label: 'Credit Required', value: l.credit_required ? 'Yes — "Prod. by [Producer]"' : 'Not required', ok: null },
    { label: 'Exclusive Rights', value: l.is_exclusive ? 'Full exclusive — beat removed' : 'Non-exclusive', ok: l.is_exclusive },
  ]
}
