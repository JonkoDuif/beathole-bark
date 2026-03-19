'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { downloadsApi } from '@/lib/api'
import toast from 'react-hot-toast'
import {
  Download, FileText, Music2, Radio, Mic2, Tv2,
  Loader2, PlayCircle, Crown, ShieldCheck, Sliders
} from 'lucide-react'
import Link from 'next/link'
import clsx from 'clsx'

interface LibraryItem {
  order_id: string
  purchased_at: string
  amount_cents: number
  beat_id: string
  beat_title: string
  genre: string
  bpm: number | null
  mood: string | null
  mp3_url: string | null
  wav_url: string | null
  preview_url: string | null
  duration_seconds: number | null
  license_id: string
  license_name: string
  license_type: 'basic' | 'standard' | 'premium' | 'exclusive'
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
  creator_username: string
  creator_display_name: string | null
  beat_type?: string
}

const LICENSE_COLORS: Record<string, string> = {
  basic: 'text-forge-muted border-forge-border',
  standard: 'text-forge-cyan border-forge-cyan/40',
  premium: 'text-forge-gold border-forge-gold/40',
  exclusive: 'text-forge-accent border-forge-accent/40',
}

const LICENSE_BG: Record<string, string> = {
  basic: 'bg-forge-muted/10',
  standard: 'bg-forge-cyan/10',
  premium: 'bg-forge-gold/10',
  exclusive: 'bg-forge-accent/10',
}

export default function LibraryPage() {
  const { user, isLoading } = useAuthStore()
  const router = useRouter()
  const [items, setItems] = useState<LibraryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [downloading, setDownloading] = useState<string | null>(null)
  const [playingUrl, setPlayingUrl] = useState<string | null>(null)
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (!isLoading && !user) router.push('/login?redirect=/library')
  }, [user, isLoading])

  useEffect(() => {
    if (user) fetchLibrary()
  }, [user])

  useEffect(() => {
    return () => { audio?.pause() }
  }, [audio])

  const fetchLibrary = async () => {
    setLoading(true)
    try {
      const res = await downloadsApi.myLibrary()
      setItems(res.data)
    } catch {
      toast.error('Failed to load library')
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async (orderId: string, format: 'mp3' | 'wav', beatTitle: string, licenseType: string) => {
    const key = `${orderId}-${format}`
    setDownloading(key)
    try {
      const url = downloadsApi.beatDownloadUrl(orderId, format)
      const filename = `${beatTitle.replace(/[^a-zA-Z0-9]/g, '_')}_${licenseType}.${format}`
      await downloadsApi.downloadFile(url, filename)
      toast.success(`${format.toUpperCase()} downloaded!`)
    } catch (err: any) {
      toast.error(err.message || 'Download failed')
    } finally {
      setDownloading(null)
    }
  }

  const handleLicensePdf = async (orderId: string, beatTitle: string) => {
    const key = `${orderId}-pdf`
    setDownloading(key)
    try {
      const url = downloadsApi.licensePdfUrl(orderId)
      const filename = `License_${beatTitle.replace(/[^a-zA-Z0-9]/g, '_')}.pdf`
      await downloadsApi.downloadFile(url, filename)
      toast.success('License PDF downloaded!')
    } catch {
      toast.error('Failed to download license')
    } finally {
      setDownloading(null)
    }
  }

  const handlePlay = (url: string) => {
    if (playingUrl === url) {
      audio?.pause()
      setPlayingUrl(null)
      setAudio(null)
      return
    }
    audio?.pause()
    const a = new Audio(url)
    a.play()
    a.onended = () => { setPlayingUrl(null); setAudio(null) }
    setAudio(a)
    setPlayingUrl(url)
  }

  const formatStreams = (n: number | null) => n === null ? 'Unlimited' : n.toLocaleString()
  const formatDate = (s: string) => new Date(s).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-5xl mx-auto px-4 py-24 pt-32">

        {/* Header */}
        <div className="flex items-center justify-between mb-10">
          <div>
            <h1 className="font-display text-4xl text-white tracking-wider">MY LIBRARY</h1>
            <p className="text-forge-muted mt-1">Your purchased beats and license documents</p>
          </div>
          <div className="hidden sm:flex items-center gap-2 text-sm text-forge-muted bg-forge-card border border-forge-border rounded-lg px-4 py-2">
            <ShieldCheck size={14} className="text-forge-green" />
            {items.length} {items.length === 1 ? 'purchase' : 'purchases'}
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="text-center py-20">
            <Loader2 size={32} className="text-forge-accent animate-spin mx-auto mb-4" />
            <p className="text-forge-muted">Loading your library...</p>
          </div>
        )}

        {/* Empty */}
        {!loading && items.length === 0 && (
          <div className="card p-16 text-center">
            <Music2 size={48} className="text-forge-muted mx-auto mb-4" />
            <h2 className="font-display text-2xl text-white mb-2">NO PURCHASES YET</h2>
            <p className="text-forge-muted mb-6">Head to the marketplace to find beats you love.</p>
            <button onClick={() => router.push('/marketplace')} className="btn-primary px-8 py-3">
              Browse Marketplace
            </button>
          </div>
        )}

        {/* Library items */}
        {!loading && items.map((item) => (
          <div key={item.order_id} className={clsx(
            'card mb-4 overflow-hidden border',
            LICENSE_COLORS[item.license_type]
          )}>
            <div className="p-6">
              <div className="flex flex-col sm:flex-row sm:items-start gap-4">

                {/* Play button + beat info */}
                <div className="flex items-start gap-4 flex-1 min-w-0">
                  {item.preview_url ? (
                    <button
                      onClick={() => handlePlay(item.preview_url!)}
                      className="flex-shrink-0 w-12 h-12 rounded-xl bg-forge-dark border border-forge-border flex items-center justify-center hover:border-forge-accent transition-colors"
                    >
                      <PlayCircle size={22} className={playingUrl === item.preview_url ? 'text-forge-accent' : 'text-forge-muted'} />
                    </button>
                  ) : (
                    <div className="flex-shrink-0 w-12 h-12 rounded-xl bg-forge-dark border border-forge-border flex items-center justify-center">
                      <Music2 size={20} className="text-forge-muted" />
                    </div>
                  )}

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-bold text-white text-lg leading-tight truncate">{item.beat_title}</h3>
                      {item.beat_type === 'midi' && (
                        <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded bg-forge-gold/15 text-forge-gold border border-forge-gold/30 flex-shrink-0">
                          MIDI
                        </span>
                      )}
                    </div>
                    <p className="text-forge-muted text-sm">
                      by {item.creator_display_name || item.creator_username}
                      {item.genre && <> · {item.genre}</>}
                      {item.bpm && <> · {item.bpm} BPM</>}
                    </p>
                    <p className="text-forge-muted text-xs mt-1">Purchased {formatDate(item.purchased_at)} · €{(item.amount_cents / 100).toFixed(2)}</p>
                  </div>
                </div>

                {/* License badge */}
                <div className={clsx(
                  'flex-shrink-0 flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-bold uppercase tracking-widest',
                  LICENSE_BG[item.license_type],
                  LICENSE_COLORS[item.license_type]
                )}>
                  {item.is_exclusive && <Crown size={12} />}
                  {item.license_name}
                </div>
              </div>

              {/* License rights grid */}
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 mt-5 pt-5 border-t border-forge-border/50">
                <RightBadge
                  icon={<Music2 size={12} />}
                  label="Streams"
                  value={formatStreams(item.max_streams)}
                  positive
                />
                <RightBadge
                  icon={<Tv2 size={12} />}
                  label="Music Videos"
                  value={item.max_music_videos === null ? 'Unlimited' : String(item.max_music_videos)}
                  positive
                />
                <RightBadge
                  icon={<Radio size={12} />}
                  label="Radio"
                  value={item.radio_allowed ? 'Allowed' : 'Not allowed'}
                  positive={item.radio_allowed}
                />
                <RightBadge
                  icon={<Mic2 size={12} />}
                  label="Stage"
                  value={
                    item.stage_performance_allowed
                      ? item.max_stage_shows ? `${item.max_stage_shows} shows/yr` : 'Unlimited'
                      : 'Not allowed'
                  }
                  positive={item.stage_performance_allowed}
                />
                <RightBadge
                  icon={<ShieldCheck size={12} />}
                  label="Commercial"
                  value={item.commercial_use ? 'Allowed' : 'Non-commercial'}
                  positive={item.commercial_use}
                />
                <RightBadge
                  icon={<Crown size={12} />}
                  label="Exclusive"
                  value={item.is_exclusive ? 'Yes' : 'Non-exclusive'}
                  positive={item.is_exclusive}
                />
              </div>

              {/* Download buttons + Studio */}
              <div className="flex flex-wrap gap-2 mt-5 pt-5 border-t border-forge-border/50">
                <Link
                  href={`/studio/${item.beat_id}`}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg border text-xs font-semibold transition-all text-forge-accent border-forge-accent/40 hover:bg-forge-accent/10"
                >
                  <Sliders size={13} /> Open Studio
                </Link>
                {item.mp3_url && (
                  <DownloadButton
                    label="MP3"
                    loading={downloading === `${item.order_id}-mp3`}
                    onClick={() => handleDownload(item.order_id, 'mp3', item.beat_title, item.license_type)}
                    color="text-forge-green border-forge-green/40 hover:bg-forge-green/10"
                  />
                )}
                {item.wav_url && ['standard', 'premium', 'exclusive'].includes(item.license_type) && (
                  <DownloadButton
                    label="WAV"
                    loading={downloading === `${item.order_id}-wav`}
                    onClick={() => handleDownload(item.order_id, 'wav', item.beat_title, item.license_type)}
                    color="text-forge-cyan border-forge-cyan/40 hover:bg-forge-cyan/10"
                  />
                )}
                <DownloadButton
                  label="License PDF"
                  icon={<FileText size={13} />}
                  loading={downloading === `${item.order_id}-pdf`}
                  onClick={() => handleLicensePdf(item.order_id, item.beat_title)}
                  color="text-forge-gold border-forge-gold/40 hover:bg-forge-gold/10"
                />
              </div>

              {/* Credit notice */}
              {item.credit_required && (
                <p className="text-xs text-forge-muted mt-3">
                  Credit required: <span className="text-forge-text font-mono">"Prod. by {item.creator_display_name || item.creator_username}"</span>
                </p>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function RightBadge({ icon, label, value, positive }: {
  icon: React.ReactNode
  label: string
  value: string
  positive?: boolean
}) {
  return (
    <div className="bg-forge-dark rounded-lg p-2.5 text-center">
      <div className="flex items-center justify-center gap-1 text-forge-muted mb-1">
        {icon}
        <span className="text-[10px] uppercase tracking-widest">{label}</span>
      </div>
      <div className={clsx(
        'text-xs font-semibold',
        positive === false ? 'text-forge-accent/70' : 'text-forge-text'
      )}>
        {value}
      </div>
    </div>
  )
}

function DownloadButton({ label, icon, loading, onClick, color }: {
  label: string
  icon?: React.ReactNode
  loading: boolean
  onClick: () => void
  color: string
}) {
  return (
    <button
      onClick={onClick}
      disabled={loading}
      className={clsx(
        'flex items-center gap-2 px-4 py-2 rounded-lg border text-xs font-semibold transition-all disabled:opacity-50',
        color
      )}
    >
      {loading
        ? <Loader2 size={13} className="animate-spin" />
        : icon || <Download size={13} />
      }
      {label}
    </button>
  )
}
