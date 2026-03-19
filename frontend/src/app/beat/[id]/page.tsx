'use client'
import { useEffect, useState, useRef } from 'react'
import { useParams } from 'next/navigation'
import Navbar from '@/components/Navbar'
import LicenseModal from '@/components/LicenseModal'
import { beatsApi } from '@/lib/api'
import { Play, Pause, Download, Share2, Zap, Music2, User, Clock, BarChart2, Crown, Star, FileAudio } from 'lucide-react'
import clsx from 'clsx'

const LICENSE_ICONS: Record<string, any> = {
  mp3_lease: { icon: <Music2 size={18} />, color: 'text-forge-muted', bg: 'bg-forge-card', border: 'border-forge-border' },
  wav_lease: { icon: <FileAudio size={18} />, color: 'text-forge-cyan', bg: 'bg-forge-cyan/10', border: 'border-forge-cyan/40' },
  premium_lease: { icon: <Star size={18} />, color: 'text-forge-gold', bg: 'bg-forge-gold/10', border: 'border-forge-gold/40' },
  exclusive: { icon: <Crown size={18} />, color: 'text-forge-accent', bg: 'bg-forge-accent/10', border: 'border-forge-accent/40' },
}

export default function BeatPage() {
  const { id } = useParams()
  const [beat, setBeat] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [showModal, setShowModal] = useState(false)
  const [modalLicense, setModalLicense] = useState<any>(null)
  const audioRef = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    beatsApi.get(id as string)
      .then(res => { setBeat(res.data) })
      .catch(() => { setBeat(null) })
      .finally(() => setLoading(false))
  }, [id])

  const togglePlay = () => {
    if (!audioRef.current) return
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) setCurrentTime(audioRef.current.currentTime)
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) setDuration(audioRef.current.duration)
  }

  const seekTo = (clientX: number, rect: DOMRect) => {
    if (!audioRef.current || !duration) return
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    audioRef.current.currentTime = ratio * duration
    setCurrentTime(ratio * duration)
  }

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    seekTo(e.clientX, e.currentTarget.getBoundingClientRect())
  }

  const handleTouchSeek = (e: React.TouchEvent<HTMLDivElement>) => {
    const touch = e.changedTouches[0]
    seekTo(touch.clientX, e.currentTarget.getBoundingClientRect())
  }

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`

  if (loading) {
    return (
      <div className="min-h-screen bg-forge-black">
        <Navbar />
        <div className="flex items-center justify-center min-h-screen">
          <div className="w-12 h-12 rounded-full border-2 border-forge-accent border-t-transparent animate-spin" />
        </div>
      </div>
    )
  }

  if (!beat) return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="flex flex-col items-center justify-center min-h-screen gap-4">
        <Music2 size={48} className="text-forge-border" />
        <p className="text-forge-muted text-lg">Beat not found or still generating.</p>
        <a href="/dashboard" className="text-forge-accent hover:underline text-sm">Go to Dashboard</a>
      </div>
    </div>
  )

  const parseJson = (v: any) => { try { return typeof v === 'string' ? JSON.parse(v) : v } catch { return null } }
  const waveform: number[] = parseJson(beat.waveform_data) || Array.from({ length: 120 }, () => Math.random())
  const progressRatio = duration ? currentTime / duration : 0
  const isGenerating = beat.status === 'generating'

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20">

        {/* Beat header */}
        <div className="card overflow-visible mb-6">
          {/* Gradient bar */}
          <div className="h-1.5 bg-gradient-forge" />

          <div className="p-8">
            <div className="flex flex-col lg:flex-row gap-8">
              {/* Left: Player */}
              <div className="flex-1">
                <div className="flex items-start gap-4 mb-6">
                  <div className="w-16 h-16 rounded-xl bg-gradient-forge flex items-center justify-center flex-shrink-0 shadow-forge">
                    <Music2 size={28} className="text-white" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <h1 className="font-display text-3xl sm:text-4xl text-white truncate">{beat.title}</h1>
                    <p className="text-forge-muted mt-1">
                      by <span className="text-forge-text hover:text-forge-accent transition-colors cursor-pointer">
                        @{beat.creator_username}
                      </span>
                    </p>
                  </div>
                </div>

                {/* Tags row */}
                <div className="flex flex-wrap gap-2 mb-6">
                  <span className="badge-accent">{beat.genre}</span>
                  {beat.mood && <span className="badge bg-forge-dark border border-forge-border text-forge-muted">{beat.mood}</span>}
                  {beat.bpm && (
                    <span className="badge bg-forge-dark border border-forge-border text-forge-muted">
                      <Zap size={10} className="text-forge-gold" /> {beat.bpm} BPM
                    </span>
                  )}
                  {beat.style && <span className="badge bg-forge-dark border border-forge-border text-forge-muted">{beat.style}</span>}
                </div>

                {/* Waveform player / Generating state */}
                {isGenerating ? (
                  <div className="bg-forge-dark rounded-xl p-6 mb-4 flex flex-col items-center justify-center gap-3 min-h-[120px]">
                    <div className="w-8 h-8 rounded-full border-2 border-forge-accent border-t-transparent animate-spin" />
                    <p className="text-forge-text font-semibold">Still Generating</p>
                    <p className="text-forge-muted text-sm text-center">This beat is being generated. Check back in a moment.</p>
                  </div>
                ) : (
                  <div className="bg-forge-dark rounded-xl p-4 mb-4">
                    {/* Waveform bars */}
                    <div
                      className="relative flex items-end gap-px h-20 mb-3 cursor-pointer"
                      onClick={handleSeek}
                      onTouchEnd={handleTouchSeek}
                    >
                      {waveform.map((val, i) => {
                        const ratio = i / waveform.length
                        const played = ratio <= progressRatio
                        return (
                          <div
                            key={i}
                            className={clsx(
                              'flex-1 rounded-sm transition-colors',
                              played ? 'bg-forge-accent' : 'bg-forge-border hover:bg-forge-accent/30'
                            )}
                            style={{ height: `${Math.max(8, val * 100)}%` }}
                          />
                        )
                      })}
                      {/* Playhead */}
                      <div
                        className="absolute top-0 bottom-0 w-0.5 bg-white/40 pointer-events-none"
                        style={{ left: `${progressRatio * 100}%` }}
                      />
                    </div>

                    {/* Controls */}
                    <div className="flex items-center gap-4">
                      <button
                        onClick={togglePlay}
                        className="w-10 h-10 rounded-full bg-forge-accent flex items-center justify-center hover:bg-forge-accent-dim transition-colors flex-shrink-0"
                      >
                        {isPlaying
                          ? <Pause size={16} className="text-white" />
                          : <Play size={16} className="text-white ml-0.5" />
                        }
                      </button>
                      <span className="text-xs text-forge-muted font-mono">
                        {formatTime(currentTime)} / {formatTime(duration || beat.duration_seconds || 0)}
                      </span>
                      <div className="ml-auto flex items-center gap-2 text-forge-muted text-xs">
                        <BarChart2 size={12} />
                        {(beat.play_count || 0).toLocaleString()} plays
                      </div>
                    </div>
                  </div>
                )}

                {/* Action buttons */}
                <div className="flex gap-3">
                  {!isGenerating && beat.preview_url && (
                    <a
                      href={beat.preview_url}
                      download
                      className="flex items-center gap-2 btn-secondary text-sm py-2 px-4"
                    >
                      <Download size={14} />
                      Preview
                    </a>
                  )}
                  <button
                    onClick={() => navigator.share?.({ title: beat.title, url: window.location.href })}
                    className="flex items-center gap-2 btn-ghost text-sm py-2 px-4"
                  >
                    <Share2 size={14} />
                    Share
                  </button>
                </div>
              </div>

              {/* Creator info */}
              <div className="lg:w-52 flex-shrink-0">
                <div className="card p-4 text-center">
                  <div className="w-16 h-16 rounded-full bg-gradient-forge flex items-center justify-center text-white text-xl font-bold mx-auto mb-3">
                    {beat.creator_display_name?.[0]?.toUpperCase() || beat.creator_username[0].toUpperCase()}
                  </div>
                  <p className="font-semibold text-forge-text">{beat.creator_display_name || beat.creator_username}</p>
                  <p className="text-forge-muted text-sm">@{beat.creator_username}</p>
                  {beat.creator_bio && (
                    <p className="text-forge-muted text-xs mt-2 leading-relaxed">{beat.creator_bio}</p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* License section */}
        <div className="mb-6">
          <h2 className="font-display text-2xl text-white mb-4">LICENSE OPTIONS</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {(beat.licenses || []).map((license: any) => {
              const style = LICENSE_ICONS[license.type] || LICENSE_ICONS.mp3_lease
              return (
                <div key={license.id} className={`card p-4 border-2 ${style.border} ${style.bg} transition-all hover:scale-[1.02]`}>
                  <div className={`flex items-center gap-2 mb-2 ${style.color}`}>
                    {style.icon}
                    <span className="font-semibold text-sm text-forge-text">{license.name}</span>
                  </div>
                  <div className="text-2xl font-bold text-forge-text mb-2">
                    €{(license.price_cents / 100).toFixed(0)}
                  </div>
                  <ul className="space-y-1 mb-4">
                    {(parseJson(license.features) || []).slice(0, 4).map((f: string, i: number) => (
                      <li key={i} className="text-xs text-forge-muted flex items-center gap-1.5">
                        <span className="w-1 h-1 rounded-full bg-forge-green flex-shrink-0" />
                        {f}
                      </li>
                    ))}
                  </ul>
                  <button
                    onClick={() => { setModalLicense(license); setShowModal(true) }}
                    className={clsx(
                      'w-full py-2 rounded-lg text-sm font-medium transition-all',
                      style.color,
                      `border ${style.border} hover:bg-forge-accent hover:text-white hover:border-forge-accent`
                    )}
                  >
                    Buy Now
                  </button>
                </div>
              )
            })}
          </div>
        </div>

        {/* Description */}
        {beat.description && (
          <div className="card p-6">
            <h2 className="font-semibold text-forge-text mb-2">About This Beat</h2>
            <p className="text-forge-muted leading-relaxed">{beat.description}</p>
          </div>
        )}
      </div>

      {/* Audio element */}
      {!isGenerating && beat.preview_url && (
        <audio
          ref={audioRef}
          src={beat.preview_url}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />
      )}

      {/* License modal */}
      {showModal && (
        <LicenseModal
          beat={beat}
          licenses={beat.licenses || []}
          initialLicense={modalLicense}
          onClose={() => { setShowModal(false); setModalLicense(null) }}
        />
      )}
    </div>
  )
}
