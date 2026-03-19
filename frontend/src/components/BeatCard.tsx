'use client'
import { useState, useRef, useEffect } from 'react'
import Link from 'next/link'
import { Play, Pause, ShoppingCart, Clock, Zap } from 'lucide-react'
import clsx from 'clsx'

interface Beat {
  id: string
  title: string
  genre: string
  mood?: string
  bpm?: number
  preview_url?: string
  waveform_data?: number[]
  creator_username: string
  creator_display_name?: string
  creator_avatar?: string
  min_price?: number
  play_count?: number
  created_at: string
}

interface BeatCardProps {
  beat: Beat
  onBuy?: (beat: Beat) => void
}

export default function BeatCard({ beat, onBuy }: BeatCardProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [mounted, setMounted] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)

  useEffect(() => {
    setMounted(true)
  }, [])

  const togglePlay = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!audioRef.current || !beat.preview_url) return

    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  const defaultWaveform = Array.from({ length: 40 }, (_, i) => 0.2 + (Math.sin(i * 0.5) + 1) * 0.3)
  const parseJson = (v: any) => { try { return typeof v === 'string' ? JSON.parse(v) : v } catch { return null } }
  const waveform = parseJson(beat.waveform_data) || (mounted ? Array.from({ length: 40 }, () => Math.random()) : defaultWaveform)
  const genreColors: Record<string, string> = {
    trap: 'badge-accent',
    hiphop: 'badge-purple',
    rnb: 'badge-gold',
    drill: 'badge-accent',
    afrobeats: 'badge-cyan',
    pop: 'badge-cyan',
    default: 'badge-purple',
  }

  const badgeClass = genreColors[beat.genre?.toLowerCase()] || genreColors.default

  return (
    <Link href={`/beat/${beat.id}`}>
      <div className="card-hover group relative overflow-hidden">
        {/* Top gradient bar */}
        <div className="h-1 bg-gradient-forge" />

        <div className="p-5">
          {/* Waveform area */}
          <div className="relative h-16 mb-4 flex items-end gap-0.5 overflow-hidden">
            {waveform.slice(0, 50).map((val, i) => (
              <div
                key={i}
                className={clsx(
                  'flex-1 rounded-sm transition-all duration-300',
                  isPlaying ? 'bg-forge-accent' : 'bg-forge-border group-hover:bg-forge-accent/50'
                )}
                style={{ height: `${Math.max(15, val * 100)}%` }}
              />
            ))}
            {/* Play overlay */}
            <button
              onClick={togglePlay}
              className="absolute inset-0 flex items-center justify-center"
            >
              <div className={clsx(
                'w-12 h-12 rounded-full flex items-center justify-center transition-all duration-200',
                isPlaying
                  ? 'bg-forge-accent shadow-forge scale-100'
                  : 'bg-forge-dark/80 border border-forge-border group-hover:border-forge-accent/50 group-hover:scale-110 opacity-0 group-hover:opacity-100'
              )}>
                {isPlaying
                  ? <Pause size={18} className="text-white" />
                  : <Play size={18} className="text-white ml-0.5" />
                }
              </div>
            </button>

            {/* Playing indicator */}
            {isPlaying && (
              <div className="absolute top-0 right-0 flex items-center gap-1 bg-forge-accent/20 rounded-full px-2 py-0.5">
                {[...Array(5)].map((_, i) => (
                  <div key={i} className="waveform-bar w-0.5 h-3" style={{ animationDelay: `${i * 0.1}s` }} />
                ))}
              </div>
            )}
          </div>

          {/* Beat info */}
          <div className="mb-3">
            <h3 className="font-semibold text-forge-text group-hover:text-white transition-colors truncate text-sm">
              {beat.title}
            </h3>
            <p className="text-forge-muted text-xs mt-0.5">
              by <span className="text-forge-text/70">@{beat.creator_username}</span>
            </p>
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-1.5 mb-4">
            <span className={badgeClass}>{beat.genre}</span>
            {beat.mood && <span className="badge bg-forge-dark text-forge-muted border border-forge-border">{beat.mood}</span>}
            {beat.bpm && (
              <span className="badge bg-forge-dark text-forge-muted border border-forge-border">
                <Zap size={10} className="text-forge-gold" />
                {beat.bpm} BPM
              </span>
            )}
          </div>

          {/* Price + buy */}
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs text-forge-muted">From</span>
              <span className="text-lg font-bold text-forge-accent ml-1">
                €{beat.min_price ? (beat.min_price / 100).toFixed(0) : '5'}
              </span>
            </div>
            <button
              onClick={(e) => {
                e.preventDefault()
                onBuy?.(beat)
              }}
              className="flex items-center gap-1.5 bg-forge-accent/10 hover:bg-forge-accent text-forge-accent hover:text-white border border-forge-accent/30 hover:border-forge-accent px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200"
            >
              <ShoppingCart size={14} />
              License
            </button>
          </div>
        </div>

        {beat.preview_url && (
          <audio
            ref={audioRef}
            src={beat.preview_url}
            onEnded={() => setIsPlaying(false)}
            preload="none"
          />
        )}
      </div>
    </Link>
  )
}
