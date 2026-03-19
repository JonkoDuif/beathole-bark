'use client'
import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { profileApi } from '@/lib/api'
import { Music2, Play, Calendar, Headphones, BarChart2, Loader2, User } from 'lucide-react'
import clsx from 'clsx'

export default function ProfilePage() {
  const { username } = useParams<{ username: string }>()
  const [profile, setProfile] = useState<any>(null)
  const [beats, setBeats] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)
  const [playingId, setPlayingId] = useState<string | null>(null)
  const [audio, setAudio] = useState<HTMLAudioElement | null>(null)

  useEffect(() => {
    profileApi.getPublic(username)
      .then(res => { setProfile(res.data.user); setBeats(res.data.beats) })
      .catch(err => { if (err.response?.status === 404) setNotFound(true) })
      .finally(() => setLoading(false))
  }, [username])

  const handlePlay = (beat: any) => {
    if (!beat.preview_url && !beat.mp3_url && !beat.wav_url) return
    const url = beat.preview_url || beat.mp3_url || beat.wav_url
    if (playingId === beat.id) {
      audio?.pause(); setPlayingId(null); setAudio(null); return
    }
    audio?.pause()
    const a = new Audio(url)
    a.play()
    a.onended = () => { setPlayingId(null); setAudio(null) }
    setPlayingId(beat.id); setAudio(a)
  }

  if (loading) return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-forge-accent" />
    </div>
  )

  if (notFound) return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="flex flex-col items-center justify-center min-h-screen text-center px-4">
        <User size={48} className="text-forge-muted mb-4" />
        <h1 className="font-display text-3xl text-white mb-2">User Not Found</h1>
        <p className="text-forge-muted mb-6">@{username} doesn't exist or has been deactivated.</p>
        <Link href="/marketplace" className="btn-primary px-6 py-2">Browse Marketplace</Link>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-4xl mx-auto px-4 py-24 pt-32">

        {/* Profile header */}
        <div className="card p-8 mb-8 flex flex-col sm:flex-row items-start sm:items-center gap-6">
          <div className="w-20 h-20 rounded-full bg-gradient-forge flex items-center justify-center flex-shrink-0 overflow-hidden border-2 border-forge-border">
            {profile.avatar_url ? (
              <img src={profile.avatar_url} alt="" className="w-full h-full object-cover" />
            ) : (
              <span className="font-display text-3xl text-white">
                {(profile.display_name || profile.username)[0].toUpperCase()}
              </span>
            )}
          </div>
          <div className="flex-1">
            <h1 className="font-display text-3xl text-white mb-1">{profile.display_name || profile.username}</h1>
            <p className="text-forge-muted text-sm mb-3">@{profile.username}</p>
            {profile.bio && <p className="text-forge-text text-sm leading-relaxed mb-4 max-w-xl">{profile.bio}</p>}
            <div className="flex items-center gap-6 text-sm text-forge-muted">
              <div className="flex items-center gap-1.5">
                <Calendar size={13} />
                Joined {new Date(profile.created_at).toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
              </div>
              <div className="flex items-center gap-1.5">
                <Music2 size={13} className="text-forge-accent" />
                <span className="text-forge-text font-semibold">{profile.published_beats}</span> beats
              </div>
              <div className="flex items-center gap-1.5">
                <Headphones size={13} className="text-forge-cyan" />
                <span className="text-forge-text font-semibold">{Number(profile.total_plays).toLocaleString()}</span> plays
              </div>
            </div>
          </div>
        </div>

        {/* Beats grid */}
        <div>
          <h2 className="font-display text-xl text-white mb-4 tracking-wider flex items-center gap-2">
            <BarChart2 size={18} className="text-forge-accent" />
            Published Beats
          </h2>

          {beats.length === 0 ? (
            <div className="card p-12 text-center">
              <Music2 size={32} className="text-forge-muted mx-auto mb-3" />
              <p className="text-forge-muted">No published beats yet.</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {beats.map(beat => (
                <div key={beat.id} className="card p-4 flex items-center gap-4 hover:border-forge-border/80 transition-colors group">
                  {/* Cover */}
                  <div className="w-14 h-14 rounded-lg bg-forge-dark flex-shrink-0 overflow-hidden border border-forge-border relative">
                    {beat.cover_art_url ? (
                      <img src={beat.cover_art_url} alt="" className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full bg-gradient-forge flex items-center justify-center">
                        <Music2 size={18} className="text-white" />
                      </div>
                    )}
                    <button
                      onClick={() => handlePlay(beat)}
                      className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity"
                    >
                      {playingId === beat.id ? (
                        <div className="w-4 h-4 flex gap-0.5 items-end">
                          {[1,2,3].map(i => (
                            <div key={i} className="w-1 bg-white rounded-full animate-bounce" style={{ height: `${8 + i * 4}px`, animationDelay: `${i * 0.1}s` }} />
                          ))}
                        </div>
                      ) : (
                        <Play size={16} className="text-white fill-white" />
                      )}
                    </button>
                  </div>
                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <Link href={`/beat/${beat.id}`} className="text-sm font-semibold text-forge-text hover:text-white transition-colors truncate block">
                      {beat.title}
                    </Link>
                    <div className="flex items-center gap-2 mt-1">
                      {beat.genre && <span className="text-[10px] text-forge-accent border border-forge-accent/30 px-1.5 py-0.5 rounded">{beat.genre}</span>}
                      {beat.bpm && <span className="text-[10px] text-forge-muted">{beat.bpm} BPM</span>}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-forge-muted flex-shrink-0">
                    <Headphones size={11} />
                    {(beat.play_count || 0).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
