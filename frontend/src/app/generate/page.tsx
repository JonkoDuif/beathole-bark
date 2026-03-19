'use client'
import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { beatsApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Zap, Music2, CheckCircle, ChevronDown, Crown, Lock, FileAudio, FileCode2, Timer, Play, Square, Clock, Upload, X } from 'lucide-react'
import clsx from 'clsx'

const GENRES = ['Trap', 'Hip-Hop', 'Drill', 'R&B', 'Afrobeats', 'Pop', 'Lo-Fi', 'Dancehall', 'Electronic', 'Boom Bap']
const MOODS = ['Dark', 'Energetic', 'Chill', 'Emotional', 'Aggressive', 'Uplifting', 'Mysterious', 'Romantic']
const STYLES = ['Modern', 'Old School', 'Club', 'Underground', 'Mainstream', 'Cinematic', 'Minimal', 'Heavy']

type Status = 'idle' | 'generating' | 'done' | 'error'

interface GenStatus {
  plan: 'free' | 'pro'
  isPro: boolean
  remaining: number
  limit: number
  canGenerate: boolean
}

export default function GeneratePage() {
  const { user, isLoading } = useAuthStore()
  const router = useRouter()
  const [form, setForm] = useState({ genre: '', mood: '', bpm: '', style: '', title: '', key: '', prompt: '' })
  const [showMore, setShowMore] = useState(false)
  const [beatType, setBeatType] = useState<'audio' | 'midi'>('audio')
  const [duration, setDuration] = useState<number | null>(null) // null = random 2-4min
  const [status, setStatus] = useState<Status>('idle')
  const [beatId, setBeatId] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [genStatus, setGenStatus] = useState<GenStatus | null>(null)
  const [generatedPlan, setGeneratedPlan] = useState<string>('free')
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [recentBeats, setRecentBeats] = useState<any[]>([])
  const [playingId, setPlayingId] = useState<string | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [referenceAudio, setReferenceAudio] = useState<File | null>(null)
  const [referenceStrength, setReferenceStrength] = useState(0.5)
  const refInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    if (!isLoading && !user) router.push('/login?redirect=/generate')
  }, [user, isLoading])

  useEffect(() => {
    if (user) {
      beatsApi.generationStatus().then(res => setGenStatus(res.data)).catch(() => {})
      beatsApi.myBeats({ limit: 10, sort: 'newest' }).then(res => setRecentBeats(res.data?.beats || res.data || [])).catch(() => {})
    }
  }, [user])

  useEffect(() => {
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  }, [])

  const handleGenerate = async () => {
    if (!form.genre && !form.prompt) return toast.error('Please provide a genre or prompt')
    if (!user) return toast.error('Please log in to generate beats')

    setStatus('generating')
    setProgress(0)

    try {
      // Encode reference audio as base64 if provided
      let referenceAudioB64: string | undefined
      if (referenceAudio) {
        const buf = await referenceAudio.arrayBuffer()
        referenceAudioB64 = btoa(Array.from(new Uint8Array(buf)).map(b => String.fromCharCode(b)).join(''))
      }

      const res = await beatsApi.generate({
        genre: form.genre,
        mood: form.mood || undefined,
        bpm: form.bpm ? parseInt(form.bpm) : undefined,
        style: form.style || undefined,
        title: form.title || undefined,
        key: form.key || undefined,
        prompt: form.prompt || undefined,
        beat_type: beatType,
        duration: duration ?? undefined,
        referenceAudio: referenceAudioB64,
        referenceStrength: referenceAudio ? referenceStrength : undefined,
      })

      const newBeatId = res.data.beatId
      setBeatId(newBeatId)
      setGeneratedPlan(res.data.plan || 'free')
      if (genStatus) {
        setGenStatus(prev => prev ? { ...prev, remaining: res.data.remaining ?? prev.remaining - 1 } : prev)
      }

      // Animate progress
      let p = 0
      const progressInterval = setInterval(() => {
        p = Math.min(p + Math.random() * 8, 85)
        setProgress(Math.floor(p))
      }, 800)

      // Poll for completion
      const checkStatus = async () => {
        try {
          const statusRes = await beatsApi.status(newBeatId)
          const s = statusRes.data.status
          if (s === 'personal' || s === 'published') {
            clearInterval(pollRef.current!)
            clearInterval(progressInterval)
            setProgress(100)
            setStatus('done')
            toast.success('Beat generated! 🎵')
            beatsApi.myBeats({ limit: 10, sort: 'newest' }).then(res => setRecentBeats(res.data?.beats || res.data || [])).catch(() => {})
          } else if (s === 'draft') {
            clearInterval(pollRef.current!)
            clearInterval(progressInterval)
            setStatus('error')
            toast.error('Generation failed. Please try again.')
          }
        } catch {}
      }
      checkStatus() // poll immediately once
      pollRef.current = setInterval(checkStatus, 2000)

    } catch (err: any) {
      setStatus('error')
      toast.error(err.response?.data?.error || 'Failed to start generation')
    }
  }

  const goToBeat = () => { if (beatId) router.push(`/beat/${beatId}`) }
  const reset = () => { setStatus('idle'); setProgress(0); setBeatId(null) }

  const handlePlayBeat = (beat: any) => {
    const url = beat.preview_url || beat.mp3_url || beat.wav_url
    if (!url) return
    if (playingId === beat.id) {
      audioRef.current?.pause(); setPlayingId(null); audioRef.current = null; return
    }
    audioRef.current?.pause()
    const a = new Audio(url)
    a.play().catch(() => {})
    a.onended = () => { setPlayingId(null); audioRef.current = null }
    audioRef.current = a; setPlayingId(beat.id)
  }

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-6xl mx-auto px-4 py-24 pt-32">

        {/* Header — full width above the flex row */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-forge mb-4 shadow-forge">
            <Zap size={28} className="text-white" />
          </div>
          <h1 className="font-display text-5xl text-white mb-3 tracking-wider">GENERATE BEAT</h1>
          <p className="text-forge-muted">Describe your sound. AI does the rest.</p>
        </div>

      <div className="flex gap-6 items-start">
      <div className="flex-1 min-w-0">

        {/* Generation quota banner */}
        {genStatus && status === 'idle' && (
          <div className={clsx(
            'mb-6 rounded-xl border p-4 flex items-center justify-between gap-4',
            genStatus.isPro
              ? 'bg-forge-gold/10 border-forge-gold/30'
              : genStatus.remaining <= 2
              ? 'bg-forge-accent/10 border-forge-accent/40'
              : 'bg-forge-dark border-forge-border'
          )}>
            <div className="flex items-center gap-3">
              {genStatus.isPro ? (
                <Crown size={16} className="text-forge-gold flex-shrink-0" />
              ) : (
                <Zap size={16} className={genStatus.remaining <= 2 ? 'text-forge-accent' : 'text-forge-muted'} />
              )}
              <div>
                <span className={clsx(
                  'text-sm font-semibold',
                  genStatus.isPro ? 'text-forge-gold' : genStatus.remaining <= 2 ? 'text-forge-accent' : 'text-forge-text'
                )}>
                  {genStatus.isPro ? 'Pro Plan' : 'Free Plan'}
                </span>
                <span className="text-forge-muted text-sm ml-2">
                  {genStatus.remaining} / {genStatus.limit} beats remaining
                  {genStatus.isPro && ' this month'}
                </span>
              </div>
            </div>

            {/* Progress bar */}
            <div className="hidden sm:flex items-center gap-3 flex-shrink-0">
              <div className="w-24 h-1.5 bg-forge-dark rounded-full overflow-hidden">
                <div
                  className={clsx(
                    'h-full rounded-full transition-all',
                    genStatus.isPro ? 'bg-forge-gold' : genStatus.remaining <= 2 ? 'bg-forge-accent' : 'bg-forge-accent'
                  )}
                  style={{ width: `${Math.max(0, 100 - (genStatus.remaining / genStatus.limit) * 100)}%` }}
                />
              </div>
              {!genStatus.isPro && (
                <Link href="/pricing" className="text-xs text-forge-accent font-bold flex items-center gap-1 hover:underline">
                  <Crown size={11} /> Upgrade
                </Link>
              )}
            </div>
          </div>
        )}

        {/* Limit reached banner */}
        {genStatus && genStatus.remaining === 0 && status === 'idle' && (
          <div className="mb-6 card p-8 text-center border-2 border-forge-accent/40">
            <Lock size={32} className="text-forge-accent mx-auto mb-3" />
            <h3 className="font-display text-xl text-white mb-2">
              {genStatus.isPro ? 'Monthly Limit Reached' : 'Free Limit Reached'}
            </h3>
            <p className="text-forge-muted text-sm mb-4">
              {genStatus.isPro
                ? 'You\'ve used all 500 beats this month. Resets on your next billing date.'
                : 'You\'ve used your 10 free beats. Upgrade to Pro for 500 beats/month and sell them with licenses.'}
            </p>
            {!genStatus.isPro && (
              <Link href="/pricing" className="btn-primary px-8 py-3 inline-flex items-center gap-2">
                <Crown size={16} /> Get Pro — €10/month
              </Link>
            )}
          </div>
        )}

        {/* Form */}
        {status === 'idle' && (
          <div className="space-y-4">
            {/* Custom Prompt / Lyrics Mode */}
            <div className="card p-6 space-y-4">
              <div className="flex items-center justify-between mb-2">
                <label className="label-forge mb-0 flex items-center gap-2">
                  <Music2 size={16} className="text-forge-accent" />
                  Description / Prompt
                </label>
                <span className="text-[10px] text-forge-muted uppercase tracking-widest">Optional</span>
              </div>
              <textarea
                placeholder="Write a prompt for your beat — e.g. 'Chill lo-fi hip hop with jazz piano and a steady boom bap drum pattern'..."
                value={form.prompt}
                onChange={e => setForm(f => ({ ...f, prompt: e.target.value }))}
                className="input-forge min-h-[120px] resize-none text-sm leading-relaxed"
              />
            </div>

            {/* Styles Section */}
            <div className="card p-6 space-y-4">
              <div className="flex items-center justify-between mb-2">
                <label className="label-forge mb-0 flex items-center gap-2">
                  <Zap size={16} className="text-forge-gold" />
                  Styles / Genre
                </label>
                <span className="text-[10px] text-forge-muted uppercase tracking-widest">Required</span>
              </div>
              <input
                type="text"
                placeholder="e.g. trap, synthwave, drill, dark, cinematic..."
                value={form.genre}
                onChange={e => setForm(f => ({ ...f, genre: e.target.value }))}
                className="input-forge text-sm"
              />
              
              <div className="flex flex-wrap gap-2 mt-2">
                {['trap', 'drill', 'hip hop', 'r&b', 'lo-fi'].map(tag => (
                  <button 
                    key={tag}
                    onClick={() => setForm(f => ({ ...f, genre: f.genre ? `${f.genre}, ${tag}` : tag }))}
                    className="text-[10px] bg-forge-dark border border-forge-border px-3 py-1 rounded-full text-forge-muted hover:text-forge-accent hover:border-forge-accent transition-colors uppercase tracking-widest"
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>

            {/* More Options Section */}
            <div className="card overflow-hidden">
              <button 
                onClick={() => setShowMore(!showMore)}
                className="w-full flex items-center justify-between p-6 hover:bg-forge-card/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <ChevronDown size={16} className={clsx("text-forge-muted transition-transform duration-300", showMore && "rotate-180")} />
                  <span className="text-sm font-semibold text-forge-text">More Options</span>
                </div>
                <span className="text-[10px] text-forge-muted uppercase tracking-widest">BPM, Key, Title</span>
              </button>

              {showMore && (
                <div className="px-6 pb-6 space-y-6 animate-fade-in">
                  {/* Beat Type toggle */}
                  <div>
                    <label className="label-forge flex items-center gap-2">
                      Beat Type
                    </label>
                    <div className="flex gap-2 mt-1">
                      <button
                        onClick={() => setBeatType('audio')}
                        className={clsx(
                          'flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl border text-sm font-semibold transition-all',
                          beatType === 'audio'
                            ? 'bg-forge-accent/15 border-forge-accent text-forge-accent'
                            : 'bg-forge-dark border-forge-border text-forge-muted hover:border-forge-text'
                        )}
                      >
                        <FileAudio size={15} />
                        Audio (MP3/WAV)
                      </button>
                      <button
                        onClick={() => setBeatType('midi')}
                        className={clsx(
                          'flex-1 flex items-center justify-center gap-2 py-2.5 rounded-xl border text-sm font-semibold transition-all',
                          beatType === 'midi'
                            ? 'bg-forge-gold/15 border-forge-gold text-forge-gold'
                            : 'bg-forge-dark border-forge-border text-forge-muted hover:border-forge-text'
                        )}
                      >
                        <FileCode2 size={15} />
                        MIDI
                      </button>
                    </div>
                  </div>

                  {/* Duration slider */}
                  <div>
                    <label className="label-forge flex items-center gap-2">
                      <Timer size={14} className="text-forge-muted" />
                      Duration
                      <span className="ml-auto text-forge-accent font-mono text-xs">
                        {duration === null
                          ? 'Random (2–4 min)'
                          : `${Math.floor(duration / 60)}:${String(duration % 60).padStart(2, '0')}`}
                      </span>
                    </label>
                    <input
                      type="range"
                      min={90}
                      max={240}
                      step={15}
                      value={duration ?? 165}
                      onChange={e => setDuration(parseInt(e.target.value))}
                      className="w-full accent-forge-accent mt-1"
                    />
                    <div className="flex justify-between text-[10px] text-forge-muted mt-1">
                      <span>1:30</span>
                      <button
                        onClick={() => setDuration(null)}
                        className={clsx(
                          'text-[10px] uppercase tracking-widest transition-colors',
                          duration === null ? 'text-forge-accent' : 'text-forge-muted hover:text-forge-accent'
                        )}
                      >
                        Random
                      </button>
                      <span>4:00</span>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="label-forge">BPM</label>
                      <input
                        type="number"
                        placeholder="e.g. 140"
                        value={form.bpm}
                        onChange={e => setForm(f => ({ ...f, bpm: e.target.value }))}
                        className="input-forge"
                        min={60} max={200}
                      />
                    </div>
                    <div>
                      <label className="label-forge">Musical Key</label>
                      <input
                        type="text"
                        placeholder="e.g. C minor, G major"
                        value={form.key}
                        onChange={e => setForm(f => ({ ...f, key: e.target.value }))}
                        className="input-forge"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="label-forge">Beat Title (optional)</label>
                    <input
                      type="text"
                      placeholder="Give your beat a name..."
                      value={form.title}
                      onChange={e => setForm(f => ({ ...f, title: e.target.value }))}
                      className="input-forge"
                      maxLength={100}
                    />
                  </div>

                  {/* Reference Audio */}
                  <div>
                    <label className="label-forge flex items-center gap-2">
                      <Upload size={14} className="text-forge-muted" />
                      Reference Audio
                      <span className="text-[10px] text-forge-muted font-normal normal-case tracking-normal ml-1">— AI mimics the style of your track</span>
                    </label>
                    <input
                      ref={refInputRef}
                      type="file"
                      accept="audio/wav,audio/mp3,audio/mpeg,audio/ogg,audio/flac"
                      className="hidden"
                      onChange={e => setReferenceAudio(e.target.files?.[0] ?? null)}
                    />
                    {referenceAudio ? (
                      <div className="flex items-center gap-3 bg-forge-dark border border-forge-accent/40 rounded-xl px-4 py-3">
                        <FileAudio size={16} className="text-forge-accent flex-shrink-0" />
                        <span className="text-sm text-forge-text truncate flex-1">{referenceAudio.name}</span>
                        <button onClick={() => setReferenceAudio(null)} className="text-forge-muted hover:text-forge-accent">
                          <X size={14} />
                        </button>
                      </div>
                    ) : (
                      <button
                        onClick={() => refInputRef.current?.click()}
                        className="w-full border-2 border-dashed border-forge-border hover:border-forge-accent/50 rounded-xl py-4 text-sm text-forge-muted hover:text-forge-accent transition-colors flex items-center justify-center gap-2"
                      >
                        <Upload size={14} /> Upload WAV / MP3 (optional)
                      </button>
                    )}
                    {referenceAudio && (
                      <div className="mt-3">
                        <div className="flex justify-between text-xs text-forge-muted mb-1">
                          <span>Style influence</span>
                          <span className="text-forge-accent font-mono">{Math.round(referenceStrength * 100)}%</span>
                        </div>
                        <input
                          type="range" min={0.1} max={0.9} step={0.1}
                          value={referenceStrength}
                          onChange={e => setReferenceStrength(parseFloat(e.target.value))}
                          className="w-full accent-forge-accent"
                        />
                        <div className="flex justify-between text-[10px] text-forge-muted mt-1">
                          <span>Subtle</span><span>Strong</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Preview of prompt */}
            {(form.genre || form.prompt) && (
              <div className="bg-forge-dark/50 rounded-lg p-4 border border-forge-border/50 animate-fade-in">
                <p className="text-xs text-forge-muted font-mono leading-relaxed">
                  Forging: <span className="text-forge-cyan">
                    {[form.genre, form.prompt, form.bpm && `${form.bpm} BPM`, form.key && `Key: ${form.key}`]
                      .filter(Boolean).join(' • ')}
                  </span>
                </p>
              </div>
            )}

            <button
              onClick={handleGenerate}
              disabled={(!form.genre && !form.prompt) || genStatus?.remaining === 0}
              className="w-full btn-primary py-4 text-lg flex items-center justify-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed shadow-forge group"
            >
              <Zap size={20} className="group-hover:animate-pulse" />
              Forge This Beat
            </button>
          </div>
        )}

        {/* Generating state */}
        {status === 'generating' && (
          <div className="card p-10 text-center">
            <div className="relative w-24 h-24 mx-auto mb-8">
              <div className="absolute inset-0 rounded-full border-2 border-forge-accent/20" />
              <div
                className="absolute inset-0 rounded-full border-2 border-forge-accent border-t-transparent animate-spin"
                style={{ animationDuration: '1.5s' }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <Music2 size={32} className="text-forge-accent" />
              </div>
            </div>

            <h2 className="font-display text-3xl text-white mb-2">FORGING YOUR BEAT</h2>
            <p className="text-forge-muted mb-8">AI is composing your instrumental...</p>

            {/* Progress bar */}
            <div className="bg-forge-dark rounded-full h-2 mb-2 overflow-hidden">
              <div
                className="h-full bg-gradient-forge rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-forge-muted text-sm font-mono">{progress}%</p>

            {/* Animated waveform */}
            <div className="flex items-end justify-center gap-1 h-12 mt-8">
              {[...Array(20)].map((_, i) => (
                <div
                  key={i}
                  className="w-2 bg-forge-accent/60 rounded-full"
                  style={{
                    animation: `waveform ${0.8 + Math.random() * 0.8}s ease-in-out infinite`,
                    animationDelay: `${i * 0.08}s`,
                  }}
                />
              ))}
            </div>
          </div>
        )}

        {/* Done state */}
        {status === 'done' && (
          <div className="card p-10 text-center">
            <div className="w-20 h-20 rounded-full bg-forge-green/20 border-2 border-forge-green flex items-center justify-center mx-auto mb-6">
              <CheckCircle size={36} className="text-forge-green" />
            </div>
            <h2 className="font-display text-3xl text-white mb-2">BEAT GENERATED!</h2>
            <p className="text-forge-muted mb-4">Your beat is saved. Publish it to the marketplace from your dashboard.</p>
            <div className={clsx(
              'rounded-xl p-4 mb-4 text-sm',
              generatedPlan === 'pro'
                ? 'bg-forge-gold/10 border border-forge-gold/30'
                : 'bg-forge-accent/10 border border-forge-accent/30'
            )}>
              <p className="text-forge-muted">
                {generatedPlan === 'pro'
                  ? <><span className="text-forge-gold font-semibold">Pro plan:</span> Go to your <Link href="/dashboard?tab=beats" className="text-forge-gold underline">Dashboard → Beats</Link> and click the publish button to put it on the marketplace.</>
                  : <><span className="text-forge-accent font-semibold">Free plan:</span> This beat is saved for personal use only. <Link href="/pricing" className="text-forge-accent underline">Upgrade to Pro</Link> to publish and sell.</>
                }
              </p>
            </div>
            <div className="flex gap-3">
              <button onClick={goToBeat} className="flex-1 btn-primary py-3">
                View Beat Page
              </button>
              <button onClick={reset} className="flex-1 btn-secondary py-3">
                Generate Another
              </button>
            </div>
          </div>
        )}

        {/* Error state */}
        {status === 'error' && (
          <div className="card p-10 text-center">
            <div className="w-20 h-20 rounded-full bg-forge-accent/20 border-2 border-forge-accent flex items-center justify-center mx-auto mb-6">
              <Zap size={36} className="text-forge-accent" />
            </div>
            <h2 className="font-display text-3xl text-white mb-2">GENERATION FAILED</h2>
            <p className="text-forge-muted mb-8">Something went wrong. Please try again.</p>
            <button onClick={reset} className="btn-primary px-8 py-3">Try Again</button>
          </div>
        )}
      </div>{/* end form column */}

      {/* Recent Beats sidebar */}
      <div className="hidden lg:flex flex-col w-80 flex-shrink-0 sticky top-28">
        <div className="card overflow-hidden">
          <div className="px-4 py-3 border-b border-forge-border flex items-center gap-2">
            <Clock size={14} className="text-forge-muted" />
            <span className="text-sm font-semibold text-forge-text">Recent Beats</span>
            <span className="text-xs text-forge-muted ml-auto">{recentBeats.length > 0 ? `${Math.min(recentBeats.length, 10)} beats` : 'Last 10'}</span>
          </div>
          {recentBeats.length === 0 ? (
            <div className="p-8 text-center text-forge-muted text-sm">
              <Music2 size={28} className="mx-auto mb-3 opacity-40" />
              <p className="font-medium text-forge-text mb-1">No beats yet</p>
              <p className="text-xs">Generate your first beat and it will appear here.</p>
            </div>
          ) : (
            <div className="divide-y divide-forge-border/50 max-h-[calc(100vh-12rem)] overflow-y-auto">
              {recentBeats.slice(0, 10).map((beat: any) => (
                <div key={beat.id} className="flex items-center gap-3 px-4 py-3 hover:bg-forge-dark/50 transition-colors group">
                  {/* Cover / play button */}
                  <button
                    onClick={() => handlePlayBeat(beat)}
                    className="w-11 h-11 rounded-lg flex-shrink-0 relative overflow-hidden border border-forge-border bg-forge-dark"
                    title={playingId === beat.id ? 'Stop' : 'Play'}
                  >
                    {beat.cover_art_url ? (
                      <img src={beat.cover_art_url} alt="" className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full bg-gradient-forge flex items-center justify-center">
                        <Music2 size={14} className="text-white" />
                      </div>
                    )}
                    <div className={clsx(
                      'absolute inset-0 bg-black/60 flex items-center justify-center transition-opacity',
                      playingId === beat.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                    )}>
                      {playingId === beat.id
                        ? <Square size={12} className="text-white fill-white" />
                        : <Play size={12} className="text-white fill-white" />
                      }
                    </div>
                    {playingId === beat.id && (
                      <div className="absolute bottom-1 left-0 right-0 flex gap-px justify-center">
                        {[1,2,3].map(i => (
                          <div key={i} className="w-0.5 bg-forge-accent rounded-full animate-bounce"
                            style={{ height: 4 + i * 2, animationDelay: `${i * 0.1}s` }} />
                        ))}
                      </div>
                    )}
                  </button>
                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <Link href={`/beat/${beat.id}`}
                      className="text-sm font-medium text-forge-text hover:text-white transition-colors truncate block leading-tight">
                      {beat.title}
                    </Link>
                    <div className="text-xs text-forge-muted truncate mt-0.5">
                      {beat.genre && <span>{beat.genre}</span>}
                      {beat.bpm && <span> · {beat.bpm} BPM</span>}
                    </div>
                    <div className="text-[10px] text-forge-muted/60 mt-0.5 uppercase tracking-widest">
                      {beat.status === 'published' ? (
                        <span className="text-forge-green">Published</span>
                      ) : beat.status === 'generating' ? (
                        <span className="text-forge-gold">Generating...</span>
                      ) : (
                        <span>Personal</span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
          {recentBeats.length > 0 && (
            <div className="px-4 py-3 border-t border-forge-border">
              <Link href="/dashboard?tab=beats" className="text-xs text-forge-accent hover:underline flex items-center gap-1">
                View all in Dashboard →
              </Link>
            </div>
          )}
        </div>
      </div>

      </div>{/* end flex row */}
      </div>
    </div>
  )
}
