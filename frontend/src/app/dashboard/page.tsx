'use client'
import { useEffect, useState, Suspense, useRef } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { dashboardApi, beatsApi, studioApi } from '@/lib/api'
import toast from 'react-hot-toast'
import {
  Music2, DollarSign, Play, Loader2,
  Wallet, ArrowUpRight, ExternalLink, BarChart2, ShoppingBag, Globe,
  MoreVertical, Pencil, ImagePlus, Trash2, Sliders, X, Check,
  ChevronDown, ChevronUp, Users, GitBranch, Clock
} from 'lucide-react'
import Link from 'next/link'
import clsx from 'clsx'

type Tab = 'overview' | 'beats' | 'sales' | 'payouts'

export default function DashboardPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-forge-black flex items-center justify-center"><div className="w-8 h-8 rounded-full border-2 border-forge-accent border-t-transparent animate-spin" /></div>}>
      <DashboardContent />
    </Suspense>
  )
}

function DashboardContent() {
  const { user, isLoading, initialize } = useAuthStore()
  const router = useRouter()
  const searchParams = useSearchParams()
  const [tab, setTab] = useState<Tab>((searchParams.get('tab') as Tab) || 'overview')
  const [overview, setOverview] = useState<any>(null)
  const [beats, setBeats] = useState<any[]>([])
  const [sales, setSales] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [withdrawAmount, setWithdrawAmount] = useState('')
  const [withdrawing, setWithdrawing] = useState(false)
  const [connectingStripe, setConnectingStripe] = useState(false)
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  const [menuPos, setMenuPos] = useState<{ top: number; right: number } | null>(null)
  const [renameModal, setRenameModal] = useState<{ beatId: string; title: string } | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [coverArtBeatId, setCoverArtBeatId] = useState<string | null>(null)
  const coverArtInputRef = useRef<HTMLInputElement>(null)
  const [expandedBeat, setExpandedBeat] = useState<string | null>(null)
  const [beatVersions, setBeatVersions] = useState<Record<string, any[]>>({})
  const [versionsLoading, setVersionsLoading] = useState<string | null>(null)
  const [collabBeats, setCollabBeats] = useState<any[]>([])

  useEffect(() => {
    const close = () => { setOpenMenu(null); setMenuPos(null) }
    window.addEventListener('click', close)
    window.addEventListener('scroll', close, true)
    return () => { window.removeEventListener('click', close); window.removeEventListener('scroll', close, true) }
  }, [])

  useEffect(() => {
    initialize()
  }, [])

  useEffect(() => {
    if (isLoading) return
    if (user === null) { router.push('/login?redirect=/dashboard'); return }
    if (searchParams.get('connected') === 'true') {
      toast.success('Stripe account connected!')
      initialize() // refresh user to get stripe_account_id
    }
    loadData()
  }, [user, isLoading])

  const loadData = async () => {
    setLoading(true)
    try {
      const [overviewRes, beatsRes, salesRes, collabRes] = await Promise.all([
        dashboardApi.overview(),
        dashboardApi.beats(),
        dashboardApi.sales(),
        studioApi.myCollabs(),
      ])
      setOverview(overviewRes.data)
      setBeats(beatsRes.data)
      setSales(salesRes.data)
      setCollabBeats(collabRes.data)
    } finally {
      setLoading(false)
    }
  }

  const toggleVersions = async (beatId: string) => {
    if (expandedBeat === beatId) { setExpandedBeat(null); return }
    setExpandedBeat(beatId)
    if (!beatVersions[beatId]) {
      setVersionsLoading(beatId)
      try {
        const res = await studioApi.listVersions(beatId)
        setBeatVersions(prev => ({ ...prev, [beatId]: res.data }))
      } catch {}
      finally { setVersionsLoading(null) }
    }
  }

  const handleWithdraw = async () => {
    const cents = Math.round(parseFloat(withdrawAmount) * 100)
    if (!cents || cents < 2000) return toast.error('Minimum withdrawal is $20')
    setWithdrawing(true)
    try {
      await dashboardApi.withdraw(cents)
      toast.success('Withdrawal initiated!')
      setWithdrawAmount('')
      initialize()
    } catch (err: any) {
      if (err.response?.data?.action === 'connect_stripe') {
        toast.error('Please connect Stripe first')
      } else {
        toast.error(err.response?.data?.error || 'Withdrawal failed')
      }
    } finally {
      setWithdrawing(false)
    }
  }

  const handleConnectStripe = async () => {
    setConnectingStripe(true)
    try {
      const res = await dashboardApi.connectStripe()
      window.location.href = res.data.url
    } catch {
      toast.error('Failed to connect Stripe')
      setConnectingStripe(false)
    }
  }

  const handleRename = async () => {
    if (!renameModal || !renameValue.trim()) return
    try {
      await beatsApi.update(renameModal.beatId, { title: renameValue.trim() })
      toast.success('Beat renamed')
      setRenameModal(null)
      loadData()
    } catch {
      toast.error('Failed to rename beat')
    }
  }

  const handleCoverArtChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !coverArtBeatId) return
    const reader = new FileReader()
    reader.onload = async (ev) => {
      const base64 = ev.target?.result as string
      try {
        await beatsApi.uploadCoverArt(coverArtBeatId, base64)
        toast.success('Cover art updated!')
        setCoverArtBeatId(null)
        loadData()
      } catch {
        toast.error('Failed to upload cover art')
      }
    }
    reader.readAsDataURL(file)
    e.target.value = ''
  }

  const handlePublishToggle = async (beat: any) => {
    try {
      if (beat.status === 'published') {
        await beatsApi.unpublish(beat.id)
        toast.success('Beat unpublished')
      } else {
        await beatsApi.publish(beat.id)
        toast.success('Beat published to marketplace!')
      }
      loadData()
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Action failed')
    }
  }

  const handleDelete = async (beatId: string) => {
    if (!confirm('Delete this beat? This cannot be undone.')) return
    try {
      await beatsApi.delete(beatId)
      toast.success('Beat deleted')
      loadData()
    } catch {
      toast.error('Failed to delete beat')
    }
  }

  if (!user || loading) {
    return (
      <div className="min-h-screen bg-forge-black flex items-center justify-center">
        <Loader2 size={32} className="animate-spin text-forge-accent" />
      </div>
    )
  }

  const tabs: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'overview', label: 'Overview', icon: <BarChart2 size={16} /> },
    { id: 'beats', label: 'My Beats', icon: <Music2 size={16} /> },
    { id: 'sales', label: 'Sales', icon: <ShoppingBag size={16} /> },
    { id: 'payouts', label: 'Payouts', icon: <Wallet size={16} /> },
  ]

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20">

        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="font-display text-4xl text-white">DASHBOARD</h1>
            <p className="text-forge-muted mt-1">Welcome back, <span className="text-forge-text">{user.display_name || user.username}</span></p>
          </div>
          <Link href="/generate" className="btn-primary flex items-center gap-2">
            <Music2 size={16} />
            New Beat
          </Link>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 bg-forge-dark rounded-xl p-1 mb-8 overflow-x-auto">
          {tabs.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap flex-shrink-0',
                tab === t.id
                  ? 'bg-forge-card text-forge-text shadow-sm'
                  : 'text-forge-muted hover:text-forge-text'
              )}
            >
              {t.icon}
              {t.label}
            </button>
          ))}
        </div>

        {/* Overview tab */}
        {tab === 'overview' && overview && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: 'Total Beats', value: overview.beats?.total_beats || 0, icon: <Music2 size={18} className="text-forge-accent" />, sub: `${overview.beats?.published_beats || 0} published` },
                { label: 'Total Plays', value: (overview.beats?.total_plays || 0).toLocaleString(), icon: <Play size={18} className="text-forge-cyan" />, sub: 'all time' },
                { label: 'Total Sales', value: overview.sales?.total_sales || 0, icon: <ShoppingBag size={18} className="text-forge-gold" />, sub: 'licenses sold' },
                { label: 'Total Revenue', value: `€${((overview.sales?.total_revenue || 0) / 100).toFixed(2)}`, icon: <DollarSign size={18} className="text-forge-green" />, sub: '80% to you' },
              ].map(stat => (
                <div key={stat.label} className="stat-card">
                  <div className="flex items-center gap-2 mb-3">{stat.icon}</div>
                  <div className="font-display text-3xl text-white">{stat.value}</div>
                  <div className="text-forge-muted text-xs mt-1">{stat.sub}</div>
                  <div className="text-forge-text text-sm font-medium mt-0.5">{stat.label}</div>
                </div>
              ))}
            </div>

            {/* Balance card */}
            <div className="card p-6 border-forge-green/30 bg-forge-green/5">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-forge-muted text-sm">Available Balance</p>
                  <p className="font-display text-5xl text-forge-green mt-1">
                    €{((user.balance_cents || 0) / 100).toFixed(2)}
                  </p>
                  <p className="text-forge-muted text-xs mt-1">
                    Total earned: €{((overview.earnings?.total_earnings_cents || 0) / 100).toFixed(2)}
                  </p>
                </div>
                <button onClick={() => setTab('payouts')} className="btn-secondary flex items-center gap-2 text-sm">
                  <Wallet size={14} />
                  Withdraw
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Beats tab */}
        {tab === 'beats' && (
          <div className="space-y-3" onClick={() => setOpenMenu(null)}>
            {/* Hidden cover art input */}
            <input
              ref={coverArtInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleCoverArtChange}
            />

            {beats.length === 0 ? (
              <div className="card p-12 text-center">
                <Music2 size={40} className="text-forge-border mx-auto mb-4" />
                <p className="text-forge-muted mb-4">No beats yet. Generate your first!</p>
                <Link href="/generate" className="btn-primary inline-flex items-center gap-2">
                  Generate Beat
                </Link>
              </div>
            ) : beats.map(beat => (
              <div key={beat.id} className="card overflow-hidden">
                <div className="p-4 flex items-center gap-4">
                  {/* Cover art / icon */}
                  <div className="w-10 h-10 rounded-lg bg-gradient-forge flex items-center justify-center flex-shrink-0 overflow-hidden">
                    {beat.cover_art_url
                      ? <img src={beat.cover_art_url} alt="" className="w-full h-full object-cover" />
                      : <Music2 size={16} className="text-white" />
                    }
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-forge-text truncate">{beat.title}</p>
                      {beat.beat_type && beat.beat_type !== 'audio' && (
                        <span className="text-[9px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded bg-forge-gold/15 text-forge-gold border border-forge-gold/30 flex-shrink-0">
                          MIDI
                        </span>
                      )}
                    </div>
                    <p className="text-forge-muted text-xs mt-0.5">
                      {beat.genre} • {beat.bpm ? `${beat.bpm} BPM • ` : ''}{(beat.play_count || 0).toLocaleString()} plays
                    </p>
                  </div>

                  {/* Revenue */}
                  <div className="text-right flex-shrink-0 hidden sm:block">
                    <p className="text-forge-green font-medium text-sm">
                      €{((beat.earnings_cents || 0) / 100).toFixed(2)}
                    </p>
                    <p className="text-forge-muted text-xs">{beat.sales_count || 0} sales</p>
                  </div>

                  {/* Status badge */}
                  <div className={clsx(
                    'badge text-xs flex-shrink-0',
                    beat.status === 'published' ? 'badge-accent' : 'bg-forge-dark border border-forge-border text-forge-muted'
                  )}>
                    {beat.status}
                  </div>

                  {/* Beat page link */}
                  <Link href={`/beat/${beat.id}`} className="p-2 text-forge-muted hover:text-forge-text transition-colors flex-shrink-0">
                    <ExternalLink size={14} />
                  </Link>

                  {/* Versions toggle button */}
                  <button
                    onClick={e => { e.stopPropagation(); toggleVersions(beat.id) }}
                    className="p-2 rounded-lg text-forge-muted hover:text-forge-accent hover:bg-forge-dark transition-colors flex-shrink-0 flex items-center gap-1"
                    title="Version history"
                  >
                    <Clock size={14} />
                    {expandedBeat === beat.id ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </button>

                  {/* 3-dot menu */}
                  <div className="relative flex-shrink-0">
                    <button
                      onClick={e => {
                        e.stopPropagation()
                        if (openMenu === beat.id) { setOpenMenu(null); setMenuPos(null) }
                        else {
                          const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
                          setMenuPos({ top: rect.bottom + window.scrollY + 4, right: window.innerWidth - rect.right })
                          setOpenMenu(beat.id)
                        }
                      }}
                      className="p-2 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-dark transition-colors"
                    >
                      <MoreVertical size={16} />
                    </button>

                    {openMenu === beat.id && menuPos && (
                      <div
                        style={{ position: 'fixed', top: menuPos.top, right: menuPos.right }}
                        className="w-48 bg-forge-card border border-forge-border rounded-xl shadow-lg z-[9999] overflow-hidden"
                        onClick={e => e.stopPropagation()}
                      >
                        {/* Rename */}
                        <button
                          onClick={() => { setRenameModal({ beatId: beat.id, title: beat.title }); setRenameValue(beat.title); setOpenMenu(null) }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-forge-text hover:bg-forge-dark transition-colors"
                        >
                          <Pencil size={14} className="text-forge-muted" /> Rename
                        </button>

                        {/* Cover art */}
                        <button
                          onClick={() => { setCoverArtBeatId(beat.id); setOpenMenu(null); setTimeout(() => coverArtInputRef.current?.click(), 50) }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-forge-text hover:bg-forge-dark transition-colors"
                        >
                          <ImagePlus size={14} className="text-forge-muted" /> Change Cover Art
                        </button>

                        {/* Publish / Unpublish */}
                        <button
                          onClick={() => { handlePublishToggle(beat); setOpenMenu(null) }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-forge-text hover:bg-forge-dark transition-colors"
                        >
                          <Globe size={14} className="text-forge-muted" />
                          {beat.status === 'published' ? 'Unpublish' : 'Publish to Market'}
                        </button>

                        {/* Studio */}
                        {beat.status === 'generating' ? (
                          <span className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-forge-muted cursor-not-allowed">
                            <Sliders size={14} /> Studio (Generating…)
                          </span>
                        ) : (
                          <Link
                            href={`/studio/${beat.id}`}
                            className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-forge-accent hover:bg-forge-dark transition-colors"
                            onClick={() => setOpenMenu(null)}
                          >
                            <Sliders size={14} /> Open Studio
                          </Link>
                        )}

                        <div className="border-t border-forge-border" />

                        {/* Delete */}
                        <button
                          onClick={() => { handleDelete(beat.id); setOpenMenu(null) }}
                          className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-400 hover:bg-forge-dark transition-colors"
                        >
                          <Trash2 size={14} /> Delete Beat
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {/* Versions panel */}
                {expandedBeat === beat.id && (
                  <div className="border-t border-forge-border bg-forge-dark/50 px-4 py-3">
                    <div className="flex items-center gap-1.5 mb-3">
                      <GitBranch size={13} className="text-forge-accent" />
                      <span className="text-xs font-medium text-forge-muted uppercase tracking-wider">Version History</span>
                    </div>
                    {versionsLoading === beat.id ? (
                      <div className="flex items-center justify-center py-4">
                        <Loader2 size={18} className="animate-spin text-forge-accent" />
                      </div>
                    ) : !beatVersions[beat.id] || beatVersions[beat.id].length === 0 ? (
                      <p className="text-forge-muted text-xs py-2">No saved versions yet</p>
                    ) : (
                      <div className="space-y-1.5">
                        {beatVersions[beat.id].map(version => (
                          <div key={version.id} className="flex items-center gap-3 py-1.5">
                            <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-forge-accent/15 border border-forge-accent/30 text-forge-accent flex-shrink-0">
                              v{version.version_number}
                            </span>
                            <span className="text-sm text-forge-text truncate flex-1">
                              {version.label || 'Auto-save'}
                            </span>
                            <span className="text-xs text-forge-muted flex-shrink-0 hidden sm:block">
                              by @{version.saved_by_username}
                            </span>
                            <span className="text-xs text-forge-muted flex-shrink-0">
                              {new Date(version.created_at).toLocaleDateString()}
                            </span>
                            <Link
                              href={`/studio/${beat.id}?version=${version.id}`}
                              className="btn-secondary text-xs px-2 py-1 flex-shrink-0"
                            >
                              Open
                            </Link>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Collaboration Beats */}
            {collabBeats.length > 0 && (
              <div className="mt-8">
                <div className="flex items-center gap-2 mb-4">
                  <Users size={16} className="text-forge-accent" />
                  <h3 className="font-display text-lg text-white">Collaboration Beats</h3>
                </div>
                {collabBeats.map(beat => (
                  <div key={beat.id} className="card p-4 flex items-center gap-4 mb-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-forge flex items-center justify-center flex-shrink-0 overflow-hidden">
                      {beat.cover_art_url ? <img src={beat.cover_art_url} alt="" className="w-full h-full object-cover" /> : <Music2 size={16} className="text-white" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-forge-text truncate">{beat.title}</p>
                      <p className="text-forge-muted text-xs mt-0.5">
                        {beat.genre} • by @{beat.creator_username} • {beat.bpm ? `${beat.bpm} BPM` : ''}
                      </p>
                    </div>
                    <span className="badge text-xs bg-forge-accent/15 border border-forge-accent/30 text-forge-accent flex-shrink-0">Collab</span>
                    <Link href={`/studio/${beat.id}`} className="btn-secondary text-xs flex items-center gap-1.5 flex-shrink-0">
                      <Sliders size={12} /> Open Studio
                    </Link>
                  </div>
                ))}
              </div>
            )}

            {/* Rename modal */}
            {renameModal && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setRenameModal(null)}>
                <div className="bg-forge-card border border-forge-border rounded-2xl p-6 w-full max-w-sm shadow-xl mx-4" onClick={e => e.stopPropagation()}>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="font-display text-lg text-white">Rename Beat</h3>
                    <button onClick={() => setRenameModal(null)} className="text-forge-muted hover:text-forge-text">
                      <X size={18} />
                    </button>
                  </div>
                  <input
                    autoFocus
                    value={renameValue}
                    onChange={e => setRenameValue(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleRename()}
                    className="input-forge w-full mb-4"
                    maxLength={100}
                  />
                  <div className="flex gap-2">
                    <button onClick={() => setRenameModal(null)} className="flex-1 btn-secondary py-2">Cancel</button>
                    <button onClick={handleRename} className="flex-1 btn-primary py-2 flex items-center justify-center gap-2">
                      <Check size={14} /> Save
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Sales tab */}
        {tab === 'sales' && (
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-forge-border">
                    <th className="text-left p-4 text-forge-muted text-xs font-medium uppercase tracking-wider">Beat</th>
                    <th className="text-left p-4 text-forge-muted text-xs font-medium uppercase tracking-wider">License</th>
                    <th className="text-left p-4 text-forge-muted text-xs font-medium uppercase tracking-wider">Buyer</th>
                    <th className="text-right p-4 text-forge-muted text-xs font-medium uppercase tracking-wider">Earnings</th>
                    <th className="text-right p-4 text-forge-muted text-xs font-medium uppercase tracking-wider">Date</th>
                  </tr>
                </thead>
                <tbody>
                  {sales.length === 0 ? (
                    <tr>
                      <td colSpan={5} className="text-center py-12 text-forge-muted">No sales yet</td>
                    </tr>
                  ) : sales.map(sale => (
                    <tr key={sale.id} className="border-b border-forge-border/50 hover:bg-forge-dark/50 transition-colors">
                      <td className="p-4 text-sm text-forge-text font-medium">{sale.beat_title}</td>
                      <td className="p-4"><span className="badge badge-accent text-[10px]">{sale.license_name}</span></td>
                      <td className="p-4 text-sm text-forge-muted">@{sale.buyer_username}</td>
                      <td className="p-4 text-right text-forge-green font-semibold">
                        +€{(sale.creator_earnings_cents / 100).toFixed(2)}
                      </td>
                      <td className="p-4 text-right text-forge-muted text-xs">
                        {new Date(sale.created_at).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Payouts tab */}
        {tab === 'payouts' && (
          <div className="space-y-4 max-w-lg">
            <div className="card p-6">
              <h3 className="font-semibold text-forge-text mb-1">Available Balance</h3>
              <p className="font-display text-4xl text-forge-green mb-4">
                €{((user.balance_cents || 0) / 100).toFixed(2)}
              </p>

              {!user.stripe_account_id ? (
                <div className="space-y-3">
                  <p className="text-forge-muted text-sm">Connect your Stripe account to withdraw earnings.</p>
                  <button
                    onClick={handleConnectStripe}
                    disabled={connectingStripe}
                    className="btn-primary w-full flex items-center justify-center gap-2"
                  >
                    {connectingStripe ? <Loader2 size={16} className="animate-spin" /> : <ArrowUpRight size={16} />}
                    Connect Stripe Account
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-forge-muted">$</span>
                      <input
                        type="number"
                        placeholder="0.00"
                        value={withdrawAmount}
                        onChange={e => setWithdrawAmount(e.target.value)}
                        className="input-forge pl-7"
                        min="20"
                        step="0.01"
                      />
                    </div>
                    <button
                      onClick={handleWithdraw}
                      disabled={withdrawing || !withdrawAmount}
                      className="btn-primary flex items-center gap-2 disabled:opacity-50 flex-shrink-0"
                    >
                      {withdrawing ? <Loader2 size={16} className="animate-spin" /> : <ArrowUpRight size={16} />}
                      Withdraw
                    </button>
                  </div>
                  <p className="text-forge-muted text-xs">Minimum: $20 • Arrives in 2–3 business days</p>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
