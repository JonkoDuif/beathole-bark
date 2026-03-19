'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { adminApi } from '@/lib/api'
import api from '@/lib/api'
import toast from 'react-hot-toast'
import {
  ShieldCheck, Users, Music2, DollarSign, Trash2, Loader2,
  Settings, BarChart2, Crown, ShoppingBag, UserX, UserCheck,
  Shield, ExternalLink, MessageSquare, ChevronDown, Zap, X, Check
} from 'lucide-react'
import Link from 'next/link'
import { supportApi } from '@/lib/api'
import clsx from 'clsx'

type Tab = 'analytics' | 'beats' | 'users' | 'orders' | 'tickets' | 'settings'

export default function AdminPage() {
  const { user, isLoading, initialize } = useAuthStore()
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('analytics')
  const [analytics, setAnalytics] = useState<any>(null)
  const [beats, setBeats] = useState<any[]>([])
  const [users, setUsers] = useState<any[]>([])
  const [orders, setOrders] = useState<any[]>([])
  const [settings, setSettings] = useState<any>({})
  const [tickets, setTickets] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [savingSettings, setSavingSettings] = useState(false)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [creditsModal, setCreditsModal] = useState<{ userId: string; username: string } | null>(null)
  const [creditsAmount, setCreditsAmount] = useState('10')

  useEffect(() => { initialize() }, [])
  useEffect(() => {
    if (isLoading) return
    if (user === null) { router.push('/'); return }
    if (user?.role !== 'admin') { router.push('/dashboard'); return }
    loadData()
  }, [user, isLoading])

  const loadData = async () => {
    setLoading(true)
    try {
      const [analyticsRes, beatsRes, usersRes, ordersRes, settingsRes, ticketsRes] = await Promise.all([
        adminApi.analytics(),
        adminApi.beats(),
        adminApi.users(),
        api.get('/admin/orders'),
        adminApi.getSettings(),
        supportApi.adminTickets(),
      ])
      setAnalytics(analyticsRes.data)
      setBeats(beatsRes.data)
      setUsers(usersRes.data)
      setOrders(ordersRes.data)
      setSettings(settingsRes.data)
      setTickets(ticketsRes.data)
    } finally {
      setLoading(false)
    }
  }

  const action = async (id: string, fn: () => Promise<any>, successMsg: string) => {
    setActionLoading(id)
    try {
      await fn()
      toast.success(successMsg)
      loadData()
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Action failed')
    } finally {
      setActionLoading(null)
    }
  }

  const handleGiveCredits = async () => {
    if (!creditsModal) return
    const amount = parseInt(creditsAmount)
    if (!amount || amount < 1) return toast.error('Enter a valid amount')
    setActionLoading('give_credits')
    try {
      await api.post(`/admin/users/${creditsModal.userId}/give-credits`, { credits: amount })
      toast.success(`${amount} credits given to @${creditsModal.username}`)
      setCreditsModal(null)
      setCreditsAmount('10')
      loadData()
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to give credits')
    } finally {
      setActionLoading(null)
    }
  }

  const handleSaveSettings = async () => {
    setSavingSettings(true)
    try {
      await adminApi.updateSettings({
        platformFeePercent: parseInt(settings.platform_fee_percent),
        minWithdrawalCents: parseInt(settings.min_withdrawal_cents),
        maxBeatsPerDay: parseInt(settings.max_beats_per_day),
        stripeProPriceId: settings.stripe_pro_price_id,
      })
      toast.success('Settings saved')
    } catch {
      toast.error('Failed to save settings')
    } finally {
      setSavingSettings(false)
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
    { id: 'analytics', label: 'Analytics', icon: <BarChart2 size={16} /> },
    { id: 'orders', label: 'Orders', icon: <ShoppingBag size={16} /> },
    { id: 'beats', label: 'Beats', icon: <Music2 size={16} /> },
    { id: 'users', label: 'Users', icon: <Users size={16} /> },
    { id: 'tickets', label: 'Support Tickets', icon: <MessageSquare size={16} /> },
    { id: 'settings', label: 'Settings', icon: <Settings size={16} /> },
  ]

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20">

        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-forge-accent/20 border border-forge-accent/30 flex items-center justify-center">
            <ShieldCheck size={20} className="text-forge-accent" />
          </div>
          <div>
            <h1 className="font-display text-4xl text-white">ADMIN PANEL</h1>
            <p className="text-forge-muted text-sm">Platform management</p>
          </div>
        </div>

        <div className="flex gap-1 bg-forge-dark rounded-xl p-1 mb-8 overflow-x-auto">
          {tabs.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap flex-shrink-0',
                tab === t.id ? 'bg-forge-card text-forge-text' : 'text-forge-muted hover:text-forge-text'
              )}
            >
              {t.icon}{t.label}
            </button>
          ))}
        </div>

        {/* Analytics */}
        {tab === 'analytics' && analytics && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              {[
                { label: 'Total Users', value: analytics.users?.total_users, icon: <Users size={18} className="text-forge-cyan" />, sub: `+${analytics.users?.new_users} this month` },
                { label: 'Total Beats', value: analytics.beats?.total_beats, icon: <Music2 size={18} className="text-forge-accent" />, sub: `${analytics.beats?.published_beats} live` },
                { label: 'Total Revenue', value: `€${((analytics.revenue?.total_revenue || 0) / 100).toFixed(0)}`, icon: <DollarSign size={18} className="text-forge-green" />, sub: 'all time' },
                { label: 'Platform Cut', value: `€${((analytics.revenue?.platform_revenue || 0) / 100).toFixed(0)}`, icon: <DollarSign size={18} className="text-forge-gold" />, sub: '20% fee' },
              ].map(stat => (
                <div key={stat.label} className="stat-card">
                  <div className="flex items-center gap-2 mb-3">{stat.icon}</div>
                  <div className="font-display text-3xl text-white">{stat.value}</div>
                  <div className="text-forge-muted text-xs mt-1">{stat.sub}</div>
                  <div className="text-forge-text text-sm font-medium mt-0.5">{stat.label}</div>
                </div>
              ))}
            </div>

            <div className="card overflow-hidden">
              <div className="p-4 border-b border-forge-border">
                <h3 className="font-semibold text-forge-text">Recent Orders</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-forge-border">
                      {['Beat', 'License', 'Buyer', 'Creator', 'Amount', 'Platform', 'Date'].map(h => (
                        <th key={h} className="text-left p-3 text-forge-muted text-xs font-medium uppercase tracking-wider">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(analytics.recentOrders || []).map((order: any) => (
                      <tr key={order.id} className="border-b border-forge-border/50 hover:bg-forge-dark/50">
                        <td className="p-3 text-sm text-forge-text">{order.beat_title}</td>
                        <td className="p-3"><span className="badge badge-accent text-[10px]">{order.license_name}</span></td>
                        <td className="p-3 text-sm text-forge-muted">@{order.buyer_username}</td>
                        <td className="p-3 text-sm text-forge-muted">@{order.creator_username}</td>
                        <td className="p-3 text-sm text-forge-green font-semibold">€{(order.amount_cents / 100).toFixed(2)}</td>
                        <td className="p-3 text-sm text-forge-gold">€{(order.platform_fee_cents / 100).toFixed(2)}</td>
                        <td className="p-3 text-xs text-forge-muted">{new Date(order.created_at).toLocaleDateString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* Orders */}
        {tab === 'orders' && (
          <div className="card overflow-hidden">
            <div className="p-4 border-b border-forge-border flex items-center justify-between">
              <h3 className="font-semibold text-forge-text">All Orders ({orders.length})</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-forge-border">
                    {['Beat', 'License', 'Buyer', 'Creator', 'Amount', 'Creator Earnings', 'Status', 'Date'].map(h => (
                      <th key={h} className="text-left p-3 text-forge-muted text-xs font-medium uppercase tracking-wider whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {orders.map((order: any) => (
                    <tr key={order.id} className="border-b border-forge-border/50 hover:bg-forge-dark/50">
                      <td className="p-3 text-sm text-forge-text font-medium max-w-[150px] truncate">{order.beat_title}</td>
                      <td className="p-3"><span className="badge badge-accent text-[10px]">{order.license_name}</span></td>
                      <td className="p-3 text-sm text-forge-muted">
                        <div>@{order.buyer_username}</div>
                        <div className="text-[10px] text-forge-border">{order.buyer_email}</div>
                      </td>
                      <td className="p-3 text-sm text-forge-muted">@{order.creator_username}</td>
                      <td className="p-3 text-sm text-forge-green font-semibold">€{(order.amount_cents / 100).toFixed(2)}</td>
                      <td className="p-3 text-sm text-forge-cyan">€{(order.creator_earnings_cents / 100).toFixed(2)}</td>
                      <td className="p-3">
                        <span className={clsx('badge text-[10px]', order.status === 'completed' ? 'bg-forge-green/20 text-forge-green border-forge-green/40' : 'bg-forge-dark border-forge-border text-forge-muted')}>
                          {order.status}
                        </span>
                      </td>
                      <td className="p-3 text-xs text-forge-muted whitespace-nowrap">{new Date(order.created_at).toLocaleDateString()}</td>
                    </tr>
                  ))}
                  {orders.length === 0 && (
                    <tr><td colSpan={8} className="text-center py-12 text-forge-muted">No orders yet</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Beats */}
        {tab === 'beats' && (
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-forge-border">
                    {['Title', 'Creator', 'Genre', 'Status', 'Sales', 'Plays', 'Actions'].map(h => (
                      <th key={h} className="text-left p-4 text-forge-muted text-xs font-medium uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {beats.map(beat => (
                    <tr key={beat.id} className="border-b border-forge-border/50 hover:bg-forge-dark/50">
                      <td className="p-4 text-sm text-forge-text font-medium max-w-[180px] truncate">{beat.title}</td>
                      <td className="p-4 text-sm text-forge-muted">@{beat.creator_username}</td>
                      <td className="p-4"><span className="badge badge-accent text-[10px]">{beat.genre}</span></td>
                      <td className="p-4">
                        <span className={clsx('badge text-[10px]', beat.status === 'published' ? 'badge-accent' : 'bg-forge-dark border border-forge-border text-forge-muted')}>
                          {beat.status}
                        </span>
                      </td>
                      <td className="p-4 text-sm text-forge-text">{beat.sales_count || 0}</td>
                      <td className="p-4 text-sm text-forge-text">{(beat.play_count || 0).toLocaleString()}</td>
                      <td className="p-4">
                        <div className="flex items-center gap-1">
                          <Link href={`/beat/${beat.id}`} target="_blank"
                            className="p-1.5 text-forge-muted hover:text-forge-cyan transition-colors rounded" title="View beat page">
                            <ExternalLink size={14} />
                          </Link>
                          <button
                            onClick={() => action(beat.id, () => api.delete(`/admin/beats/${beat.id}`), 'Beat removed')}
                            disabled={actionLoading === beat.id}
                            className="p-1.5 text-forge-muted hover:text-forge-accent transition-colors rounded"
                            title="Remove beat"
                          >
                            {actionLoading === beat.id ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />}
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Users */}
        {tab === 'users' && (
          <div className="card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-forge-border">
                    {['User', 'Email', 'Plan', 'Role', 'Balance', 'Joined', 'Actions'].map(h => (
                      <th key={h} className="text-left p-4 text-forge-muted text-xs font-medium uppercase tracking-wider whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {users.map((u: any) => (
                    <tr key={u.id} className="border-b border-forge-border/50 hover:bg-forge-dark/50">
                      <td className="p-4 text-sm text-forge-text font-medium">@{u.username}</td>
                      <td className="p-4 text-sm text-forge-muted">{u.email}</td>
                      <td className="p-4">
                        <span className={clsx('badge text-[10px]', u.subscription_plan === 'pro' ? 'bg-forge-gold/20 text-forge-gold border-forge-gold/40' : 'bg-forge-dark border-forge-border text-forge-muted')}>
                          {u.subscription_plan}
                        </span>
                      </td>
                      <td className="p-4">
                        <span className={clsx('badge text-[10px]',
                          u.role === 'admin' ? 'badge-accent' :
                          u.role === 'moderator' ? 'bg-forge-cyan/20 text-forge-cyan border-forge-cyan/40' :
                          'bg-forge-dark border-forge-border text-forge-muted'
                        )}>
                          {u.role}
                        </span>
                      </td>
                      <td className="p-4 text-sm text-forge-green">€{(u.balance_cents / 100).toFixed(2)}</td>
                      <td className="p-4 text-xs text-forge-muted">{new Date(u.created_at).toLocaleDateString()}</td>
                      <td className="p-4">
                        <div className="flex items-center gap-1">
                          {/* Give/remove Pro */}
                          {u.subscription_plan !== 'pro' ? (
                            <button
                              onClick={() => action(u.id + '_pro', () => api.put(`/admin/users/${u.id}/subscription`, { plan: 'pro' }), `Pro given to @${u.username}`)}
                              disabled={actionLoading === u.id + '_pro'}
                              className="p-1.5 text-forge-muted hover:text-forge-gold transition-colors rounded"
                              title="Give Pro"
                            >
                              {actionLoading === u.id + '_pro' ? <Loader2 size={13} className="animate-spin" /> : <Crown size={13} />}
                            </button>
                          ) : (
                            <button
                              onClick={() => action(u.id + '_free', () => api.put(`/admin/users/${u.id}/subscription`, { plan: 'free' }), `Pro removed from @${u.username}`)}
                              disabled={actionLoading === u.id + '_free'}
                              className="p-1.5 text-forge-gold hover:text-forge-muted transition-colors rounded"
                              title="Remove Pro"
                            >
                              {actionLoading === u.id + '_free' ? <Loader2 size={13} className="animate-spin" /> : <Crown size={13} />}
                            </button>
                          )}

                          {/* Role buttons */}
                          {u.role === 'creator' && (
                            <>
                              <button
                                onClick={() => action(u.id + '_mod', () => api.put(`/admin/users/${u.id}/role`, { role: 'moderator' }), `@${u.username} is now moderator`)}
                                disabled={actionLoading === u.id + '_mod'}
                                className="p-1.5 text-forge-muted hover:text-forge-cyan transition-colors rounded"
                                title="Make moderator"
                              >
                                {actionLoading === u.id + '_mod' ? <Loader2 size={13} className="animate-spin" /> : <MessageSquare size={13} />}
                              </button>
                              <button
                                onClick={() => action(u.id + '_admin', () => api.put(`/admin/users/${u.id}/role`, { role: 'admin' }), `@${u.username} is now admin`)}
                                disabled={actionLoading === u.id + '_admin'}
                                className="p-1.5 text-forge-muted hover:text-forge-accent transition-colors rounded"
                                title="Make admin"
                              >
                                {actionLoading === u.id + '_admin' ? <Loader2 size={13} className="animate-spin" /> : <Shield size={13} />}
                              </button>
                            </>
                          )}
                          {u.role === 'moderator' && (
                            <button
                              onClick={() => action(u.id + '_creator', () => api.put(`/admin/users/${u.id}/role`, { role: 'creator' }), `@${u.username} demoted to creator`)}
                              disabled={actionLoading === u.id + '_creator'}
                              className="p-1.5 text-forge-cyan hover:text-forge-muted transition-colors rounded"
                              title="Remove moderator"
                            >
                              {actionLoading === u.id + '_creator' ? <Loader2 size={13} className="animate-spin" /> : <MessageSquare size={13} />}
                            </button>
                          )}
                          {u.role === 'admin' && (
                            <button
                              onClick={() => action(u.id + '_creator', () => api.put(`/admin/users/${u.id}/role`, { role: 'creator' }), `@${u.username} demoted to creator`)}
                              disabled={actionLoading === u.id + '_creator'}
                              className="p-1.5 text-forge-accent hover:text-forge-muted transition-colors rounded"
                              title="Remove admin"
                            >
                              {actionLoading === u.id + '_creator' ? <Loader2 size={13} className="animate-spin" /> : <Shield size={13} />}
                            </button>
                          )}

                          {/* Give credits */}
                          <button
                            onClick={() => { setCreditsModal({ userId: u.id, username: u.username }); setCreditsAmount('10') }}
                            className="p-1.5 text-forge-muted hover:text-forge-cyan transition-colors rounded"
                            title={`Give credits (has ${u.extra_beat_credits ?? 0})`}
                          >
                            <Zap size={13} />
                          </button>

                          {/* Ban / unban */}
                          {u.is_active ? (
                            <button
                              onClick={() => action(u.id + '_ban', () => api.put(`/admin/users/${u.id}/deactivate`), `@${u.username} banned`)}
                              disabled={actionLoading === u.id + '_ban'}
                              className="p-1.5 text-forge-muted hover:text-forge-accent transition-colors rounded"
                              title="Ban user"
                            >
                              {actionLoading === u.id + '_ban' ? <Loader2 size={13} className="animate-spin" /> : <UserX size={13} />}
                            </button>
                          ) : (
                            <button
                              onClick={() => action(u.id + '_unban', () => api.put(`/admin/users/${u.id}/activate`), `@${u.username} unbanned`)}
                              disabled={actionLoading === u.id + '_unban'}
                              className="p-1.5 text-forge-muted hover:text-forge-green transition-colors rounded"
                              title="Unban user"
                            >
                              {actionLoading === u.id + '_unban' ? <Loader2 size={13} className="animate-spin" /> : <UserCheck size={13} />}
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Support Tickets */}
        {tab === 'tickets' && (
          <div className="card overflow-hidden">
            <div className="p-4 border-b border-forge-border">
              <h3 className="font-semibold text-forge-text">Support Tickets ({tickets.length})</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-forge-border">
                    {['#', 'User', 'Category', 'Message', 'Status', 'Replies', 'Date', 'Actions'].map(h => (
                      <th key={h} className="text-left p-3 text-forge-muted text-xs font-medium uppercase tracking-wider whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tickets.map((t: any) => (
                    <tr key={t.id} className="border-b border-forge-border/50 hover:bg-forge-dark/50">
                      <td className="p-3 text-xs text-forge-muted font-mono">#{t.ticket_number}</td>
                      <td className="p-3 text-sm text-forge-text">@{t.username}</td>
                      <td className="p-3"><span className="badge badge-accent text-[10px]">{t.category}</span></td>
                      <td className="p-3 text-sm text-forge-muted max-w-[200px] truncate">{t.message}</td>
                      <td className="p-3">
                        <span className={clsx('badge text-[10px]',
                          t.status === 'resolved' ? 'bg-forge-green/20 text-forge-green border-forge-green/40' :
                          t.status === 'in_progress' ? 'bg-forge-gold/20 text-forge-gold border-forge-gold/40' :
                          t.status === 'closed' ? 'bg-forge-dark text-forge-muted border-forge-border' :
                          'bg-forge-accent/20 text-forge-accent border-forge-accent/40'
                        )}>{t.status}</span>
                      </td>
                      <td className="p-3 text-sm text-forge-muted">{t.reply_count}</td>
                      <td className="p-3 text-xs text-forge-muted whitespace-nowrap">{new Date(t.created_at).toLocaleDateString()}</td>
                      <td className="p-3">
                        <Link href={`/mod?ticket=${t.id}`} className="p-1.5 text-forge-muted hover:text-forge-cyan transition-colors rounded inline-flex" title="View & reply">
                          <MessageSquare size={14} />
                        </Link>
                      </td>
                    </tr>
                  ))}
                  {tickets.length === 0 && (
                    <tr><td colSpan={8} className="text-center py-12 text-forge-muted">No support tickets</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Settings */}
        {tab === 'settings' && (
          <div className="max-w-md space-y-4">
            <div className="card p-6 space-y-4">
              <h3 className="font-semibold text-forge-text">Platform Settings</h3>

              <div>
                <label className="label-forge">Platform Fee (%)</label>
                <input type="number" value={settings.platform_fee_percent || ''}
                  onChange={e => setSettings((s: any) => ({ ...s, platform_fee_percent: e.target.value }))}
                  className="input-forge" min="0" max="50" />
                <p className="text-forge-muted text-xs mt-1">Creators keep {100 - parseInt(settings.platform_fee_percent || '20')}%</p>
              </div>

              <div>
                <label className="label-forge">Minimum Withdrawal (cents)</label>
                <input type="number" value={settings.min_withdrawal_cents || ''}
                  onChange={e => setSettings((s: any) => ({ ...s, min_withdrawal_cents: e.target.value }))}
                  className="input-forge" />
                <p className="text-forge-muted text-xs mt-1">= €{((parseInt(settings.min_withdrawal_cents || '2000')) / 100).toFixed(2)}</p>
              </div>

              <div>
                <label className="label-forge">Max Beats Per Day (per user)</label>
                <input type="number" value={settings.max_beats_per_day || ''}
                  onChange={e => setSettings((s: any) => ({ ...s, max_beats_per_day: e.target.value }))}
                  className="input-forge" min="1" max="50" />
              </div>

              <div>
                <label className="label-forge">Stripe Pro Price ID</label>
                <input type="text" value={settings.stripe_pro_price_id || ''}
                  onChange={e => setSettings((s: any) => ({ ...s, stripe_pro_price_id: e.target.value }))}
                  className="input-forge font-mono text-sm" placeholder="price_..." />
              </div>

              <button onClick={handleSaveSettings} disabled={savingSettings}
                className="btn-primary w-full flex items-center justify-center gap-2">
                {savingSettings ? <Loader2 size={16} className="animate-spin" /> : null}
                Save Settings
              </button>
            </div>
          </div>
        )}
      </div>

      {creditsModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setCreditsModal(null)}>
          <div className="bg-forge-card border border-forge-border rounded-2xl p-6 w-full max-w-sm shadow-xl mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display text-lg text-white">Give Beat Credits</h3>
              <button onClick={() => setCreditsModal(null)} className="text-forge-muted hover:text-forge-text"><X size={18} /></button>
            </div>
            <p className="text-forge-muted text-sm mb-4">Give credits to <span className="text-forge-text font-medium">@{creditsModal.username}</span></p>
            <input
              type="number"
              value={creditsAmount}
              onChange={e => setCreditsAmount(e.target.value)}
              className="input-forge w-full mb-4"
              min="1" max="10000" placeholder="Amount"
            />
            <div className="flex gap-2">
              <button onClick={() => setCreditsModal(null)} className="flex-1 btn-secondary py-2">Cancel</button>
              <button onClick={handleGiveCredits} className="flex-1 btn-primary py-2 flex items-center justify-center gap-2">
                <Zap size={14} /> Give Credits
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
