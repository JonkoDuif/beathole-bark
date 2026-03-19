'use client'
import { useEffect, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { supportApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Shield, MessageSquare, Send, Loader2, ChevronRight, ArrowLeft, User } from 'lucide-react'
import clsx from 'clsx'

const STATUS_OPTIONS = ['open', 'in_progress', 'resolved', 'closed']
const STATUS_STYLES: Record<string, string> = {
  open: 'bg-forge-accent/20 text-forge-accent border-forge-accent/40',
  in_progress: 'bg-forge-gold/20 text-forge-gold border-forge-gold/40',
  resolved: 'bg-forge-green/20 text-forge-green border-forge-green/40',
  closed: 'bg-forge-dark text-forge-muted border-forge-border',
}

export default function ModPage() {
  const { user, isLoading, initialize } = useAuthStore()
  const router = useRouter()
  const searchParams = useSearchParams()
  const initialTicket = searchParams.get('ticket')

  const [tickets, setTickets] = useState<any[]>([])
  const [selectedTicket, setSelectedTicket] = useState<any>(null)
  const [replies, setReplies] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [replyText, setReplyText] = useState('')
  const [sending, setSending] = useState(false)
  const [filterStatus, setFilterStatus] = useState('')

  useEffect(() => { initialize() }, [])
  useEffect(() => {
    if (isLoading) return
    if (!user || !['admin', 'moderator'].includes(user.role)) {
      router.push('/'); return
    }
    loadTickets()
  }, [user, isLoading])

  useEffect(() => {
    if (initialTicket && tickets.length > 0) {
      const t = tickets.find(t => t.id === initialTicket)
      if (t) openTicket(t)
    }
  }, [initialTicket, tickets])

  const loadTickets = async () => {
    setLoading(true)
    try {
      const res = await supportApi.adminTickets(filterStatus ? { status: filterStatus } : {})
      setTickets(res.data)
    } finally { setLoading(false) }
  }

  useEffect(() => { if (user) loadTickets() }, [filterStatus])

  const openTicket = async (t: any) => {
    setSelectedTicket(t)
    try {
      const res = await supportApi.getTicket(t.id)
      setSelectedTicket(res.data.ticket); setReplies(res.data.replies)
    } catch {}
  }

  const handleReply = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!replyText.trim() || !selectedTicket) return
    setSending(true)
    try {
      const res = await supportApi.reply(selectedTicket.id, replyText)
      setReplies(prev => [...prev, res.data.reply])
      setReplyText('')
      if (selectedTicket.status === 'open') {
        setSelectedTicket((t: any) => ({ ...t, status: 'in_progress' }))
        setTickets(prev => prev.map(t => t.id === selectedTicket.id ? { ...t, status: 'in_progress' } : t))
      }
      toast.success('Reply sent')
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to send')
    } finally { setSending(false) }
  }

  const handleStatusChange = async (status: string) => {
    if (!selectedTicket) return
    try {
      await supportApi.updateStatus(selectedTicket.id, status)
      setSelectedTicket((t: any) => ({ ...t, status }))
      setTickets(prev => prev.map(t => t.id === selectedTicket.id ? { ...t, status } : t))
      toast.success('Status updated')
    } catch { toast.error('Failed to update status') }
  }

  if (isLoading || !user) return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-forge-accent" />
    </div>
  )

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 pt-24 pb-20">

        <div className="flex items-center gap-3 mb-6">
          <div className="w-10 h-10 rounded-xl bg-forge-cyan/20 border border-forge-cyan/30 flex items-center justify-center">
            <Shield size={20} className="text-forge-cyan" />
          </div>
          <div>
            <h1 className="font-display text-3xl text-white tracking-wider">MODERATOR PANEL</h1>
            <p className="text-forge-muted text-sm">Support ticket management</p>
          </div>
        </div>

        <div className="flex gap-6 h-[calc(100vh-220px)]">

          {/* Ticket list */}
          <div className={clsx('flex flex-col', selectedTicket ? 'hidden md:flex w-80 flex-shrink-0' : 'flex-1 max-w-2xl')}>
            <div className="flex items-center gap-2 mb-3">
              <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)}
                className="input-forge text-sm flex-1">
                <option value="">All statuses</option>
                {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
              </select>
              <button onClick={loadTickets} className="btn-ghost text-sm px-3">Refresh</button>
            </div>

            {loading ? (
              <div className="flex-1 flex items-center justify-center">
                <Loader2 size={24} className="animate-spin text-forge-muted" />
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto space-y-2 pr-1">
                {tickets.length === 0 ? (
                  <div className="text-center py-12 text-forge-muted text-sm">No tickets found</div>
                ) : tickets.map((t: any) => (
                  <button key={t.id} onClick={() => openTicket(t)}
                    className={clsx(
                      'w-full text-left card p-3 hover:border-forge-border/80 transition-colors',
                      selectedTicket?.id === t.id ? 'border-forge-cyan/40 bg-forge-cyan/5' : ''
                    )}>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-mono text-xs text-forge-muted">#{t.ticket_number}</span>
                      <span className={clsx('badge text-[10px]', STATUS_STYLES[t.status])}>{t.status.replace('_', ' ')}</span>
                      <span className="badge badge-accent text-[10px] ml-auto">{t.category}</span>
                    </div>
                    <p className="text-sm text-forge-text truncate">@{t.username}: {t.message}</p>
                    <p className="text-xs text-forge-muted mt-1">{t.reply_count} replies · {new Date(t.created_at).toLocaleDateString()}</p>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Ticket detail */}
          {selectedTicket ? (
            <div className="flex-1 flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <button onClick={() => setSelectedTicket(null)} className="flex items-center gap-1.5 text-sm text-forge-muted hover:text-forge-text md:hidden">
                  <ArrowLeft size={14} /> Back
                </button>
                <div className="flex items-center gap-2 ml-auto">
                  <span className="text-xs text-forge-muted">Status:</span>
                  <select value={selectedTicket.status} onChange={e => handleStatusChange(e.target.value)}
                    className="input-forge text-sm py-1 w-36">
                    {STATUS_OPTIONS.map(s => <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
                  </select>
                </div>
              </div>

              <div className="card p-4 mb-3">
                <div className="flex items-center gap-2 mb-2">
                  <span className="font-mono text-xs text-forge-muted">#{selectedTicket.ticket_number}</span>
                  <span className="badge badge-accent text-[10px]">{selectedTicket.category}</span>
                  <span className={clsx('badge text-[10px]', STATUS_STYLES[selectedTicket.status])}>{selectedTicket.status.replace('_', ' ')}</span>
                </div>
                <div className="flex items-center gap-1 text-xs text-forge-muted mb-2">
                  <User size={11} /> @{selectedTicket.username || '—'}
                </div>
                <p className="text-forge-text text-sm leading-relaxed">{selectedTicket.message}</p>
                <p className="text-xs text-forge-muted mt-2">{new Date(selectedTicket.created_at).toLocaleString()}</p>
              </div>

              {/* Replies */}
              <div className="flex-1 overflow-y-auto space-y-3 mb-3">
                {replies.map((r: any) => (
                  <div key={r.id} className={clsx(
                    'p-3 rounded-xl border text-sm',
                    r.is_staff ? 'bg-forge-accent/5 border-forge-accent/20 ml-8' : 'bg-forge-dark border-forge-border mr-8'
                  )}>
                    <div className="flex items-center gap-2 mb-1.5">
                      {r.is_staff ? (
                        <span className="flex items-center gap-1 text-forge-accent text-xs font-semibold">
                          <Shield size={10} /> Staff · {r.display_name || r.username}
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-forge-muted text-xs">
                          <User size={10} /> @{r.username}
                        </span>
                      )}
                      <span className="text-forge-border text-xs ml-auto">{new Date(r.created_at).toLocaleString()}</span>
                    </div>
                    <p className="text-forge-text leading-relaxed">{r.message}</p>
                  </div>
                ))}
              </div>

              {/* Reply form */}
              {selectedTicket.status !== 'closed' && (
                <form onSubmit={handleReply} className="card p-3 flex gap-2">
                  <textarea
                    value={replyText}
                    onChange={e => setReplyText(e.target.value)}
                    placeholder="Type your reply..."
                    className="input-forge flex-1 min-h-[60px] resize-none text-sm"
                  />
                  <button type="submit" disabled={sending || !replyText.trim()}
                    className="btn-primary flex items-center gap-1.5 self-end text-sm px-4">
                    {sending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                    Reply
                  </button>
                </form>
              )}
            </div>
          ) : (
            <div className="hidden md:flex flex-1 items-center justify-center">
              <div className="text-center">
                <MessageSquare size={40} className="text-forge-muted mx-auto mb-3" />
                <p className="text-forge-muted">Select a ticket to view and respond</p>
              </div>
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
