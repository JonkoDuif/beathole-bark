'use client'
import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { supportApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { ArrowLeft, Send, Loader2, Shield, User } from 'lucide-react'
import clsx from 'clsx'

const STATUS_STYLES: Record<string, string> = {
  open: 'bg-forge-accent/20 text-forge-accent border-forge-accent/40',
  in_progress: 'bg-forge-gold/20 text-forge-gold border-forge-gold/40',
  resolved: 'bg-forge-green/20 text-forge-green border-forge-green/40',
  closed: 'bg-forge-dark text-forge-muted border-forge-border',
}

export default function TicketDetailPage() {
  const { id } = useParams<{ id: string }>()
  const { user, initialize } = useAuthStore()
  const router = useRouter()
  const [ticket, setTicket] = useState<any>(null)
  const [replies, setReplies] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [reply, setReply] = useState('')
  const [sending, setSending] = useState(false)

  useEffect(() => { initialize() }, [])
  useEffect(() => {
    if (user) loadTicket()
  }, [user])

  const loadTicket = async () => {
    setLoading(true)
    try {
      const res = await supportApi.getTicket(id)
      setTicket(res.data.ticket); setReplies(res.data.replies)
    } catch { router.push('/support') }
    finally { setLoading(false) }
  }

  const handleReply = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!reply.trim()) return
    setSending(true)
    try {
      const res = await supportApi.reply(id, reply)
      setReplies(prev => [...prev, res.data.reply])
      setReply('')
      if (ticket.status === 'open') setTicket((t: any) => ({ ...t, status: 'in_progress' }))
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to send reply')
    } finally { setSending(false) }
  }

  if (loading) return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-forge-accent" />
    </div>
  )

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-2xl mx-auto px-4 py-24 pt-32">
        <Link href="/support" className="flex items-center gap-2 text-forge-muted hover:text-forge-text text-sm mb-6 transition-colors">
          <ArrowLeft size={14} /> Back to tickets
        </Link>

        {ticket && (
          <>
            {/* Ticket header */}
            <div className="card p-5 mb-4">
              <div className="flex items-center gap-2 mb-2">
                <span className="font-mono text-xs text-forge-muted">#{ticket.ticket_number}</span>
                <span className="badge badge-accent text-[10px]">{ticket.category}</span>
                <span className={clsx('badge text-[10px]', STATUS_STYLES[ticket.status])}>{ticket.status.replace('_', ' ')}</span>
              </div>
              <p className="text-forge-text text-sm leading-relaxed">{ticket.message}</p>
              <p className="text-xs text-forge-muted mt-2">
                {new Date(ticket.created_at).toLocaleString('en-US', { dateStyle: 'medium', timeStyle: 'short' })}
              </p>
            </div>

            {/* Replies */}
            {replies.length > 0 && (
              <div className="space-y-3 mb-4">
                {replies.map((r: any) => (
                  <div key={r.id} className={clsx(
                    'p-4 rounded-xl border text-sm',
                    r.is_staff
                      ? 'bg-forge-accent/5 border-forge-accent/20 ml-4'
                      : 'bg-forge-dark border-forge-border mr-4'
                  )}>
                    <div className="flex items-center gap-2 mb-2">
                      {r.is_staff ? (
                        <span className="flex items-center gap-1 text-forge-accent text-xs font-semibold">
                          <Shield size={11} /> Staff · {r.display_name || r.username}
                        </span>
                      ) : (
                        <span className="flex items-center gap-1 text-forge-muted text-xs">
                          <User size={11} /> @{r.username}
                        </span>
                      )}
                      <span className="text-forge-border text-xs ml-auto">
                        {new Date(r.created_at).toLocaleString('en-US', { dateStyle: 'short', timeStyle: 'short' })}
                      </span>
                    </div>
                    <p className="text-forge-text leading-relaxed">{r.message}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Reply form */}
            {ticket.status !== 'closed' && (
              <form onSubmit={handleReply} className="card p-4">
                <textarea
                  value={reply}
                  onChange={e => setReply(e.target.value)}
                  placeholder="Add a reply..."
                  className="input-forge min-h-[80px] resize-none text-sm mb-3"
                />
                <button type="submit" disabled={sending || !reply.trim()}
                  className="btn-primary flex items-center gap-2 text-sm">
                  {sending ? <Loader2 size={14} className="animate-spin" /> : <Send size={14} />}
                  Send Reply
                </button>
              </form>
            )}
          </>
        )}
      </div>
    </div>
  )
}
