'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { supportApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { MessageSquare, Send, ChevronRight, Loader2, CheckCircle, Clock, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

const CATEGORIES = ['Licensing', 'Generation', 'Subscription', 'Features', 'Other', 'Payment']

const STATUS_STYLES: Record<string, string> = {
  open: 'bg-forge-accent/20 text-forge-accent border-forge-accent/40',
  in_progress: 'bg-forge-gold/20 text-forge-gold border-forge-gold/40',
  resolved: 'bg-forge-green/20 text-forge-green border-forge-green/40',
  closed: 'bg-forge-dark text-forge-muted border-forge-border',
}

export default function SupportPage() {
  const { user, isLoading, initialize } = useAuthStore()
  const router = useRouter()
  const [form, setForm] = useState({ category: '', message: '' })
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState<any>(null)
  const [tickets, setTickets] = useState<any[]>([])
  const [loadingTickets, setLoadingTickets] = useState(true)

  useEffect(() => { initialize() }, [])
  useEffect(() => {
    if (!isLoading && !user) router.push('/login?redirect=/support')
    if (user) loadTickets()
  }, [user, isLoading])

  const loadTickets = async () => {
    setLoadingTickets(true)
    try { const res = await supportApi.myTickets(); setTickets(res.data) }
    catch {} finally { setLoadingTickets(false) }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!form.category) return toast.error('Please select a category')
    if (!form.message.trim() || form.message.trim().length < 10) return toast.error('Please describe your issue (min. 10 characters)')
    setSubmitting(true)
    try {
      const res = await supportApi.createTicket(form)
      setSubmitted(res.data.ticket)
      setForm({ category: '', message: '' })
      loadTickets()
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to submit ticket')
    } finally { setSubmitting(false) }
  }

  if (isLoading) return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-forge-accent" />
    </div>
  )

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-3xl mx-auto px-4 py-24 pt-32">

        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-forge-accent/20 border border-forge-accent/30 flex items-center justify-center">
            <MessageSquare size={20} className="text-forge-accent" />
          </div>
          <div>
            <h1 className="font-display text-3xl text-white tracking-wider">SUPPORT</h1>
            <p className="text-forge-muted text-sm">Submit a request and we'll get back to you</p>
          </div>
        </div>

        {/* Submit ticket form */}
        <div className="card p-6 mb-8">
          <h2 className="font-semibold text-forge-text mb-5">Open a Support Ticket</h2>

          {submitted ? (
            <div className="text-center py-6">
              <CheckCircle size={40} className="text-forge-green mx-auto mb-3" />
              <h3 className="font-display text-xl text-white mb-1">Ticket Submitted!</h3>
              <p className="text-forge-muted text-sm mb-2">
                Your ticket <span className="font-mono text-forge-accent">#{submitted.ticket_number}</span> has been created.
              </p>
              <p className="text-forge-muted text-xs mb-4">We'll respond as soon as possible. You can track it below.</p>
              <button onClick={() => setSubmitted(null)} className="btn-ghost text-sm">Submit Another</button>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="label-forge">Category</label>
                <select
                  value={form.category}
                  onChange={e => setForm(f => ({ ...f, category: e.target.value }))}
                  className="input-forge"
                >
                  <option value="">Select a category...</option>
                  {CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label className="label-forge">Tell us your problem</label>
                <textarea
                  value={form.message}
                  onChange={e => setForm(f => ({ ...f, message: e.target.value }))}
                  placeholder="Describe your issue in detail..."
                  className="input-forge min-h-[140px] resize-none text-sm"
                  maxLength={2000}
                />
                <div className="text-right text-xs text-forge-muted mt-1">{form.message.length}/2000</div>
              </div>
              <button type="submit" disabled={submitting}
                className="btn-primary flex items-center gap-2">
                {submitting ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
                Send Ticket
              </button>
            </form>
          )}
        </div>

        {/* My tickets list */}
        <div>
          <h2 className="font-semibold text-forge-text mb-4 flex items-center gap-2">
            <Clock size={16} className="text-forge-muted" />
            My Tickets
          </h2>

          {loadingTickets ? (
            <div className="text-center py-8"><Loader2 size={20} className="animate-spin text-forge-muted mx-auto" /></div>
          ) : tickets.length === 0 ? (
            <div className="card p-8 text-center">
              <AlertCircle size={24} className="text-forge-muted mx-auto mb-2" />
              <p className="text-forge-muted text-sm">No tickets yet. Submit one above if you need help.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {tickets.map((t: any) => (
                <Link key={t.id} href={`/support/${t.id}`}
                  className="card p-4 flex items-center justify-between gap-4 hover:border-forge-border/80 transition-colors group">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-mono text-forge-muted">#{t.ticket_number}</span>
                      <span className="badge badge-accent text-[10px]">{t.category}</span>
                      <span className={clsx('badge text-[10px]', STATUS_STYLES[t.status])}>{t.status.replace('_', ' ')}</span>
                    </div>
                    <p className="text-sm text-forge-text truncate">{t.message}</p>
                    <p className="text-xs text-forge-muted mt-1">
                      {new Date(t.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })}
                      {t.reply_count > 0 && <span className="ml-2">· {t.reply_count} {t.reply_count === 1 ? 'reply' : 'replies'}</span>}
                    </p>
                  </div>
                  <ChevronRight size={16} className="text-forge-muted group-hover:text-forge-text transition-colors flex-shrink-0" />
                </Link>
              ))}
            </div>
          )}
        </div>

      </div>
    </div>
  )
}
