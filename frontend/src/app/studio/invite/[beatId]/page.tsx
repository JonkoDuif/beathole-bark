'use client'
import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { studioApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { UserPlus, Check, Loader2, X, ArrowRight } from 'lucide-react'
import Link from 'next/link'

export default function StudioInvitePage() {
  const { beatId } = useParams<{ beatId: string }>()
  const { user, initialize } = useAuthStore()
  const router = useRouter()

  const [invite, setInvite] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [accepting, setAccepting] = useState(false)
  const [declining, setDeclining] = useState(false)
  const [status, setStatus] = useState<'pending' | 'accepted' | 'declined' | 'none'>('pending')

  useEffect(() => { initialize() }, [])

  useEffect(() => {
    if (!beatId) return
    studioApi.pendingInvitations()
      .then(res => {
        const invitations: any[] = res.data?.invitations ?? res.data ?? []
        const found = invitations.find(
          (inv: any) => String(inv.beat_id ?? inv.beatId) === String(beatId)
        )
        if (found) {
          setInvite(found)
          setStatus('pending')
        } else {
          setStatus('none')
        }
      })
      .catch(() => setStatus('none'))
      .finally(() => setLoading(false))
  }, [beatId])

  const handleAccept = async () => {
    setAccepting(true)
    try {
      await studioApi.acceptInvite(beatId)
      setStatus('accepted')
      toast.success('Invitation accepted! Opening studio…')
      setTimeout(() => router.push(`/studio/${beatId}`), 1200)
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to accept invitation')
    } finally {
      setAccepting(false)
    }
  }

  const handleDecline = async () => {
    setDeclining(true)
    try {
      await studioApi.declineInvite(beatId)
      setStatus('declined')
      toast.success('Invitation declined.')
      setTimeout(() => router.push('/dashboard'), 1200)
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to decline invitation')
    } finally {
      setDeclining(false)
    }
  }

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />

      <div className="flex items-center justify-center min-h-[calc(100vh-64px)] px-4 py-16">
        {loading ? (
          <div className="flex flex-col items-center gap-4 text-forge-muted">
            <Loader2 className="w-10 h-10 animate-spin text-forge-orange" />
            <p className="text-sm">Loading invitation…</p>
          </div>
        ) : !user ? (
          <div className="forge-card max-w-md w-full text-center p-10 space-y-4">
            <UserPlus className="w-12 h-12 mx-auto text-forge-muted" />
            <h1 className="text-2xl font-bold text-white">Login Required</h1>
            <p className="text-forge-muted text-sm">
              You need to be logged in to view this invitation.
            </p>
            <Link
              href={`/login?redirect=/studio/invite/${beatId}`}
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-forge-orange hover:bg-forge-orange/90 text-white font-semibold transition-colors"
            >
              Login to continue
              <ArrowRight className="w-5 h-5" />
            </Link>
          </div>
        ) : status === 'accepted' ? (
          <div className="forge-card max-w-md w-full text-center p-10 space-y-4">
            <div className="flex items-center justify-center w-20 h-20 mx-auto rounded-2xl bg-green-500/10 border border-green-500/30">
              <Check className="w-10 h-10 text-green-400" />
            </div>
            <h1 className="text-2xl font-bold text-white">Invitation Accepted!</h1>
            <p className="text-forge-muted text-sm">Redirecting you to the studio…</p>
            <Loader2 className="w-5 h-5 animate-spin text-forge-orange mx-auto" />
          </div>
        ) : status === 'declined' ? (
          <div className="forge-card max-w-md w-full text-center p-10 space-y-4">
            <div className="flex items-center justify-center w-20 h-20 mx-auto rounded-2xl bg-white/5 border border-forge-border">
              <X className="w-10 h-10 text-forge-muted" />
            </div>
            <h1 className="text-2xl font-bold text-white">Invitation Declined</h1>
            <p className="text-forge-muted text-sm">Redirecting you to your dashboard…</p>
            <Loader2 className="w-5 h-5 animate-spin text-forge-orange mx-auto" />
          </div>
        ) : status === 'none' ? (
          <div className="forge-card max-w-md w-full text-center p-10 space-y-4">
            <UserPlus className="w-12 h-12 mx-auto text-forge-muted" />
            <h1 className="text-2xl font-bold text-white">No Pending Invitation</h1>
            <p className="text-forge-muted text-sm">
              You don't have a pending invitation for this project. You may already
              be a collaborator, or the invitation has expired.
            </p>
            <div className="flex flex-col sm:flex-row gap-3 justify-center pt-2">
              <Link
                href={`/studio/${beatId}`}
                className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg bg-forge-orange hover:bg-forge-orange/90 text-white font-semibold text-sm transition-colors"
              >
                Open Studio
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                href="/dashboard"
                className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg border border-forge-border hover:bg-white/5 text-white font-medium text-sm transition-colors"
              >
                Go to Dashboard
              </Link>
            </div>
          </div>
        ) : (
          /* Main invite card */
          <div className="forge-card max-w-md w-full text-center p-10 space-y-6">
            {/* Icon */}
            <div className="flex items-center justify-center w-20 h-20 mx-auto rounded-2xl bg-forge-orange/10 border border-forge-orange/20">
              <UserPlus className="w-10 h-10 text-forge-orange" />
            </div>

            {/* Invite info */}
            <div className="space-y-2">
              <p className="text-forge-muted text-xs uppercase tracking-widest font-medium">
                Studio Invitation
              </p>
              <h1 className="text-2xl font-bold text-white leading-snug">
                Invited to collaborate on{' '}
                <span className="text-forge-orange">
                  {invite?.beat_title ?? invite?.beatTitle ?? 'this beat'}
                </span>
              </h1>
              {(invite?.inviter_username ?? invite?.inviterUsername) && (
                <p className="text-forge-muted text-sm">
                  By{' '}
                  <span className="text-white font-medium">
                    @{invite?.inviter_username ?? invite?.inviterUsername}
                  </span>
                </p>
              )}
            </div>

            <p className="text-forge-muted text-sm leading-relaxed">
              You've been invited to collaborate in the studio. Accept to join and
              start working together in real time.
            </p>

            {/* Action buttons */}
            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={handleAccept}
                disabled={accepting || declining}
                className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-lg bg-forge-orange hover:bg-forge-orange/90 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold transition-colors"
              >
                {accepting ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Accepting…
                  </>
                ) : (
                  <>
                    <Check className="w-5 h-5" />
                    Accept &amp; Open Studio
                  </>
                )}
              </button>

              <button
                onClick={handleDecline}
                disabled={accepting || declining}
                className="flex-1 flex items-center justify-center gap-2 px-5 py-3 rounded-lg border border-red-500/40 hover:bg-red-500/10 disabled:opacity-60 disabled:cursor-not-allowed text-red-400 font-medium transition-colors"
              >
                {declining ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Declining…
                  </>
                ) : (
                  <>
                    <X className="w-5 h-5" />
                    Decline
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
