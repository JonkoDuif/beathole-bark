'use client'
import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { presetsApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Sliders, Check, Loader2, ArrowRight } from 'lucide-react'

export default function PresetSharePage() {
  const { token } = useParams<{ token: string }>()
  const { user, initialize } = useAuthStore()
  const router = useRouter()
  const [preset, setPreset] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [notFound, setNotFound] = useState(false)

  useEffect(() => { initialize() }, [])

  useEffect(() => {
    if (!token) return
    presetsApi.getShared(token)
      .then(res => setPreset(res.data))
      .catch(() => setNotFound(true))
      .finally(() => setLoading(false))
  }, [token])

  const handleSave = async () => {
    if (!user) {
      router.push(`/login?redirect=/presets/share/${token}`)
      return
    }
    setSaving(true)
    try {
      await presetsApi.saveShared(token)
      setSaved(true)
      toast.success('Preset saved to your library!')
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to save preset')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />

      <div className="flex items-center justify-center min-h-[calc(100vh-64px)] px-4 py-16">
        {loading ? (
          <div className="flex flex-col items-center gap-4 text-forge-muted">
            <Loader2 className="w-10 h-10 animate-spin text-forge-orange" />
            <p className="text-sm">Loading preset…</p>
          </div>
        ) : notFound ? (
          <div className="forge-card max-w-md w-full text-center p-10 space-y-4">
            <Sliders className="w-12 h-12 mx-auto text-forge-muted" />
            <h1 className="text-2xl font-bold text-white">Preset Not Found</h1>
            <p className="text-forge-muted text-sm">
              This share link is invalid or has expired.
            </p>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-2 text-forge-orange hover:underline text-sm"
            >
              Go to Dashboard <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
        ) : (
          <div className="forge-card max-w-md w-full text-center p-10 space-y-6">
            {/* Icon */}
            <div className="flex items-center justify-center w-20 h-20 mx-auto rounded-2xl bg-forge-orange/10 border border-forge-orange/20">
              <Sliders className="w-10 h-10 text-forge-orange" />
            </div>

            {/* Preset info */}
            <div className="space-y-2">
              <h1 className="text-3xl font-bold text-white tracking-tight">
                {preset?.name ?? 'Untitled Preset'}
              </h1>
              {preset?.creator_username && (
                <p className="text-forge-muted text-sm">
                  By{' '}
                  <span className="text-white font-medium">
                    @{preset.creator_username}
                  </span>
                </p>
              )}
            </div>

            <p className="text-forge-muted text-sm leading-relaxed">
              Save this preset to your profile to use it in the studio.
            </p>

            {/* Action */}
            {user ? (
              saved ? (
                <div className="flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-green-500/10 border border-green-500/30 text-green-400 font-medium">
                  <Check className="w-5 h-5" />
                  Saved to your library
                </div>
              ) : (
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-forge-orange hover:bg-forge-orange/90 disabled:opacity-60 disabled:cursor-not-allowed text-white font-semibold transition-colors"
                >
                  {saving ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Saving…
                    </>
                  ) : (
                    <>
                      <Sliders className="w-5 h-5" />
                      Save to My Presets
                    </>
                  )}
                </button>
              )
            ) : (
              <div className="space-y-3">
                <p className="text-forge-muted text-sm">
                  You need to be logged in to save this preset.
                </p>
                <Link
                  href={`/login?redirect=/presets/share/${token}`}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg bg-forge-orange hover:bg-forge-orange/90 text-white font-semibold transition-colors"
                >
                  Login to save this preset
                  <ArrowRight className="w-5 h-5" />
                </Link>
              </div>
            )}

            {saved && (
              <button
                onClick={() => router.push('/dashboard')}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-lg border border-forge-border hover:bg-white/5 text-white font-medium transition-colors"
              >
                Go to Dashboard
                <ArrowRight className="w-5 h-5" />
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
