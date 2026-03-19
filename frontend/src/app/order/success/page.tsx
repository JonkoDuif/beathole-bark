'use client'
import { useEffect, useState, Suspense } from 'react'
import { useSearchParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { ordersApi } from '@/lib/api'
import { CheckCircle, Download, Music2, Loader2, ArrowRight } from 'lucide-react'

function OrderSuccessContent() {
  const searchParams = useSearchParams()
  const router = useRouter()
  const [order, setOrder] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const sessionId = searchParams.get('session_id')
    if (!sessionId) { router.push('/marketplace'); return }

    ordersApi.success(sessionId)
      .then(res => setOrder(res.data))
      .catch(() => router.push('/dashboard'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="min-h-screen bg-forge-black flex items-center justify-center">
        <Loader2 size={32} className="animate-spin text-forge-accent" />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-lg mx-auto px-4 pt-32 pb-20 text-center">
        <div className="w-20 h-20 rounded-full bg-forge-green/20 border-2 border-forge-green flex items-center justify-center mx-auto mb-6">
          <CheckCircle size={36} className="text-forge-green" />
        </div>

        <h1 className="font-display text-4xl text-white mb-2">PURCHASE COMPLETE!</h1>
        <p className="text-forge-muted mb-8">Your license has been confirmed.</p>

        {order && (
          <div className="card p-6 text-left mb-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-lg bg-gradient-forge flex items-center justify-center">
                <Music2 size={16} className="text-white" />
              </div>
              <div>
                <p className="font-semibold text-forge-text">{order.beat_title}</p>
                <p className="text-forge-muted text-sm">{order.license_name}</p>
              </div>
              <span className="ml-auto text-forge-green font-bold">
                €{(order.amount_cents / 100).toFixed(2)}
              </span>
            </div>

            {(order.mp3_url || order.wav_url) && (
              <div className="space-y-2">
                {order.mp3_url && (
                  <a href={order.mp3_url} download
                    className="flex items-center gap-2 btn-secondary w-full justify-center text-sm py-2">
                    <Download size={14} />
                    Download MP3
                  </a>
                )}
                {order.wav_url && ['standard', 'premium', 'exclusive'].includes(order.license_type) && (
                  <a href={order.wav_url} download
                    className="flex items-center gap-2 btn-secondary w-full justify-center text-sm py-2">
                    <Download size={14} />
                    Download WAV
                  </a>
                )}
              </div>
            )}
          </div>
        )}

        <div className="flex gap-3">
          <Link href="/library" className="flex-1 btn-secondary flex items-center justify-center gap-2 py-3">
            My Library
          </Link>
          <Link href="/marketplace" className="flex-1 btn-primary flex items-center justify-center gap-2 py-3">
            Browse More <ArrowRight size={14} />
          </Link>
        </div>
      </div>
    </div>
  )
}

export default function OrderSuccessPage() {
  return (
    <Suspense fallback={
      <div className="min-h-screen bg-forge-black flex items-center justify-center">
        <Loader2 size={32} className="animate-spin text-forge-accent" />
      </div>
    }>
      <OrderSuccessContent />
    </Suspense>
  )
}
