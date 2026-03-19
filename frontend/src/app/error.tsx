'use client'
import { useEffect } from 'react'
import Link from 'next/link'
import { AlertTriangle, RefreshCw } from 'lucide-react'

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error('Page error:', error)
  }, [error])

  return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center px-4">
      <div className="text-center max-w-md">
        <div className="w-16 h-16 rounded-2xl bg-forge-accent/20 border border-forge-accent/40 flex items-center justify-center mx-auto mb-6">
          <AlertTriangle size={28} className="text-forge-accent" />
        </div>
        <h2 className="font-display text-3xl text-white mb-3 tracking-wider">SOMETHING WENT WRONG</h2>
        <p className="text-forge-muted mb-8 text-sm">
          An unexpected error occurred. Try again or go back home.
        </p>
        <div className="flex gap-3 justify-center">
          <button
            onClick={reset}
            className="flex items-center gap-2 bg-forge-accent hover:bg-forge-accent-dim text-white font-semibold px-6 py-3 rounded-lg transition-all"
          >
            <RefreshCw size={16} />
            Try Again
          </button>
          <Link href="/" className="flex items-center gap-2 bg-forge-card border border-forge-border hover:border-forge-accent text-forge-text font-semibold px-6 py-3 rounded-lg transition-all">
            Go Home
          </Link>
        </div>
      </div>
    </div>
  )
}
