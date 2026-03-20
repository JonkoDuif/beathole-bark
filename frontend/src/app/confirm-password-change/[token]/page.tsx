'use client'
import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import Link from 'next/link'
import { authApi } from '@/lib/api'
import { Loader2, CheckCircle, XCircle } from 'lucide-react'

export default function ConfirmPasswordChangePage() {
  const { token } = useParams<{ token: string }>()
  const [status, setStatus] = useState<'loading' | 'success' | 'error'>('loading')
  const [message, setMessage] = useState('')

  useEffect(() => {
    if (!token) return
    authApi.confirmPasswordChange(token)
      .then(() => {
        setStatus('success')
        setMessage('Your password has been updated. Please log in with your new password.')
      })
      .catch((err: any) => {
        setStatus('error')
        setMessage(err.response?.data?.error || 'This link is invalid or has expired.')
      })
  }, [token])

  return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-grid-pattern bg-grid opacity-30" />
      <div className="relative w-full max-w-md">
        <div className="card p-8 text-center space-y-5">
          {status === 'loading' && (
            <>
              <Loader2 size={40} className="animate-spin text-forge-accent mx-auto" />
              <p className="text-forge-muted">Confirming password change...</p>
            </>
          )}
          {status === 'success' && (
            <>
              <CheckCircle size={40} className="text-green-400 mx-auto" />
              <h2 className="font-display text-2xl text-white">Password Updated</h2>
              <p className="text-forge-muted">{message}</p>
              <Link href="/login" className="btn-primary inline-block px-6 py-2">
                Go to Login
              </Link>
            </>
          )}
          {status === 'error' && (
            <>
              <XCircle size={40} className="text-forge-accent mx-auto" />
              <h2 className="font-display text-2xl text-white">Confirmation Failed</h2>
              <p className="text-forge-muted">{message}</p>
              <Link href="/login" className="btn-primary inline-block px-6 py-2">
                Go to Login
              </Link>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
