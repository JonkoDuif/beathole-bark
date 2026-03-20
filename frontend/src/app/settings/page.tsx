'use client'
import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import Navbar from '@/components/Navbar'
import { useAuthStore } from '@/store/auth'
import { accountApi, authApi } from '@/lib/api'
import toast from 'react-hot-toast'
import { Settings, User, Mail, Lock, Save, Loader2, Eye, EyeOff, Shield } from 'lucide-react'
import clsx from 'clsx'

export default function SettingsPage() {
  const { user, isLoading, initialize, refresh } = useAuthStore()
  const router = useRouter()

  const [emailForm, setEmailForm] = useState({ newEmail: '', currentPassword: '' })
  const [usernameForm, setUsernameForm] = useState({ username: '', displayName: '' })
  const [passwordForm, setPasswordForm] = useState({ currentPassword: '', newPassword: '', confirmPassword: '' })
  const [showPw, setShowPw] = useState({ current: false, new: false, confirm: false })
  const [saving, setSaving] = useState<string | null>(null)
  const [twoFaEnabled, setTwoFaEnabled] = useState(false)

  useEffect(() => { initialize() }, [])
  useEffect(() => {
    if (!isLoading && !user) router.push('/login?redirect=/settings')
    if (user) {
      setUsernameForm({ username: user.username, displayName: user.display_name || '' })
      setTwoFaEnabled(!!(user as any).two_fa_enabled)
    }
  }, [user, isLoading])

  const handleEmailChange = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!emailForm.newEmail || !emailForm.currentPassword) return toast.error('Fill in all fields')
    setSaving('email')
    try {
      await accountApi.requestEmailChange(emailForm)
      toast.success('A confirmation link has been sent to your current email address.')
      setEmailForm({ newEmail: '', currentPassword: '' })
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to request email change')
    } finally { setSaving(null) }
  }

  const handleUsernameChange = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!usernameForm.username) return toast.error('Username is required')
    setSaving('username')
    try {
      await accountApi.changeUsername(usernameForm)
      toast.success('Profile updated successfully')
      await refresh()
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to update profile')
    } finally { setSaving(null) }
  }

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault()
    if (passwordForm.newPassword !== passwordForm.confirmPassword) return toast.error('Passwords do not match')
    if (passwordForm.newPassword.length < 8) return toast.error('Password must be at least 8 characters')
    setSaving('password')
    try {
      await accountApi.requestPasswordChange(passwordForm)
      toast.success('A confirmation link has been sent to your email address.')
      setPasswordForm({ currentPassword: '', newPassword: '', confirmPassword: '' })
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to request password change')
    } finally { setSaving(null) }
  }

  const handle2faToggle = async () => {
    const newValue = !twoFaEnabled
    setSaving('2fa')
    try {
      await accountApi.toggle2fa(newValue)
      setTwoFaEnabled(newValue)
      toast.success(`Two-factor authentication ${newValue ? 'enabled' : 'disabled'}`)
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to update 2FA setting')
    } finally { setSaving(null) }
  }

  if (isLoading || !user) return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-forge-accent" />
    </div>
  )

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-2xl mx-auto px-4 py-24 pt-32">

        <div className="flex items-center gap-3 mb-8">
          <div className="w-10 h-10 rounded-xl bg-forge-accent/20 border border-forge-accent/30 flex items-center justify-center">
            <Settings size={20} className="text-forge-accent" />
          </div>
          <div>
            <h1 className="font-display text-3xl text-white tracking-wider">ACCOUNT SETTINGS</h1>
            <p className="text-forge-muted text-sm">Manage your account details</p>
          </div>
        </div>

        <div className="space-y-6">

          {/* Username & Display Name */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-5">
              <User size={16} className="text-forge-cyan" />
              <h2 className="font-semibold text-forge-text">Profile Information</h2>
            </div>
            <form onSubmit={handleUsernameChange} className="space-y-4">
              <div>
                <label className="label-forge">Username</label>
                <div className="relative">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-forge-muted text-sm">@</span>
                  <input
                    type="text"
                    value={usernameForm.username}
                    onChange={e => setUsernameForm(f => ({ ...f, username: e.target.value }))}
                    className="input-forge pl-7"
                    placeholder="your_username"
                    minLength={3}
                    maxLength={30}
                  />
                </div>
                <p className="text-forge-muted text-xs mt-1">Letters, numbers, and underscores only. Must be unique.</p>
              </div>
              <div>
                <label className="label-forge">Display Name</label>
                <input
                  type="text"
                  value={usernameForm.displayName}
                  onChange={e => setUsernameForm(f => ({ ...f, displayName: e.target.value }))}
                  className="input-forge"
                  placeholder="Your display name"
                  maxLength={50}
                />
              </div>
              <button type="submit" disabled={saving === 'username'}
                className="btn-primary flex items-center gap-2">
                {saving === 'username' ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                Save Profile
              </button>
            </form>
          </div>

          {/* Change Email */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-5">
              <Mail size={16} className="text-forge-gold" />
              <h2 className="font-semibold text-forge-text">Change Email</h2>
            </div>
            <p className="text-forge-muted text-xs mb-4">Current email: <span className="text-forge-text">{user.email}</span></p>
            <form onSubmit={handleEmailChange} className="space-y-4">
              <div>
                <label className="label-forge">New Email Address</label>
                <input
                  type="email"
                  value={emailForm.newEmail}
                  onChange={e => setEmailForm(f => ({ ...f, newEmail: e.target.value }))}
                  className="input-forge"
                  placeholder="new@email.com"
                />
              </div>
              <div>
                <label className="label-forge">Current Password</label>
                <div className="relative">
                  <input
                    type={showPw.current ? 'text' : 'password'}
                    value={emailForm.currentPassword}
                    onChange={e => setEmailForm(f => ({ ...f, currentPassword: e.target.value }))}
                    className="input-forge pr-10"
                    placeholder="Enter current password"
                  />
                  <button type="button" onClick={() => setShowPw(s => ({ ...s, current: !s.current }))}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                    {showPw.current ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>
              <button type="submit" disabled={saving === 'email'}
                className="btn-primary flex items-center gap-2">
                {saving === 'email' ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                Update Email
              </button>
            </form>
          </div>

          {/* Change Password */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-5">
              <Lock size={16} className="text-forge-accent" />
              <h2 className="font-semibold text-forge-text">Change Password</h2>
            </div>
            <form onSubmit={handlePasswordChange} className="space-y-4">
              <div>
                <label className="label-forge">Current Password</label>
                <div className="relative">
                  <input
                    type={showPw.current ? 'text' : 'password'}
                    value={passwordForm.currentPassword}
                    onChange={e => setPasswordForm(f => ({ ...f, currentPassword: e.target.value }))}
                    className="input-forge pr-10"
                    placeholder="Enter current password"
                  />
                  <button type="button" onClick={() => setShowPw(s => ({ ...s, current: !s.current }))}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                    {showPw.current ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>
              <div>
                <label className="label-forge">New Password</label>
                <div className="relative">
                  <input
                    type={showPw.new ? 'text' : 'password'}
                    value={passwordForm.newPassword}
                    onChange={e => setPasswordForm(f => ({ ...f, newPassword: e.target.value }))}
                    className="input-forge pr-10"
                    placeholder="Min. 8 characters"
                    minLength={8}
                  />
                  <button type="button" onClick={() => setShowPw(s => ({ ...s, new: !s.new }))}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                    {showPw.new ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
              </div>
              <div>
                <label className="label-forge">Confirm New Password</label>
                <div className="relative">
                  <input
                    type={showPw.confirm ? 'text' : 'password'}
                    value={passwordForm.confirmPassword}
                    onChange={e => setPasswordForm(f => ({ ...f, confirmPassword: e.target.value }))}
                    className={clsx(
                      'input-forge pr-10',
                      passwordForm.confirmPassword && passwordForm.newPassword !== passwordForm.confirmPassword
                        ? 'border-forge-accent' : ''
                    )}
                    placeholder="Repeat new password"
                  />
                  <button type="button" onClick={() => setShowPw(s => ({ ...s, confirm: !s.confirm }))}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted hover:text-forge-text">
                    {showPw.confirm ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
                {passwordForm.confirmPassword && passwordForm.newPassword !== passwordForm.confirmPassword && (
                  <p className="text-forge-accent text-xs mt-1">Passwords do not match</p>
                )}
              </div>
              <button type="submit" disabled={saving === 'password'}
                className="btn-primary flex items-center gap-2">
                {saving === 'password' ? <Loader2 size={15} className="animate-spin" /> : <Save size={15} />}
                Change Password
              </button>
            </form>
          </div>

          {/* Two-Factor Authentication */}
          <div className="card p-6">
            <div className="flex items-center gap-2 mb-5">
              <Shield size={16} className="text-forge-cyan" />
              <h2 className="font-semibold text-forge-text">Two-Factor Authentication</h2>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex-1 mr-4">
                <p className="text-forge-text text-sm font-medium mb-1">
                  Email 2FA is currently <span className={twoFaEnabled ? 'text-green-400' : 'text-forge-muted'}>{twoFaEnabled ? 'enabled' : 'disabled'}</span>
                </p>
                <p className="text-forge-muted text-xs">When enabled, you'll receive a 6-digit code by email each time you log in.</p>
              </div>
              <button
                type="button"
                onClick={handle2faToggle}
                disabled={saving === '2fa'}
                className={clsx(
                  'relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none flex-shrink-0',
                  twoFaEnabled ? 'bg-forge-accent' : 'bg-forge-border',
                  saving === '2fa' ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
                )}
                aria-label="Toggle 2FA"
              >
                <span
                  className={clsx(
                    'inline-block h-4 w-4 transform rounded-full bg-white transition-transform',
                    twoFaEnabled ? 'translate-x-6' : 'translate-x-1'
                  )}
                />
              </button>
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}
