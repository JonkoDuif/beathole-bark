'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useRef, useState } from 'react'
import { useAuthStore } from '@/store/auth'
import { beatsApi } from '@/lib/api'
import {
  Music2, Zap, LayoutDashboard, LogOut, ShieldCheck,
  Wallet, Menu, X, Crown, BookOpen, User, Settings,
  MessageSquare, ChevronDown, Shield, Bell
} from 'lucide-react'
import clsx from 'clsx'

export default function Navbar() {
  const { user, logout, initialize } = useAuthStore()
  const pathname = usePathname()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [scrolled, setScrolled] = useState(false)
  const [genRemaining, setGenRemaining] = useState<number | null>(null)
  const [genLimit, setGenLimit] = useState<number | null>(null)
  const [profileOpen, setProfileOpen] = useState(false)
  const profileRef = useRef<HTMLDivElement>(null)
  const [notifications, setNotifications] = useState<Array<{
    id: string; type: string; title: string; body: string | null
    link: string | null; is_read: boolean; created_at: string
  }>>([])
  const [notifOpen, setNotifOpen] = useState(false)
  const [unreadCount, setUnreadCount] = useState(0)
  const notifRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    initialize()
    const handleScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  useEffect(() => {
    if (!user) return
    beatsApi.generationStatus()
      .then((res) => { setGenRemaining(res.data.remaining); setGenLimit(res.data.limit) })
      .catch(() => {})
  }, [user])

  useEffect(() => {
    if (!user) return
    const fetchNotifs = async () => {
      try {
        const token = localStorage.getItem('bf_token')
        if (!token) return
        const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'}/api/notifications`, {
          headers: { Authorization: `Bearer ${token}` }
        })
        if (!res.ok) return
        const data = await res.json()
        setNotifications(data.notifications || [])
        setUnreadCount(data.unread_count || 0)
      } catch {}
    }
    fetchNotifs()
    const interval = setInterval(fetchNotifs, 30000)
    return () => clearInterval(interval)
  }, [user])

  // Close profile dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (profileRef.current && !profileRef.current.contains(e.target as Node)) {
        setProfileOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (notifRef.current && !notifRef.current.contains(e.target as Node)) {
        setNotifOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const markRead = async (id: string) => {
    const token = localStorage.getItem('bf_token')
    if (!token) return
    await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'}/api/notifications/${id}/read`, {
      method: 'PUT',
      headers: { Authorization: `Bearer ${token}` }
    })
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, is_read: true } : n))
    setUnreadCount(prev => Math.max(0, prev - 1))
  }

  const markAllRead = async () => {
    const token = localStorage.getItem('bf_token')
    if (!token) return
    await fetch(`${process.env.NEXT_PUBLIC_API_URL || 'https://api.beathole.com'}/api/notifications/read-all`, {
      method: 'PUT',
      headers: { Authorization: `Bearer ${token}` }
    })
    setNotifications(prev => prev.map(n => ({ ...n, is_read: true })))
    setUnreadCount(0)
  }

  const isPro = user?.subscription_plan === 'pro' && user?.subscription_status === 'active'
  const isStaff = user?.role === 'admin' || user?.role === 'moderator'

  const navLinks = [
    { href: '/marketplace', label: 'Marketplace' },
    { href: '/generate', label: 'Generate', icon: <Zap size={14} /> },
    { href: '/pricing', label: 'Pricing' },
    { href: '/faq', label: 'FAQ' },
  ]

  const avatarLetter = user?.display_name?.[0]?.toUpperCase() || user?.username?.[0]?.toUpperCase() || '?'

  return (
    <nav className={clsx(
      'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
      scrolled ? 'bg-forge-black/95 backdrop-blur-md border-b border-forge-border' : 'bg-transparent'
    )}>
      <div className="max-w-[1600px] mx-auto px-6 sm:px-10 lg:px-20">
        <div className="flex items-center justify-between h-16">

          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 group mr-auto">
            <div className="w-8 h-8 bg-gradient-forge rounded-lg flex items-center justify-center group-hover:shadow-forge transition-shadow">
              <Music2 size={18} className="text-white" />
            </div>
            <span className="font-display text-2xl tracking-wider text-white uppercase">
              BEAT<span className="text-forge-accent">HOLE</span>
            </span>
            <span className="text-[10px] font-mono text-forge-cyan border border-forge-cyan/30 px-1.5 py-0.5 rounded">AI</span>
          </Link>

          {/* Desktop nav */}
          <div className="hidden md:flex items-center gap-4 ml-auto">
            <div className="flex items-center gap-1 border-r border-forge-border pr-4 mr-2">
              {navLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={clsx(
                    'flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all',
                    pathname === link.href
                      ? 'bg-forge-accent/20 text-forge-accent'
                      : 'text-forge-muted hover:text-forge-text hover:bg-forge-card'
                  )}
                >
                  {link.icon}
                  {link.label}
                </Link>
              ))}
            </div>

            {/* Auth area */}
            <div className="flex items-center gap-3">
              {user ? (
                <>
                  {/* Subscription badge */}
                  {isPro ? (
                    <div className="flex items-center gap-1.5 px-2.5 py-1 bg-forge-gold/10 border border-forge-gold/40 rounded-lg">
                      <Crown size={12} className="text-forge-gold" />
                      <span className="text-xs font-bold text-forge-gold">PRO</span>
                    </div>
                  ) : (
                    <Link
                      href="/pricing"
                      className="flex items-center gap-1.5 px-2.5 py-1 bg-forge-accent/10 border border-forge-accent/30 rounded-lg hover:bg-forge-accent/20 transition-colors"
                    >
                      <Zap size={12} className="text-forge-accent" />
                      <span className="text-xs font-bold text-forge-accent">Upgrade</span>
                    </Link>
                  )}

                  {/* Generation counter */}
                  {genRemaining !== null && (
                    <Link href="/credits" className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 bg-forge-dark border border-forge-border rounded-lg hover:border-forge-accent/50 transition-colors" title="Buy more beat credits">
                      <Zap size={11} className="text-forge-muted" />
                      <span className="text-xs font-mono text-forge-muted">{genRemaining}/{genLimit}</span>
                    </Link>
                  )}

                  <Link href="/dashboard" className="btn-ghost text-sm flex items-center gap-1.5">
                    <LayoutDashboard size={14} />
                    Dashboard
                  </Link>

                  <Link href="/library" className="btn-ghost text-sm flex items-center gap-1.5">
                    <BookOpen size={14} />
                    Library
                  </Link>

                  {/* Wallet */}
                  <div className="flex items-center gap-2 px-3 py-1.5 bg-forge-card rounded-lg border border-forge-border">
                    <Wallet size={14} className="text-forge-green" />
                    <span className="text-sm font-mono text-forge-green">
                      €{((user.balance_cents || 0) / 100).toFixed(2)}
                    </span>
                  </div>

                  {/* Notification bell */}
                  <div className="relative" ref={notifRef}>
                    <button
                      onClick={() => setNotifOpen(v => !v)}
                      className="relative p-2 text-forge-muted hover:text-forge-text transition-colors rounded-lg hover:bg-forge-card"
                    >
                      <Bell size={18} />
                      {unreadCount > 0 && (
                        <span className="absolute -top-0.5 -right-0.5 w-4 h-4 bg-forge-accent rounded-full text-[9px] text-white flex items-center justify-center font-bold">
                          {unreadCount > 9 ? '9+' : unreadCount}
                        </span>
                      )}
                    </button>

                    {notifOpen && (
                      <div className="absolute right-0 top-full mt-2 w-80 bg-forge-dark border border-forge-border rounded-xl shadow-lg overflow-hidden z-50">
                        <div className="px-4 py-3 border-b border-forge-border flex items-center justify-between">
                          <span className="text-sm font-bold text-forge-text">Notifications</span>
                          {unreadCount > 0 && (
                            <button onClick={markAllRead} className="text-xs text-forge-accent hover:underline">
                              Mark all read
                            </button>
                          )}
                        </div>
                        <div className="max-h-80 overflow-y-auto">
                          {notifications.length === 0 ? (
                            <div className="py-8 text-center text-sm text-forge-muted">No notifications</div>
                          ) : (
                            notifications.slice(0, 20).map(n => (
                              <div
                                key={n.id}
                                className={`px-4 py-3 border-b border-forge-border/50 last:border-0 cursor-pointer hover:bg-forge-card transition-colors ${!n.is_read ? 'bg-forge-accent/5' : ''}`}
                                onClick={() => {
                                  markRead(n.id)
                                  if (n.link) { window.location.href = n.link }
                                  setNotifOpen(false)
                                }}
                              >
                                <div className="flex items-start gap-2">
                                  {!n.is_read && <span className="w-2 h-2 rounded-full bg-forge-accent flex-shrink-0 mt-1.5" />}
                                  <div className={!n.is_read ? '' : 'pl-4'}>
                                    <p className="text-sm font-semibold text-forge-text leading-tight">{n.title}</p>
                                    {n.body && <p className="text-xs text-forge-muted mt-0.5 leading-tight">{n.body}</p>}
                                    <p className="text-[10px] text-forge-border mt-1">
                                      {new Date(n.created_at).toLocaleDateString('en', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })}
                                    </p>
                                  </div>
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Profile dropdown */}
                  <div className="relative" ref={profileRef}>
                    <button
                      onClick={() => setProfileOpen(v => !v)}
                      className="flex items-center gap-1.5 hover:opacity-80 transition-opacity"
                    >
                      {user.avatar_url ? (
                        <img src={user.avatar_url} alt="" className="w-8 h-8 rounded-full object-cover border border-forge-border" />
                      ) : (
                        <div className="w-8 h-8 rounded-full bg-gradient-forge flex items-center justify-center text-white text-xs font-bold">
                          {avatarLetter}
                        </div>
                      )}
                      <ChevronDown size={12} className={clsx('text-forge-muted transition-transform', profileOpen && 'rotate-180')} />
                    </button>

                    {profileOpen && (
                      <div className="absolute right-0 top-full mt-2 w-52 bg-forge-dark border border-forge-border rounded-xl shadow-lg overflow-hidden z-50">
                        <div className="px-4 py-3 border-b border-forge-border">
                          <div className="text-sm font-semibold text-forge-text truncate">{user.display_name || user.username}</div>
                          <div className="text-xs text-forge-muted truncate">@{user.username}</div>
                        </div>
                        <div className="py-1">
                          <Link href={`/profile/${user.username}`} onClick={() => setProfileOpen(false)}
                            className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-forge-muted hover:text-forge-text hover:bg-forge-card transition-colors">
                            <User size={14} /> View Profile
                          </Link>
                          <Link href="/settings" onClick={() => setProfileOpen(false)}
                            className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-forge-muted hover:text-forge-text hover:bg-forge-card transition-colors">
                            <Settings size={14} /> Account Settings
                          </Link>
                          <Link href="/support" onClick={() => setProfileOpen(false)}
                            className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-forge-muted hover:text-forge-text hover:bg-forge-card transition-colors">
                            <MessageSquare size={14} /> Support
                          </Link>
                          {user.role === 'admin' && (
                            <Link href="/admin" onClick={() => setProfileOpen(false)}
                              className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-forge-muted hover:text-forge-text hover:bg-forge-card transition-colors">
                              <ShieldCheck size={14} /> Admin Panel
                            </Link>
                          )}
                          {user.role === 'moderator' && (
                            <Link href="/mod" onClick={() => setProfileOpen(false)}
                              className="flex items-center gap-2.5 px-4 py-2.5 text-sm text-forge-muted hover:text-forge-text hover:bg-forge-card transition-colors">
                              <Shield size={14} /> Mod Panel
                            </Link>
                          )}
                        </div>
                        <div className="border-t border-forge-border py-1">
                          <button
                            onClick={() => { logout(); setProfileOpen(false) }}
                            className="w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-forge-accent hover:bg-forge-card transition-colors"
                          >
                            <LogOut size={14} /> Sign Out
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </>
              ) : (
                <>
                  <Link href="/login" className="btn-ghost text-sm">Sign In</Link>
                  <Link href="/register" className="btn-primary text-sm py-2">Get Started</Link>
                </>
              )}
            </div>
          </div>

          {/* Mobile toggle */}
          <button
            type="button"
            aria-label="Toggle menu"
            aria-expanded={mobileOpen}
            className="md:hidden p-2 text-forge-muted hover:text-forge-text transition-colors"
            onClick={() => setMobileOpen(prev => !prev)}
          >
            {mobileOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <div className={clsx(
        'md:hidden bg-forge-dark border-t border-forge-border px-4 py-4 space-y-2 transition-all duration-200',
        mobileOpen ? 'block' : 'hidden'
      )}>
        {navLinks.map((link) => (
          <Link
            key={link.href}
            href={link.href}
            onClick={() => setMobileOpen(false)}
            className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card"
          >
            {link.icon}
            {link.label}
          </Link>
        ))}
        {user ? (
          <>
            <div className="px-4 py-2 flex items-center gap-2">
              {isPro ? (
                <span className="flex items-center gap-1.5 text-xs font-bold text-forge-gold">
                  <Crown size={12} /> PRO PLAN
                </span>
              ) : (
                <Link href="/pricing" onClick={() => setMobileOpen(false)} className="text-xs text-forge-accent font-bold flex items-center gap-1">
                  <Zap size={12} /> Upgrade to Pro
                </Link>
              )}
              {genRemaining !== null && (
                <span className="text-xs font-mono text-forge-muted ml-auto">{genRemaining}/{genLimit} beats</span>
              )}
            </div>
            <Link href={`/profile/${user.username}`} onClick={() => setMobileOpen(false)} className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card">
              <User size={14} /> My Profile
            </Link>
            <Link href="/settings" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card">
              <Settings size={14} /> Account Settings
            </Link>
            <Link href="/library" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card">
              <BookOpen size={14} /> My Library
            </Link>
            <Link href="/dashboard" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card">
              <LayoutDashboard size={14} /> Dashboard
            </Link>
            <Link href="/support" onClick={() => setMobileOpen(false)} className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card">
              <MessageSquare size={14} /> Support
            </Link>
            {isStaff && (
              <Link href={user.role === 'admin' ? '/admin' : '/mod'} onClick={() => setMobileOpen(false)} className="flex items-center gap-2 px-4 py-3 rounded-lg text-forge-muted hover:text-forge-text hover:bg-forge-card">
                <ShieldCheck size={14} /> {user.role === 'admin' ? 'Admin Panel' : 'Mod Panel'}
              </Link>
            )}
            <button onClick={() => { logout(); setMobileOpen(false) }} className="w-full text-left flex items-center gap-2 px-4 py-3 rounded-lg text-forge-accent">
              <LogOut size={14} /> Sign Out
            </button>
          </>
        ) : (
          <div className="flex gap-2 pt-2">
            <Link href="/login" onClick={() => setMobileOpen(false)} className="flex-1 btn-secondary text-sm text-center py-2">Sign In</Link>
            <Link href="/register" onClick={() => setMobileOpen(false)} className="flex-1 btn-primary text-sm text-center py-2">Get Started</Link>
          </div>
        )}
      </div>
    </nav>
  )
}
