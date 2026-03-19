'use client'
import { useState } from 'react'
import Link from 'next/link'
import Navbar from '@/components/Navbar'
import { ChevronDown, HelpCircle, Zap, Music2, ShieldCheck, Crown, Wallet, Mic, Scissors } from 'lucide-react'
import clsx from 'clsx'

interface FAQItem { q: string; a: string | React.ReactNode }
interface FAQSection { title: string; icon: React.ReactNode; items: FAQItem[] }

const FAQ_SECTIONS: FAQSection[] = [
  {
    title: 'Beat Generation',
    icon: <Zap size={16} className="text-forge-accent" />,
    items: [
      { q: 'How does AI beat generation work?', a: 'BeatHole uses advanced AI music models (Stability AI or MusicGen) trained on millions of professional tracks. You describe your sound through genre, mood, BPM, key, and a free-text prompt — the AI generates a unique instrumental beat tailored to your input.' },
      { q: 'How many beats can I generate?', a: 'Free accounts get 10 lifetime generations. Pro subscribers get 500 beats per month, resetting on each billing date.' },
      { q: 'How long does generation take?', a: 'Most beats generate in 30–120 seconds. Complex prompts or longer durations may take up to 3 minutes.' },
      { q: 'Can I regenerate a beat if I don\'t like it?', a: 'Yes — each generation uses one credit from your quota. Simply go back to the Generate page and create a new beat with adjusted settings.' },
      { q: 'What genres and styles are supported?', a: 'We support Trap, Drill, Hip-Hop, Boom Bap, R&B, Afrobeats, Dancehall, Lo-Fi, Pop, Electronic, Phonk, Cloud Rap, and any genre you describe in the free-text prompt.' },
    ],
  },
  {
    title: 'AI Studio',
    icon: <Music2 size={16} className="text-forge-accent" />,
    items: [
      { q: 'What is the AI Studio?', a: 'The AI Studio is a browser-based digital audio workstation (DAW). After generating a beat, you can open it in the studio to edit, record vocals, add MIDI instruments, cut and trim clips, apply effects (reverb, delay, EQ, compression), and export a final mix.' },
      { q: 'Can I record my own vocals in the studio?', a: 'Yes. Click the microphone icon on any audio track to arm it for recording, then press Record. Make sure to grant microphone permissions in your browser. The MON toggle controls whether you hear yourself during recording.' },
      { q: 'What is the Piano Roll?', a: 'The Piano Roll is a MIDI editor. Double-click a MIDI clip to open it. You can draw, move, and resize notes on the piano grid to create custom melodies and chord progressions. Click notes on the left keyboard to preview their sound.' },
      { q: 'How do I export my mix?', a: 'Click the Download button in the studio toolbar. Your entire mix (all tracks combined) is exported as a WAV file.' },
      { q: 'Can I undo mistakes in the studio?', a: 'Yes — press Ctrl+Z (or Cmd+Z on Mac) to undo the last action. Multiple undo steps are supported.' },
    ],
  },
  {
    title: 'Marketplace & Licensing',
    icon: <ShieldCheck size={16} className="text-forge-accent" />,
    items: [
      { q: 'How do I publish a beat to the marketplace?', a: 'You need a Pro subscription. Open the beat, click "Publish to Marketplace", set your prices for each license tier, and your beat will be live for buyers worldwide.' },
      { q: 'What license tiers are available?', a: 'Basic (€5 — MP3, up to 50K streams), Standard (€20 — WAV+MP3, commercial use), Premium (€50 — WAV+MP3+Stems, radio & TV), and Exclusive (€200 — full ownership, beat removed from market).' },
      { q: 'What rights does each license grant?', a: 'Basic: personal/non-commercial use only. Standard: commercial releases up to 500K streams, 2 music videos. Premium: unlimited streams, radio/TV broadcasting, sync licensing. Exclusive: unlimited everything, full ownership, no credit required.' },
      { q: 'Can I buy beats without a Pro subscription?', a: 'Yes — anyone can purchase beats from the marketplace. You need a BeatHole account to buy and download licensed beats.' },
      { q: 'Are licenses legally binding?', a: 'Yes. Every purchase generates a legally valid PDF license document with your name, the beat title, license terms, and a unique license ID. Download it from your Library.' },
    ],
  },
  {
    title: 'Subscriptions & Payments',
    icon: <Crown size={16} className="text-forge-accent" />,
    items: [
      { q: 'How much does Pro cost?', a: 'Pro is €19.99/month. You can cancel anytime from your dashboard — your subscription stays active until the end of the billing period.' },
      { q: 'What payment methods are accepted?', a: 'All major credit and debit cards (Visa, Mastercard, Amex), iDEAL, SEPA Direct Debit, and other methods via Stripe depending on your region.' },
      { q: 'Can I cancel my subscription?', a: 'Yes, anytime. Go to Pricing or your Dashboard and click "Manage Subscription". Your Pro features stay active until the end of the current billing cycle.' },
      { q: 'Do unused beat credits roll over?', a: 'No — Pro beat generations reset every month on your billing date. Unused credits do not carry over.' },
      { q: 'Is VAT included in the prices?', a: 'Prices are displayed excluding VAT. Applicable VAT is added at checkout based on your country.' },
    ],
  },
  {
    title: 'Creator Earnings',
    icon: <Wallet size={16} className="text-forge-accent" />,
    items: [
      { q: 'How much do I earn per sale?', a: 'Creators earn 80% of every sale. BeatHole takes a 20% platform fee. Earnings are credited to your balance immediately after a sale completes.' },
      { q: 'How do I withdraw my earnings?', a: 'Connect your Stripe account in the Dashboard, then click Withdraw. Minimum withdrawal is €20. Payouts arrive in 1–5 business days depending on your bank.' },
      { q: 'Do I need to set up Stripe to sell?', a: 'Yes — you need a free Stripe Connect account to receive payouts. Connect it from your Dashboard before publishing beats.' },
    ],
  },
  {
    title: 'Account & Security',
    icon: <ShieldCheck size={16} className="text-forge-accent" />,
    items: [
      { q: 'How do I change my password or email?', a: 'Go to Account Settings (click your profile picture in the top right, then "Account Settings"). You can change your email, username, display name, and password from there.' },
      { q: 'What is a display name vs. username?', a: 'Your username is your unique @handle used for profile URLs and mentions — it can only contain letters, numbers, and underscores. Your display name is the name shown publicly on your profile and can be anything.' },
      { q: 'How do I contact support?', a: 'Use the Support page (accessible from your profile menu). Submit a ticket with your issue and our team will respond as soon as possible.' },
      { q: 'Can my account be banned?', a: 'Accounts that violate our Terms of Service — including copyright infringement, fraudulent activity, or abuse — may be suspended. Appeal by contacting support.' },
    ],
  },
]

function FAQItem({ item }: { item: FAQItem }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="border-b border-forge-border/50 last:border-0">
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-start justify-between gap-4 py-4 text-left"
      >
        <span className={clsx('text-sm font-medium transition-colors', open ? 'text-white' : 'text-forge-text')}>
          {item.q}
        </span>
        <ChevronDown size={16} className={clsx('text-forge-muted flex-shrink-0 mt-0.5 transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <div className="pb-4 text-sm text-forge-muted leading-relaxed">
          {item.a}
        </div>
      )}
    </div>
  )
}

export default function FAQPage() {
  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-3xl mx-auto px-4 py-24 pt-32">

        <div className="text-center mb-12">
          <span className="inline-block text-xs font-mono text-forge-accent border border-forge-accent/30 px-3 py-1 rounded-full mb-4 uppercase tracking-widest">
            Help Center
          </span>
          <h1 className="font-display text-5xl text-white mb-4 tracking-wider">
            FREQUENTLY ASKED<br />
            <span className="text-forge-accent">QUESTIONS</span>
          </h1>
          <p className="text-forge-muted">Everything you need to know about BeatHole AI.</p>
        </div>

        <div className="space-y-6">
          {FAQ_SECTIONS.map((section) => (
            <div key={section.title} className="card overflow-hidden">
              <div className="px-6 py-4 border-b border-forge-border bg-forge-dark/50 flex items-center gap-2">
                {section.icon}
                <h2 className="font-semibold text-forge-text text-sm uppercase tracking-widest">{section.title}</h2>
              </div>
              <div className="px-6">
                {section.items.map((item, i) => (
                  <FAQItem key={i} item={item} />
                ))}
              </div>
            </div>
          ))}
        </div>

        <div className="mt-12 card p-6 text-center">
          <HelpCircle size={28} className="text-forge-muted mx-auto mb-3" />
          <h3 className="font-semibold text-forge-text mb-2">Still have questions?</h3>
          <p className="text-forge-muted text-sm mb-4">Our support team is here to help.</p>
          <Link href="/support" className="btn-primary px-6 py-2 inline-flex items-center gap-2 text-sm">
            Open a Support Ticket
          </Link>
        </div>

      </div>
    </div>
  )
}
