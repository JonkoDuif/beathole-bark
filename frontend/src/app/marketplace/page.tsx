'use client'
import { useState, useEffect, useCallback } from 'react'
import Navbar from '@/components/Navbar'
import BeatCard from '@/components/BeatCard'
import LicenseModal from '@/components/LicenseModal'
import { beatsApi } from '@/lib/api'
import { Search, SlidersHorizontal, X, ChevronDown, Loader2 } from 'lucide-react'
import clsx from 'clsx'

const GENRES = ['All', 'Trap', 'Hip-Hop', 'Drill', 'R&B', 'Afrobeats', 'Pop', 'Lo-Fi', 'Dancehall', 'Electronic', 'Boom Bap']
const MOODS = ['All', 'Dark', 'Energetic', 'Chill', 'Emotional', 'Aggressive', 'Uplifting', 'Mysterious', 'Romantic']
const SORT_OPTIONS = [
  { value: 'newest', label: 'Newest' },
  { value: 'popular', label: 'Most Played' },
  { value: 'price_asc', label: 'Price: Low to High' },
  { value: 'price_desc', label: 'Price: High to Low' },
]

export default function MarketplacePage() {
  const [beats, setBeats] = useState<any[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [showFilters, setShowFilters] = useState(false)

  const [filters, setFilters] = useState({
    search: '', genre: '', mood: '', bpmMin: '', bpmMax: '', sort: 'newest',
  })

  const [selectedBeat, setSelectedBeat] = useState<any>(null)
  const [beatLicenses, setBeatLicenses] = useState<any[]>([])

  const fetchBeats = useCallback(async (reset = true) => {
    reset ? setLoading(true) : setLoadingMore(true)
    try {
      const params: any = { page: reset ? 1 : page, limit: 12, sort: filters.sort }
      if (filters.search) params.search = filters.search
      if (filters.genre && filters.genre !== 'All') params.genre = filters.genre
      if (filters.mood && filters.mood !== 'All') params.mood = filters.mood
      if (filters.bpmMin) params.bpmMin = filters.bpmMin
      if (filters.bpmMax) params.bpmMax = filters.bpmMax

      const res = await beatsApi.list(params)
      if (reset) {
        setBeats(res.data.beats)
        setPage(2)
      } else {
        setBeats(prev => [...prev, ...res.data.beats])
        setPage(p => p + 1)
      }
      setTotal(res.data.total)
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }, [filters, page])

  useEffect(() => {
    const timer = setTimeout(() => fetchBeats(true), 300)
    return () => clearTimeout(timer)
  }, [filters])

  const handleBuyClick = async (beat: any) => {
    const res = await beatsApi.get(beat.id)
    setSelectedBeat(res.data)
    setBeatLicenses(res.data.licenses || [])
  }

  const clearFilter = (key: string) => setFilters(f => ({ ...f, [key]: '' }))

  const activeFilters = [
    filters.genre && filters.genre !== 'All' && { key: 'genre', label: filters.genre },
    filters.mood && filters.mood !== 'All' && { key: 'mood', label: filters.mood },
    filters.bpmMin && { key: 'bpmMin', label: `BPM ≥ ${filters.bpmMin}` },
    filters.bpmMax && { key: 'bpmMax', label: `BPM ≤ ${filters.bpmMax}` },
  ].filter(Boolean) as { key: string; label: string }[]

  return (
    <div className="min-h-screen bg-forge-black">
      <Navbar />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-20">

        {/* Header */}
        <div className="mb-8">
          <h1 className="font-display text-5xl text-white mb-2">MARKETPLACE</h1>
          <p className="text-forge-muted">
            {(total || 0).toLocaleString()} beats available • AI-generated, instantly licensed
          </p>
        </div>

        {/* Search + filter bar */}
        <div className="flex gap-3 mb-4 flex-wrap">
          <div className="relative flex-1 min-w-[200px]">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-forge-muted" />
            <input
              type="text"
              placeholder="Search beats, genres, producers..."
              value={filters.search}
              onChange={e => setFilters(f => ({ ...f, search: e.target.value }))}
              className="input-forge pl-10"
            />
          </div>

          <button
            onClick={() => setShowFilters(!showFilters)}
            className={clsx(
              'flex items-center gap-2 px-4 py-3 rounded-lg border transition-all text-sm font-medium',
              showFilters
                ? 'bg-forge-accent/20 border-forge-accent text-forge-accent'
                : 'bg-forge-card border-forge-border text-forge-muted hover:text-forge-text'
            )}
          >
            <SlidersHorizontal size={16} />
            Filters
            {activeFilters.length > 0 && (
              <span className="w-4 h-4 rounded-full bg-forge-accent text-white text-[10px] flex items-center justify-center">
                {activeFilters.length}
              </span>
            )}
          </button>

          <div className="relative">
            <select
              value={filters.sort}
              onChange={e => setFilters(f => ({ ...f, sort: e.target.value }))}
              className="input-forge py-3 pr-8 text-sm appearance-none cursor-pointer"
            >
              {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-forge-muted pointer-events-none" />
          </div>
        </div>

        {/* Expanded filters */}
        {showFilters && (
          <div className="card p-4 mb-4 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="label-forge text-xs">Genre</label>
              <div className="flex flex-wrap gap-1">
                {GENRES.map(g => (
                  <button key={g}
                    onClick={() => setFilters(f => ({ ...f, genre: g === 'All' ? '' : g }))}
                    className={clsx(
                      'text-xs px-2.5 py-1 rounded-full border transition-all',
                      (g === 'All' ? !filters.genre : filters.genre === g)
                        ? 'bg-forge-accent/20 border-forge-accent text-forge-accent'
                        : 'border-forge-border text-forge-muted hover:border-forge-accent/50'
                    )}
                  >
                    {g}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="label-forge text-xs">Mood</label>
              <div className="flex flex-wrap gap-1">
                {MOODS.map(m => (
                  <button key={m}
                    onClick={() => setFilters(f => ({ ...f, mood: m === 'All' ? '' : m }))}
                    className={clsx(
                      'text-xs px-2.5 py-1 rounded-full border transition-all',
                      (m === 'All' ? !filters.mood : filters.mood === m)
                        ? 'bg-forge-cyan/20 border-forge-cyan text-forge-cyan'
                        : 'border-forge-border text-forge-muted hover:border-forge-cyan/50'
                    )}
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="label-forge text-xs">BPM Range</label>
              <div className="flex gap-2 items-center">
                <input type="number" placeholder="Min" value={filters.bpmMin}
                  onChange={e => setFilters(f => ({ ...f, bpmMin: e.target.value }))}
                  className="input-forge text-sm py-1.5" min={60} max={200} />
                <span className="text-forge-muted text-xs">—</span>
                <input type="number" placeholder="Max" value={filters.bpmMax}
                  onChange={e => setFilters(f => ({ ...f, bpmMax: e.target.value }))}
                  className="input-forge text-sm py-1.5" min={60} max={200} />
              </div>
            </div>
          </div>
        )}

        {/* Active filter chips */}
        {activeFilters.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-4">
            {activeFilters.map(f => (
              <span key={f.key} className="badge bg-forge-card border border-forge-border text-forge-text text-xs gap-1">
                {f.label}
                <button onClick={() => clearFilter(f.key)} className="hover:text-forge-accent">
                  <X size={10} />
                </button>
              </span>
            ))}
            <button
              onClick={() => setFilters(f => ({ ...f, genre: '', mood: '', bpmMin: '', bpmMax: '' }))}
              className="text-xs text-forge-muted hover:text-forge-accent underline"
            >
              Clear all
            </button>
          </div>
        )}

        {/* Genre quick filters */}
        <div className="flex gap-2 overflow-x-auto pb-2 mb-6 scrollbar-hide">
          {GENRES.map(g => (
            <button
              key={g}
              onClick={() => setFilters(f => ({ ...f, genre: g === 'All' ? '' : g }))}
              className={clsx(
                'whitespace-nowrap px-4 py-1.5 rounded-full text-sm font-medium border transition-all flex-shrink-0',
                (g === 'All' ? !filters.genre : filters.genre === g)
                  ? 'bg-forge-accent text-white border-forge-accent'
                  : 'border-forge-border text-forge-muted hover:border-forge-accent/50 hover:text-forge-text bg-forge-card'
              )}
            >
              {g}
            </button>
          ))}
        </div>

        {/* Beats grid */}
        {loading ? (
          <div className="flex items-center justify-center py-24">
            <Loader2 size={32} className="animate-spin text-forge-accent" />
          </div>
        ) : beats.length === 0 ? (
          <div className="text-center py-24">
            <p className="text-forge-muted text-lg">No beats found. Try adjusting your filters.</p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {beats.map(beat => (
                <BeatCard key={beat.id} beat={beat} onBuy={handleBuyClick} />
              ))}
            </div>

            {/* Load more */}
            {beats.length < total && (
              <div className="text-center mt-10">
                <button
                  onClick={() => fetchBeats(false)}
                  disabled={loadingMore}
                  className="btn-secondary flex items-center gap-2 mx-auto"
                >
                  {loadingMore ? <Loader2 size={16} className="animate-spin" /> : null}
                  Load More ({total - beats.length} remaining)
                </button>
              </div>
            )}
          </>
        )}
      </div>

      {/* License modal */}
      {selectedBeat && (
        <LicenseModal
          beat={selectedBeat}
          licenses={beatLicenses}
          onClose={() => setSelectedBeat(null)}
        />
      )}
    </div>
  )
}
