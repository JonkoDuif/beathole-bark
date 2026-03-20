'use client'
import { useState, useEffect, useRef, useCallback } from 'react'
import { useParams, useRouter, useSearchParams } from 'next/navigation'
import Link from 'next/link'
import { useAuthStore } from '@/store/auth'
import { beatsApi, studioApi, presetsApi } from '@/lib/api'
import toast from 'react-hot-toast'
import clsx from 'clsx'
import {
  Play, Pause, Circle, Scissors, MousePointer2, Trash2,
  Volume2, Music2, Sliders, Download, Loader2, X,
  Mic, SkipBack, ArrowLeft, Save, FileAudio,
  Settings, Check, Keyboard, Wand2, TrendingUp,
  ZoomIn, ZoomOut, RotateCcw, Plus, Timer, Minus, Globe,
  UserPlus, Lock, Users, Share2
} from 'lucide-react'

// ─── Types ────────────────────────────────────────────────────────────────────
type Tool = 'pointer' | 'cut' | 'select' | 'fade' | 'gain'
type InstrumentType = 'piano' | 'epiano' | 'synth' | 'bass' | 'pad' | 'drums'

const NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
const TRACK_COLORS = ['#ff3c00','#00d4ff','#ffd700','#00ff88','#ff00aa','#aa00ff','#ff8800','#00ffcc']
const INSTRUMENTS: { id: InstrumentType; label: string }[] = [
  { id: 'piano',  label: 'Piano' },
  { id: 'epiano', label: 'E-Piano' },
  { id: 'synth',  label: 'Synth' },
  { id: 'bass',   label: 'Bass' },
  { id: 'pad',    label: 'Pad' },
  { id: 'drums',  label: 'Drums' },
]
const TIMELINE_DURATION = 180

interface MidiNote { id: string; note: number; startBeat: number; durationBeats: number; velocity: number }
interface Clip { id: string; startTime: number; duration: number; offset: number; fadeIn?: number; fadeOut?: number; clipGain?: number }
interface EQBand {
  id: string
  type: 'lowshelf' | 'highshelf' | 'peaking' | 'lowpass' | 'highpass'
  freq: number    // Hz
  gain: number    // dB (not used for LP/HP)
  Q: number       // bandwidth/slope
}

interface TrackEffects {
  reverbWet: number; delayTime: number; delayFeedback: number
  lowGain: number; midGain: number; highGain: number  // keep for compat
  eqBands: EQBand[]                                    // new parametric EQ
  pitchShift: number; autotuneKey: string; autotuneScale: 'major'|'minor'|'pentatonic'
  autotuneStrength: number; compThreshold: number; compRatio: number
  // Per-effect enabled toggles
  eqEnabled: boolean; reverbEnabled: boolean; delayEnabled: boolean
  compEnabled: boolean; autotuneEnabled: boolean
  chorusRate: number      // 0.1–8 Hz LFO rate
  chorusDepth: number     // 0–30 ms modulation depth
  chorusEnabled: boolean
  distortion: number      // 0–1 drive amount
  distortionEnabled: boolean
  stereoWidth: number     // 0–2 (1 = normal, 2 = double width)
  stereoWidthEnabled: boolean
  tremoloRate: number     // 0.1–20 Hz
  tremoloDepth: number    // 0–1 amplitude modulation depth
  tremoloEnabled: boolean
}
interface Track {
  id: string; type: 'beat'|'audio'|'midi'|'vocal'; name: string; color: string
  volume: number; pan: number; muted: boolean; solo: boolean; armed: boolean
  audioUrl?: string; audioBlob?: Blob; clips: Clip[]; midiNotes: MidiNote[]
  effects: TrackEffects; instrument: InstrumentType
}
interface Preset { id: string; name: string; effects: TrackEffects }

interface TrackGraph {
  input: GainNode; output: StereoPannerNode
  eqNodes: BiquadFilterNode[]               // replaces low, mid, high
  comp: DynamicsCompressorNode
  reverbConv: ConvolverNode; reverbGain: GainNode; dryGain: GainNode
  delayNode: DelayNode; delayFb: GainNode; delayWet: GainNode
  volGain: GainNode; panner: StereoPannerNode
  distortionNode: WaveShaperNode
  tremoloGain: GainNode
  tremoloOsc: OscillatorNode
}

const DEFAULT_EQ_BANDS: EQBand[] = [
  { id: 'b1', type: 'highpass',  freq: 30,   gain: 0, Q: 0.7 },
  { id: 'b2', type: 'lowshelf',  freq: 100,  gain: 0, Q: 1.0 },
  { id: 'b3', type: 'peaking',   freq: 1000, gain: 0, Q: 1.0 },
  { id: 'b4', type: 'peaking',   freq: 5000, gain: 0, Q: 1.0 },
  { id: 'b5', type: 'highshelf', freq: 10000,gain: 0, Q: 1.0 },
]

const DEFAULT_FX: TrackEffects = {
  reverbWet: 0, delayTime: 0, delayFeedback: 0,
  lowGain: 0, midGain: 0, highGain: 0,
  eqBands: DEFAULT_EQ_BANDS,
  pitchShift: 0, autotuneKey: 'none', autotuneScale: 'major',
  autotuneStrength: 0.8, compThreshold: -24, compRatio: 4,
  eqEnabled: true, reverbEnabled: true, delayEnabled: true,
  compEnabled: true, autotuneEnabled: true,
  chorusRate: 1.5, chorusDepth: 5, chorusEnabled: false,
  distortion: 0, distortionEnabled: false,
  stereoWidth: 1, stereoWidthEnabled: false,
  tremoloRate: 4, tremoloDepth: 0, tremoloEnabled: false,
}
const VOCAL_PRESET: TrackEffects = { ...DEFAULT_FX, reverbWet: 0.3, compThreshold: -30, compRatio: 6, autotuneKey: 'C', autotuneStrength: 0.85 }
const TRAP_PRESET:  TrackEffects = { ...DEFAULT_FX, reverbWet: 0.15, delayTime: 0.25, delayFeedback: 0.3, lowGain: 3, highGain: 2 }
const LOFI_PRESET:  TrackEffects = { ...DEFAULT_FX, reverbWet: 0.4, lowGain: -2, highGain: -4, compThreshold: -20, compRatio: 3 }

// ─── Utils ────────────────────────────────────────────────────────────────────
const fmt = (s: number) => `${Math.floor(s/60)}:${String(Math.floor(s%60)).padStart(2,'0')}.${Math.floor((s%1)*10)}`
const uid = () => Math.random().toString(36).slice(2)
const noteToName = (n: number) => NOTE_NAMES[n % 12] + (Math.floor(n / 12) - 2)
const noteToFreq = (n: number) => 440 * Math.pow(2, (n - 69) / 12)

// ─── Autotune utils ───────────────────────────────────────────────────────────
const SCALE_INTERVALS: Record<string, number[]> = {
  major: [0,2,4,5,7,9,11], minor: [0,2,3,5,7,8,10], pentatonic: [0,2,4,7,9],
}
const NOTE_ROOT: Record<string, number> = {
  'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11,
}
function detectPitch(buf: AudioBuffer): number | null {
  const data = buf.getChannelData(0)
  const sr = buf.sampleRate
  const SIZE = Math.min(4096, data.length)
  let rms = 0
  for (let i = 0; i < SIZE; i++) rms += data[i] * data[i]
  if (Math.sqrt(rms / SIZE) < 0.01) return null
  const minLag = Math.floor(sr / 800), maxLag = Math.min(Math.floor(sr / 50), SIZE - 1)
  let bestLag = minLag, bestCorr = -Infinity
  for (let lag = minLag; lag <= maxLag; lag++) {
    let sum = 0
    for (let i = 0; i < SIZE - lag; i++) sum += data[i] * data[i + lag]
    if (sum > bestCorr) { bestCorr = sum; bestLag = lag }
  }
  return sr / bestLag
}
function calcAutotuneDetune(hz: number, key: string, scale: string, strength: number): number {
  if (key === 'none' || strength === 0 || hz <= 0) return 0
  const root = NOTE_ROOT[key] ?? 0
  const intervals = SCALE_INTERVALS[scale] || SCALE_INTERVALS.major
  const midiNote = 12 * Math.log2(hz / 440) + 69
  const noteInOct = ((midiNote % 12) + 12) % 12
  const rel = ((noteInOct - root) + 12) % 12
  let nearest = intervals[0], minDist = 12
  for (const iv of intervals) {
    const d = Math.min(Math.abs(rel - iv), Math.abs(rel - iv + 12), Math.abs(rel - iv - 12))
    if (d < minDist) { minDist = d; nearest = iv }
  }
  let shift = (root + nearest) - noteInOct
  if (shift > 6) shift -= 12
  if (shift < -6) shift += 12
  return shift * 100 * strength
}

function encodeWAV(buf: AudioBuffer): Blob {
  const nc = buf.numberOfChannels, sr = buf.sampleRate, nl = buf.length
  const ab = new ArrayBuffer(44 + nl * nc * 2), v = new DataView(ab)
  const ws = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)) }
  ws(0,'RIFF'); v.setUint32(4,36+nl*nc*2,true); ws(8,'WAVE'); ws(12,'fmt ')
  v.setUint32(16,16,true); v.setUint16(20,1,true); v.setUint16(22,nc,true)
  v.setUint32(24,sr,true); v.setUint32(28,sr*nc*2,true); v.setUint16(32,nc*2,true); v.setUint16(34,16,true)
  ws(36,'data'); v.setUint32(40,nl*nc*2,true)
  let offset = 44
  for (let i = 0; i < nl; i++) for (let c = 0; c < nc; c++) {
    const sv = Math.max(-1, Math.min(1, buf.getChannelData(c)[i]))
    v.setInt16(offset, sv < 0 ? sv * 0x8000 : sv * 0x7FFF, true); offset += 2
  }
  return new Blob([ab], { type: 'audio/wav' })
}

function createImpulse(ctx: BaseAudioContext): AudioBuffer {
  const len = Math.floor(ctx.sampleRate * 2.5), ibuf = ctx.createBuffer(2, len, ctx.sampleRate)
  for (let c = 0; c < 2; c++) { const d = ibuf.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random()*2-1)*Math.pow(1-i/len, 2.5) }
  return ibuf
}

// ─── Persistent Track Graph ───────────────────────────────────────────────────
function buildTrackGraph(ctx: AudioContext, fx: TrackEffects, volume: number, pan: number, impulse: AudioBuffer): TrackGraph {
  const input   = ctx.createGain()

  // Build EQ chain from bands
  const eqNodes: BiquadFilterNode[] = (fx.eqEnabled ? fx.eqBands : DEFAULT_EQ_BANDS.map(b => ({...b, gain: 0}))).map(band => {
    const f = ctx.createBiquadFilter()
    f.type = band.type as BiquadFilterType
    f.frequency.value = band.freq
    f.gain.value = band.gain
    f.Q.value = band.Q
    return f
  })

  // Chain EQ nodes
  for (let i = 0; i < eqNodes.length - 1; i++) eqNodes[i].connect(eqNodes[i + 1])
  const eqIn  = eqNodes[0]
  const eqOut = eqNodes[eqNodes.length - 1]

  const comp = ctx.createDynamicsCompressor()
  comp.threshold.value = fx.compEnabled ? fx.compThreshold : -100
  comp.ratio.value     = fx.compEnabled ? fx.compRatio : 1
  comp.attack.value = 0.003; comp.release.value = 0.25

  const dryGain    = ctx.createGain(); dryGain.gain.value    = fx.reverbEnabled ? (1 - fx.reverbWet * 0.5) : 1
  const reverbConv = ctx.createConvolver(); reverbConv.buffer = impulse
  const reverbGain = ctx.createGain(); reverbGain.gain.value = fx.reverbEnabled ? fx.reverbWet : 0
  const delayNode  = ctx.createDelay(1.0); delayNode.delayTime.value = fx.delayEnabled ? fx.delayTime : 0
  const delayFb    = ctx.createGain(); delayFb.gain.value    = fx.delayEnabled ? fx.delayFeedback : 0
  const delayWet   = ctx.createGain(); delayWet.gain.value   = (fx.delayEnabled && fx.delayTime > 0) ? 0.4 : 0
  const volGain    = ctx.createGain(); volGain.gain.value    = volume
  const panner     = ctx.createStereoPanner(); panner.pan.value = pan

  // Distortion — waveshaper before comp
  const distortionNode = ctx.createWaveShaper()
  function makeDistortionCurve(amount: number): Float32Array<ArrayBuffer> {
    const n = 256
    const curve = new Float32Array(new ArrayBuffer(n * 4))
    const k = amount * 200
    for (let i = 0; i < n; i++) {
      const x = (i * 2) / n - 1
      curve[i] = k > 0 ? ((Math.PI + k) * x) / (Math.PI + k * Math.abs(x)) : x
    }
    return curve
  }
  distortionNode.curve = fx.distortionEnabled && fx.distortion > 0
    ? makeDistortionCurve(fx.distortion)
    : null

  // Tremolo — amplitude LFO after volume
  const tremoloOsc = ctx.createOscillator()
  const tremoloGain = ctx.createGain()
  tremoloOsc.type = 'sine'
  tremoloOsc.frequency.value = fx.tremoloRate
  tremoloGain.gain.value = fx.tremoloEnabled && fx.tremoloDepth > 0 ? fx.tremoloDepth : 0
  tremoloOsc.connect(tremoloGain)
  tremoloGain.connect(volGain.gain)  // modulate volume gain
  tremoloOsc.start()

  input.connect(distortionNode); distortionNode.connect(eqIn); eqOut.connect(comp)
  comp.connect(dryGain); dryGain.connect(volGain)
  comp.connect(reverbConv); reverbConv.connect(reverbGain); reverbGain.connect(volGain)
  comp.connect(delayNode); delayNode.connect(delayFb); delayFb.connect(delayNode); delayFb.connect(delayWet); delayWet.connect(volGain)
  volGain.connect(panner)

  return { input, output: panner, eqNodes, comp, reverbConv, reverbGain, dryGain, delayNode, delayFb, delayWet, volGain, panner, distortionNode, tremoloGain, tremoloOsc }
}

function updateTrackGraph(graph: TrackGraph, track: Track) {
  const fx = track.effects
  // Update EQ bands - if count changed, can't update in place, just update gains/freqs
  const bands = fx.eqEnabled ? fx.eqBands : DEFAULT_EQ_BANDS.map(b => ({...b, gain: 0}))
  bands.forEach((band, i) => {
    if (i < graph.eqNodes.length) {
      graph.eqNodes[i].frequency.value = band.freq
      graph.eqNodes[i].gain.value      = band.gain
      graph.eqNodes[i].Q.value         = band.Q
    }
  })
  graph.comp.threshold.value   = fx.compEnabled ? fx.compThreshold : -100
  graph.comp.ratio.value       = fx.compEnabled ? fx.compRatio : 1
  graph.reverbGain.gain.value  = fx.reverbEnabled ? fx.reverbWet : 0
  graph.dryGain.gain.value     = fx.reverbEnabled ? (1 - fx.reverbWet * 0.5) : 1
  graph.delayNode.delayTime.value = fx.delayEnabled ? fx.delayTime : 0
  graph.delayFb.gain.value     = fx.delayEnabled ? fx.delayFeedback : 0
  graph.delayWet.gain.value    = (fx.delayEnabled && fx.delayTime > 0) ? 0.4 : 0
  graph.volGain.gain.value     = track.muted ? 0 : track.volume
  graph.panner.pan.value       = track.pan
  // Distortion update
  if (fx.distortionEnabled && fx.distortion > 0) {
    const n = 256, k = fx.distortion * 200
    const curve = new Float32Array(new ArrayBuffer(n * 4))
    for (let i = 0; i < n; i++) { const x = (i * 2) / n - 1; curve[i] = ((Math.PI + k) * x) / (Math.PI + k * Math.abs(x)) }
    graph.distortionNode.curve = curve
  } else {
    graph.distortionNode.curve = null
  }
  // Tremolo update
  graph.tremoloOsc.frequency.value = fx.tremoloRate
  graph.tremoloGain.gain.value = fx.tremoloEnabled ? fx.tremoloDepth : 0
}

// For offline export (OfflineAudioContext)
function buildFxChainOffline(
  ctx: OfflineAudioContext,
  fx: TrackEffects, volume: number, pan: number, impulse: AudioBuffer
): { input: AudioNode; output: AudioNode } {
  // EQ chain
  const bands = fx.eqEnabled ? fx.eqBands : DEFAULT_EQ_BANDS.map(b => ({...b, gain: 0}))
  const eqNodes = bands.map(band => {
    const f = ctx.createBiquadFilter()
    f.type = band.type as BiquadFilterType
    f.frequency.value = band.freq
    f.gain.value = band.gain
    f.Q.value = band.Q
    return f
  })
  for (let i = 0; i < eqNodes.length - 1; i++) eqNodes[i].connect(eqNodes[i + 1])
  const eqOut = eqNodes[eqNodes.length - 1]

  const comp = ctx.createDynamicsCompressor()
  comp.threshold.value = fx.compEnabled ? fx.compThreshold : -100
  comp.ratio.value     = fx.compEnabled ? fx.compRatio : 1
  comp.attack.value = 0.003; comp.release.value = 0.25
  const volGain = ctx.createGain(); volGain.gain.value = volume
  const panner  = ctx.createStereoPanner(); panner.pan.value = pan
  eqOut.connect(comp)
  if (fx.reverbEnabled && fx.reverbWet > 0.005) {
    const reverb     = ctx.createConvolver(); reverb.buffer = impulse
    const reverbGain = ctx.createGain(); reverbGain.gain.value = fx.reverbWet
    const dryGain    = ctx.createGain(); dryGain.gain.value = 1 - fx.reverbWet * 0.5
    comp.connect(dryGain); dryGain.connect(volGain)
    comp.connect(reverb); reverb.connect(reverbGain); reverbGain.connect(volGain)
  } else {
    comp.connect(volGain)
  }
  if (fx.delayEnabled && fx.delayTime > 0) {
    const delay   = ctx.createDelay(1.0); delay.delayTime.value = fx.delayTime
    const delayFb = ctx.createGain(); delayFb.gain.value = fx.delayFeedback
    const delayWetG = ctx.createGain(); delayWetG.gain.value = 0.4
    comp.connect(delay); delay.connect(delayFb); delayFb.connect(delay); delayFb.connect(delayWetG); delayWetG.connect(volGain)
  }
  volGain.connect(panner)
  return { input: eqNodes[0], output: panner }
}

// ─── Drum Synthesis ───────────────────────────────────────────────────────────
// GM drum note map: 36=Kick, 38=Snare, 39=Clap, 42=Closed HH, 46=Open HH, 49=Crash, 51=Ride
function synthesizeDrum(ctx: BaseAudioContext, note: number, velocity: number, output: AudioNode, startTime: number) {
  const vel = velocity / 127

  if (note === 36) {
    // Kick: sine with freq envelope + noise thump
    const osc = ctx.createOscillator()
    const env = ctx.createGain()
    osc.type = 'sine'
    osc.frequency.setValueAtTime(150, startTime)
    osc.frequency.exponentialRampToValueAtTime(50, startTime + 0.12)
    env.gain.setValueAtTime(vel * 1.0, startTime)
    env.gain.exponentialRampToValueAtTime(0.001, startTime + 0.35)
    osc.connect(env); env.connect(output)
    osc.start(startTime); osc.stop(startTime + 0.35)
  } else if (note === 38 || note === 39) {
    // Snare / Clap: bandpass noise + body oscillator
    const bufSize = ctx.sampleRate * 0.2
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1
    const noise = ctx.createBufferSource(); noise.buffer = buf
    const filt = ctx.createBiquadFilter(); filt.type = 'bandpass'; filt.frequency.value = 3000; filt.Q.value = 0.8
    const env = ctx.createGain()
    env.gain.setValueAtTime(vel * 0.9, startTime)
    env.gain.exponentialRampToValueAtTime(0.001, startTime + (note === 39 ? 0.08 : 0.18))
    noise.connect(filt); filt.connect(env); env.connect(output)
    noise.start(startTime); noise.stop(startTime + 0.2)

    // Body tone for snare
    if (note === 38) {
      const body = ctx.createOscillator(); body.type = 'triangle'; body.frequency.value = 200
      const bEnv = ctx.createGain()
      bEnv.gain.setValueAtTime(vel * 0.3, startTime)
      bEnv.gain.exponentialRampToValueAtTime(0.001, startTime + 0.05)
      body.connect(bEnv); bEnv.connect(output)
      body.start(startTime); body.stop(startTime + 0.08)
    }
  } else if (note === 42) {
    // Closed hi-hat: very short high-pass noise
    const bufSize = ctx.sampleRate * 0.05
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1
    const noise = ctx.createBufferSource(); noise.buffer = buf
    const filt = ctx.createBiquadFilter(); filt.type = 'highpass'; filt.frequency.value = 8000
    const env = ctx.createGain()
    env.gain.setValueAtTime(vel * 0.5, startTime)
    env.gain.exponentialRampToValueAtTime(0.001, startTime + 0.04)
    noise.connect(filt); filt.connect(env); env.connect(output)
    noise.start(startTime); noise.stop(startTime + 0.05)
  } else if (note === 46) {
    // Open hi-hat: longer high-pass noise
    const bufSize = ctx.sampleRate * 0.25
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1
    const noise = ctx.createBufferSource(); noise.buffer = buf
    const filt = ctx.createBiquadFilter(); filt.type = 'highpass'; filt.frequency.value = 7000
    const env = ctx.createGain()
    env.gain.setValueAtTime(vel * 0.45, startTime)
    env.gain.exponentialRampToValueAtTime(0.001, startTime + 0.22)
    noise.connect(filt); filt.connect(env); env.connect(output)
    noise.start(startTime); noise.stop(startTime + 0.25)
  } else {
    // Generic percussive: short bandpass noise
    const bufSize = ctx.sampleRate * 0.1
    const buf = ctx.createBuffer(1, bufSize, ctx.sampleRate)
    const data = buf.getChannelData(0)
    for (let i = 0; i < bufSize; i++) data[i] = Math.random() * 2 - 1
    const noise = ctx.createBufferSource(); noise.buffer = buf
    const filt = ctx.createBiquadFilter(); filt.type = 'bandpass'; filt.frequency.value = 2000
    const env = ctx.createGain()
    env.gain.setValueAtTime(vel * 0.4, startTime)
    env.gain.exponentialRampToValueAtTime(0.001, startTime + 0.08)
    noise.connect(filt); filt.connect(env); env.connect(output)
    noise.start(startTime); noise.stop(startTime + 0.1)
  }
}

// ─── MIDI Synthesis ───────────────────────────────────────────────────────────
function getADSR(inst: InstrumentType) {
  switch (inst) {
    case 'piano':  return { attack: 0.005, decay: 0.3,  sustain: 0.4, release: 0.3 }
    case 'epiano': return { attack: 0.01,  decay: 0.2,  sustain: 0.6, release: 0.4 }
    case 'synth':  return { attack: 0.02,  decay: 0.15, sustain: 0.7, release: 0.1 }
    case 'bass':   return { attack: 0.01,  decay: 0.1,  sustain: 0.8, release: 0.05 }
    case 'pad':    return { attack: 0.3,   decay: 0.2,  sustain: 0.8, release: 0.5 }
    case 'drums':  return { attack: 0.001, decay: 0.05, sustain: 0.0, release: 0.05 }
  }
}

function synthesizeMidiNote(
  ctx: BaseAudioContext, note: number, velocity: number, duration: number,
  instrument: InstrumentType, output: AudioNode, startTime: number
) {
  if (instrument === 'drums') {
    synthesizeDrum(ctx, note, velocity, output, startTime)
    return
  }

  const freq = noteToFreq(note)
  const vel  = velocity / 127
  const adsr = getADSR(instrument)

  const env = ctx.createGain()
  env.connect(output)
  env.gain.setValueAtTime(0, startTime)
  env.gain.linearRampToValueAtTime(vel * 0.7, startTime + adsr.attack)
  env.gain.linearRampToValueAtTime(vel * adsr.sustain * 0.7, startTime + adsr.attack + adsr.decay)
  env.gain.setValueAtTime(vel * adsr.sustain * 0.7, startTime + duration)
  env.gain.linearRampToValueAtTime(0, startTime + duration + adsr.release)

  const stopAt = startTime + duration + adsr.release + 0.1
  const oscs: OscillatorNode[] = []

  if (instrument === 'piano') {
    const o1 = ctx.createOscillator(); o1.type = 'triangle'; o1.frequency.value = freq
    const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = freq * 2
    const o3 = ctx.createOscillator(); o3.type = 'sine'; o3.frequency.value = freq * 3
    const g2 = ctx.createGain(); g2.gain.value = 0.25
    const g3 = ctx.createGain(); g3.gain.value = 0.08
    o1.connect(env); o2.connect(g2); g2.connect(env); o3.connect(g3); g3.connect(env)
    oscs.push(o1, o2, o3)
  } else if (instrument === 'epiano') {
    const o1 = ctx.createOscillator(); o1.type = 'sine'; o1.frequency.value = freq
    const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = freq * 2.01
    const g2 = ctx.createGain(); g2.gain.value = 0.3
    o1.connect(env); o2.connect(g2); g2.connect(env)
    oscs.push(o1, o2)
  } else if (instrument === 'synth') {
    const osc  = ctx.createOscillator(); osc.type = 'sawtooth'; osc.frequency.value = freq
    const filt = ctx.createBiquadFilter(); filt.type = 'lowpass'; filt.frequency.value = 2000; filt.Q.value = 2
    filt.frequency.setValueAtTime(freq * 2, startTime)
    filt.frequency.linearRampToValueAtTime(freq * 8, startTime + 0.1)
    filt.frequency.linearRampToValueAtTime(freq * 3, startTime + 0.4)
    osc.connect(filt); filt.connect(env)
    oscs.push(osc)
  } else if (instrument === 'bass') {
    const osc  = ctx.createOscillator(); osc.type = 'triangle'; osc.frequency.value = freq
    const sub  = ctx.createOscillator(); sub.type = 'sine'; sub.frequency.value = freq / 2
    const filt = ctx.createBiquadFilter(); filt.type = 'lowpass'; filt.frequency.value = 600
    const gsub = ctx.createGain(); gsub.gain.value = 0.5
    osc.connect(filt); sub.connect(gsub); gsub.connect(filt); filt.connect(env)
    oscs.push(osc, sub)
  } else if (instrument === 'pad') {
    const o1 = ctx.createOscillator(); o1.type = 'sine'; o1.frequency.value = freq
    const o2 = ctx.createOscillator(); o2.type = 'sine'; o2.frequency.value = freq * 1.005
    const o3 = ctx.createOscillator(); o3.type = 'triangle'; o3.frequency.value = freq * 0.5
    const g3 = ctx.createGain(); g3.gain.value = 0.3
    o1.connect(env); o2.connect(env); o3.connect(g3); g3.connect(env)
    oscs.push(o1, o2, o3)
  }

  oscs.forEach(o => { o.start(startTime); o.stop(stopAt) })
}

// ─── Rotary Knob Component ────────────────────────────────────────────────────
function Knob({ label, value, min, max, step, format: fmtFn, onChange, color = '#ff3c00' }: {
  label: string; value: number; min: number; max: number; step: number
  format?: (v: number) => string; onChange: (v: number) => void; color?: string
}) {
  const dragRef = useRef<{ y: number; startVal: number } | null>(null)
  const pct = Math.max(0, Math.min(1, (value - min) / (max - min)))

  const polarToXY = (deg: number, r: number) => {
    const rad = ((deg - 90) * Math.PI) / 180
    return { x: 20 + r * Math.cos(rad), y: 20 + r * Math.sin(rad) }
  }
  const arc = (s: number, e: number, r: number) => {
    const sp = polarToXY(s, r), ep = polarToXY(e, r)
    const large = e - s > 180 ? 1 : 0
    return `M ${sp.x} ${sp.y} A ${r} ${r} 0 ${large} 1 ${ep.x} ${ep.y}`
  }
  const endDeg = 135 + pct * 270
  const fmt2 = fmtFn || ((v: number) => v.toFixed(1))
  const dot = polarToXY(endDeg, 11)

  return (
    <div
      className="flex flex-col items-center gap-0.5 select-none"
      style={{ cursor: 'ns-resize' }}
      onMouseDown={e => {
        e.preventDefault()
        dragRef.current = { y: e.clientY, startVal: value }
        const onMove = (ev: MouseEvent) => {
          if (!dragRef.current) return
          const dy = dragRef.current.y - ev.clientY
          const range = max - min
          const nv = Math.max(min, Math.min(max, dragRef.current.startVal + dy * (range / 100)))
          onChange(parseFloat((Math.round(nv / step) * step).toFixed(10)))
        }
        const onUp = () => {
          dragRef.current = null
          window.removeEventListener('mousemove', onMove)
          window.removeEventListener('mouseup', onUp)
        }
        window.addEventListener('mousemove', onMove)
        window.addEventListener('mouseup', onUp)
      }}
      onDoubleClick={() => onChange(0)}
    >
      <svg width={40} height={40} viewBox="0 0 40 40">
        <path d={arc(135, 405, 14)} fill="none" stroke="#2a2a2a" strokeWidth={4} strokeLinecap="round" />
        {pct > 0.002 && <path d={arc(135, endDeg, 14)} fill="none" stroke={color} strokeWidth={4} strokeLinecap="round" />}
        <circle cx={dot.x} cy={dot.y} r={2.5} fill="white" />
        <circle cx={20} cy={20} r={8} fill="#111" />
      </svg>
      <span className="text-[9px] font-mono text-forge-accent">{fmt2(value)}</span>
      <span className="text-[8px] text-forge-muted uppercase tracking-widest text-center w-12 truncate">{label}</span>
    </div>
  )
}

// ─── Piano Roll ───────────────────────────────────────────────────────────────
const PIANO_ROLL_NOTES = 72
const PIANO_ROLL_START_NOTE = 24
const ROW_H = 14
const BEAT_W = 80
const KEY_W = 48

function PianoRoll({ notes, instrument, onChange, onInstrumentChange, onClose, bpm = 120 }: {
  notes: MidiNote[]; instrument: InstrumentType
  onChange: (n: MidiNote[]) => void; onInstrumentChange: (i: InstrumentType) => void; onClose: () => void
  bpm?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [localNotes, setLocalNotes] = useState<MidiNote[]>(notes)
  const [bars, setBars] = useState(8)
  const [dragging, setDragging] = useState<{ id: string; mode: 'move'|'resize'; startX: number; startY: number; origStart: number; origDur: number; origNote: number } | null>(null)
  const [prTool, setPrTool] = useState<'draw'|'erase'>('draw')
  const [velocity, setVelocity] = useState(100)
  const [snap, setSnap] = useState(0.25)
  const [prPlaying, setPrPlaying] = useState(false)
  const prAudioCtxRef = useRef<AudioContext | null>(null)
  const totalBeats = bars * 4

  const handlePrPlay = () => {
    if (prPlaying) {
      if (prAudioCtxRef.current) { prAudioCtxRef.current.close(); prAudioCtxRef.current = null }
      setPrPlaying(false); return
    }
    if (localNotes.length === 0) { toast('No notes to play'); return }
    const ctx = new AudioContext(); prAudioCtxRef.current = ctx
    const secPerBeat = 60 / bpm
    for (const n of localNotes) {
      synthesizeMidiNote(ctx, n.note, n.velocity, n.durationBeats * secPerBeat, instrument, ctx.destination, ctx.currentTime + 0.05 + n.startBeat * secPerBeat)
    }
    setPrPlaying(true)
    const maxEnd = Math.max(...localNotes.map(n => n.startBeat + n.durationBeats)) * secPerBeat
    setTimeout(() => { setPrPlaying(false); prAudioCtxRef.current = null }, (maxEnd + 0.8) * 1000)
  }

  const previewKey = (note: number) => {
    const ctx = new AudioContext()
    synthesizeMidiNote(ctx, note, velocity, 0.5, instrument, ctx.destination, ctx.currentTime + 0.01)
    setTimeout(() => ctx.close(), 2000)
  }
  const isBlack = (n: number) => [1,3,6,8,10].includes(n % 12)

  const draw = useCallback(() => {
    const c = canvasRef.current; if (!c) return
    const ctx = c.getContext('2d')!
    const W = KEY_W + totalBeats * BEAT_W
    c.width = W; c.height = PIANO_ROLL_NOTES * ROW_H

    for (let i = 0; i < PIANO_ROLL_NOTES; i++) {
      const noteNum = PIANO_ROLL_NOTES - 1 - i + PIANO_ROLL_START_NOTE
      ctx.fillStyle = isBlack(noteNum) ? '#1a1a1a' : '#222'
      ctx.fillRect(KEY_W, i * ROW_H, W - KEY_W, ROW_H)
      ctx.strokeStyle = '#333'; ctx.lineWidth = 0.5
      ctx.beginPath(); ctx.moveTo(KEY_W, i * ROW_H); ctx.lineTo(W, i * ROW_H); ctx.stroke()
      if (noteNum % 12 === 0) {
        ctx.strokeStyle = '#444'; ctx.lineWidth = 1
        ctx.beginPath(); ctx.moveTo(KEY_W, i * ROW_H); ctx.lineTo(W, i * ROW_H); ctx.stroke()
      }
    }
    for (let b = 0; b <= totalBeats; b++) {
      const x = KEY_W + b * BEAT_W
      ctx.strokeStyle = b % 4 === 0 ? '#555' : '#2a2a2a'; ctx.lineWidth = b % 4 === 0 ? 1.5 : 0.5
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke()
      if (b % 4 === 0) { ctx.fillStyle = '#666'; ctx.font = '10px monospace'; ctx.fillText(`${b/4+1}`, x+2, 10) }
    }
    for (let b = 0; b < totalBeats; b++) {
      for (let s = 1; s < 4; s++) {
        const x = KEY_W + (b + s * 0.25) * BEAT_W
        ctx.strokeStyle = '#282828'; ctx.lineWidth = 0.5
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, c.height); ctx.stroke()
      }
    }
    for (let i = 0; i < PIANO_ROLL_NOTES; i++) {
      const noteNum = PIANO_ROLL_NOTES - 1 - i + PIANO_ROLL_START_NOTE
      const y = i * ROW_H
      if (isBlack(noteNum)) {
        ctx.fillStyle = '#111'; ctx.fillRect(0, y, KEY_W * 0.65, ROW_H)
      } else {
        ctx.fillStyle = '#ddd'; ctx.fillRect(0, y, KEY_W, ROW_H - 0.5)
        ctx.strokeStyle = '#888'; ctx.lineWidth = 0.5; ctx.strokeRect(0, y, KEY_W, ROW_H - 0.5)
        if (noteNum % 12 === 0) { ctx.fillStyle = '#555'; ctx.font = '9px sans-serif'; ctx.fillText(noteToName(noteNum), 2, y + ROW_H - 3) }
      }
    }
    for (const note of localNotes) {
      const row = PIANO_ROLL_NOTES - 1 - (note.note - PIANO_ROLL_START_NOTE)
      if (row < 0 || row >= PIANO_ROLL_NOTES) continue
      const x = KEY_W + note.startBeat * BEAT_W
      const w = Math.max(note.durationBeats * BEAT_W - 2, 2)
      const y = row * ROW_H + 1
      const h = ROW_H - 2
      const alpha = 0.5 + (note.velocity / 127) * 0.5
      ctx.fillStyle = `rgba(255,60,0,${alpha})`
      ctx.beginPath(); ctx.roundRect(x, y, w, h, 2); ctx.fill()
      ctx.strokeStyle = 'rgba(255,150,80,0.8)'; ctx.lineWidth = 1; ctx.strokeRect(x, y, w, h)
    }
  }, [localNotes, totalBeats])

  useEffect(() => { draw() }, [draw])

  const snapBeat = (b: number) => Math.round(b / snap) * snap
  const getPos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current!.getBoundingClientRect()
    const sx = canvasRef.current!.width / rect.width
    const sy = canvasRef.current!.height / rect.height
    const cx = (e.clientX - rect.left) * sx
    const cy = (e.clientY - rect.top) * sy
    const beat = snapBeat((cx - KEY_W) / BEAT_W)
    const noteRow = Math.floor(cy / ROW_H)
    const note = PIANO_ROLL_NOTES - 1 - noteRow + PIANO_ROLL_START_NOTE
    return { cx, cy, beat, note, noteRow }
  }

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { cx, cy, beat, note, noteRow } = getPos(e)
    if (cx < KEY_W) { previewKey(note); return }
    for (const n of localNotes) {
      const row = PIANO_ROLL_NOTES - 1 - (n.note - PIANO_ROLL_START_NOTE)
      const x = KEY_W + n.startBeat * BEAT_W; const w = n.durationBeats * BEAT_W; const y = row * ROW_H
      if (cy >= y && cy < y + ROW_H && cx >= x && cx < x + w) {
        if (prTool === 'erase' || e.button === 2) { setLocalNotes(prev => prev.filter(nn => nn.id !== n.id)); return }
        const isResize = cx > x + w - 10
        setDragging({ id: n.id, mode: isResize ? 'resize' : 'move', startX: cx, startY: cy, origStart: n.startBeat, origDur: n.durationBeats, origNote: n.note })
        return
      }
    }
    if (prTool === 'draw' && beat >= 0 && beat < totalBeats && noteRow >= 0 && noteRow < PIANO_ROLL_NOTES) {
      const newNote: MidiNote = { id: uid(), note, startBeat: beat, durationBeats: snap, velocity }
      setLocalNotes(prev => [...prev, newNote])
      setDragging({ id: newNote.id, mode: 'resize', startX: cx, startY: cy, origStart: beat, origDur: snap, origNote: note })
    }
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragging) return
    const { cx, note } = getPos(e)
    const dx = (cx - dragging.startX) / BEAT_W
    setLocalNotes(prev => prev.map(n => {
      if (n.id !== dragging.id) return n
      if (dragging.mode === 'move') {
        return { ...n, startBeat: Math.max(0, snapBeat(dragging.origStart + dx)), note: Math.max(PIANO_ROLL_START_NOTE, Math.min(PIANO_ROLL_START_NOTE + PIANO_ROLL_NOTES - 1, note)) }
      } else {
        return { ...n, durationBeats: Math.max(snap, snapBeat(dragging.origDur + dx)) }
      }
    }))
  }

  return (
    <div className="fixed inset-0 z-50 bg-black/95 flex flex-col">
      <div className="flex items-center gap-3 px-4 py-3 bg-forge-dark border-b border-forge-border flex-shrink-0">
        <span className="text-forge-accent text-lg">⊞</span>
        <span className="font-display text-white">PIANO ROLL</span>
        <div className="flex items-center gap-1 ml-4 bg-forge-black rounded-lg p-1 border border-forge-border">
          {INSTRUMENTS.map(inst => (
            <button key={inst.id} onClick={() => onInstrumentChange(inst.id)}
              className={clsx('px-3 py-1 rounded text-xs font-bold transition-colors', instrument === inst.id ? 'bg-forge-accent text-white' : 'text-forge-muted hover:text-forge-text')}>
              {inst.label}
            </button>
          ))}
        </div>
        <div className="flex items-center gap-2 ml-4">
          <button onClick={() => setPrTool('draw')} className={clsx('px-3 py-1 rounded text-xs font-bold', prTool==='draw' ? 'bg-forge-accent text-white' : 'bg-forge-card text-forge-muted')}>Draw</button>
          <button onClick={() => setPrTool('erase')} className={clsx('px-3 py-1 rounded text-xs font-bold', prTool==='erase' ? 'bg-red-500 text-white' : 'bg-forge-card text-forge-muted')}>Erase</button>
        </div>
        <div className="flex items-center gap-2 ml-2">
          <span className="text-xs text-forge-muted">Snap:</span>
          <select value={snap} onChange={e => setSnap(Number(e.target.value))} className="bg-forge-card text-forge-text text-xs rounded px-2 py-1 border border-forge-border">
            <option value={1}>1 Beat</option><option value={0.5}>1/2</option><option value={0.25}>1/4</option><option value={0.125}>1/8</option>
          </select>
        </div>
        <div className="flex items-center gap-2 ml-2">
          <span className="text-xs text-forge-muted">Bars:</span>
          <select value={bars} onChange={e => setBars(Number(e.target.value))} className="bg-forge-card text-forge-text text-xs rounded px-2 py-1 border border-forge-border">
            {[2,4,8,16,32].map(b => <option key={b} value={b}>{b}</option>)}
          </select>
        </div>
        <div className="flex items-center gap-2 ml-2">
          <span className="text-xs text-forge-muted">Vel:</span>
          <input type="range" min={1} max={127} value={velocity} onChange={e => setVelocity(Number(e.target.value))} className="w-20 accent-forge-accent" />
          <span className="text-xs font-mono text-forge-accent w-6">{velocity}</span>
        </div>
        <div className="flex-1" />
        <button onClick={handlePrPlay}
          className={clsx('flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold border transition-colors',
            prPlaying ? 'bg-forge-accent border-forge-accent text-white' : 'border-forge-border text-forge-muted hover:text-forge-accent hover:border-forge-accent'
          )}>
          {prPlaying ? <><Pause size={12} /> Stop</> : <><Play size={12} fill="currentColor" /> Play</>}
        </button>
        <button onClick={() => { if (confirm('Clear all notes?')) setLocalNotes([]) }} className="text-xs text-forge-muted hover:text-red-400 transition-colors">Clear All</button>
        <button onClick={() => { onChange(localNotes); onClose() }} className="btn-primary py-1.5 px-4 text-sm flex items-center gap-2"><Check size={14} /> Save</button>
        <button onClick={onClose} className="p-1.5 text-forge-muted hover:text-forge-text"><X size={18} /></button>
      </div>
      <div className="flex-1 overflow-auto bg-forge-black">
        <canvas ref={canvasRef}
          style={{ display: 'block', imageRendering: 'pixelated', cursor: prTool === 'erase' ? 'crosshair' : 'default' }}
          onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={() => setDragging(null)}
          onContextMenu={e => {
            e.preventDefault()
            const { cx: cx2, cy: cy2 } = getPos(e)
            setLocalNotes(prev => prev.filter(n => {
              const row = PIANO_ROLL_NOTES - 1 - (n.note - PIANO_ROLL_START_NOTE)
              const x = KEY_W + n.startBeat * BEAT_W; const w = n.durationBeats * BEAT_W; const y = row * ROW_H
              return !(cy2 >= y && cy2 < y + ROW_H && cx2 >= x && cx2 < x + w)
            }))
          }}
        />
      </div>
      <div className="px-4 py-2 bg-forge-dark border-t border-forge-border text-xs text-forge-muted">
        Click piano key: preview note • Left-click grid: draw • Right-click/Erase: delete • Drag note: move • Drag right edge: resize
      </div>
    </div>
  )
}

// ─── Device Picker ────────────────────────────────────────────────────────────
function DevicePicker({ current, onSelect, onClose }: { current: string; onSelect: (id: string) => void; onClose: () => void }) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [selected, setSelected] = useState(current)

  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(() => navigator.mediaDevices.enumerateDevices())
      .then(devs => { const a = devs.filter(d => d.kind === 'audioinput'); setDevices(a); if (!current && a[0]) setSelected(a[0].deviceId) })
      .catch(() => toast.error('Microphone permission denied'))
  }, [])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="bg-forge-card border border-forge-border rounded-2xl p-6 w-full max-w-sm shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-display text-white flex items-center gap-2"><Mic size={16} className="text-forge-accent" /> Select Mic Input</h3>
          <button onClick={onClose}><X size={18} className="text-forge-muted" /></button>
        </div>
        <div className="space-y-2 mb-4">
          {devices.map(d => (
            <button key={d.deviceId} onClick={() => setSelected(d.deviceId)}
              className={clsx('w-full text-left px-4 py-3 rounded-xl border text-sm transition-all', selected === d.deviceId ? 'border-forge-accent bg-forge-accent/10 text-forge-accent' : 'border-forge-border text-forge-text hover:border-forge-text')}>
              {d.label || `Microphone ${d.deviceId.slice(0,8)}`}
            </button>
          ))}
          {devices.length === 0 && <p className="text-forge-muted text-sm text-center py-4">No microphones found</p>}
        </div>
        <button onClick={() => { onSelect(selected); onClose() }} disabled={!selected} className="btn-primary w-full py-2 disabled:opacity-40">Use This Mic</button>
      </div>
    </div>
  )
}

// ─── Keybinds Overlay ─────────────────────────────────────────────────────────
function KeybindsOverlay({ onClose }: { onClose: () => void }) {
  const binds = [
    ['Space', 'Play / Pause'],
    ['R', 'Record'],
    ['Escape', 'Stop'],
    ['1 / Q', 'Pointer tool'],
    ['2 / W', 'Cut tool'],
    ['3 / E', 'Select tool'],
    ['4 / A', 'Fade tool'],
    ['5 / S', 'Gain tool'],
    ['Ctrl + S', 'Save project'],
    ['Ctrl + E', 'Export WAV'],
    ['Ctrl + Z', 'Undo'],
  ]
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={onClose}>
      <div className="bg-forge-card border border-forge-border rounded-2xl p-6 w-full max-w-md shadow-xl" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-5">
          <h3 className="font-display text-white flex items-center gap-2"><Keyboard size={16} className="text-forge-accent" /> Keyboard Shortcuts</h3>
          <button onClick={onClose}><X size={18} className="text-forge-muted" /></button>
        </div>
        <div className="space-y-2">
          {binds.map(([key, desc]) => (
            <div key={key} className="flex items-center justify-between py-1.5 border-b border-forge-border/40">
              <span className="text-forge-muted text-sm">{desc}</span>
              <kbd className="bg-forge-dark border border-forge-border rounded px-2 py-0.5 text-xs font-mono text-forge-accent">{key}</kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Parametric EQ Component ──────────────────────────────────────────────────
const EQ_COLORS = ['#ffd700', '#ff8800', '#00ff88', '#00d4ff', '#ff00aa']

function ParametricEQ({ bands, onChange }: { bands: EQBand[]; onChange: (bands: EQBand[]) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const svgRef    = useRef<SVGSVGElement>(null)
  const [dragging, setDragging] = useState<number | null>(null)
  const dragStartRef = useRef<{ x: number; y: number; origFreq: number; origGain: number } | null>(null)
  const WIDTH = 280, HEIGHT = 100

  // Log-scale helpers
  const freqToX = (f: number) => WIDTH * Math.log10(f / 20) / Math.log10(20000 / 20)
  const xToFreq = (x: number) => 20 * Math.pow(10, x / WIDTH * Math.log10(20000 / 20))
  const gainToY = (g: number) => HEIGHT / 2 - (g / 12) * (HEIGHT / 2 - 4)
  const yToGain = (y: number) => -((y - HEIGHT / 2) / (HEIGHT / 2 - 4)) * 12

  // Draw frequency response on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, WIDTH, HEIGHT)

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 1
    // Freq grid lines at 100, 1k, 10k
    for (const f of [100, 1000, 10000]) {
      const x = freqToX(f)
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, HEIGHT); ctx.stroke()
    }
    // 0dB center line
    ctx.strokeStyle = 'rgba(255,255,255,0.12)'
    ctx.beginPath(); ctx.moveTo(0, HEIGHT/2); ctx.lineTo(WIDTH, HEIGHT/2); ctx.stroke()

    // Compute combined response
    const SAMPLES = WIDTH
    const response = new Float32Array(SAMPLES)
    const freqArr = new Float32Array(SAMPLES)
    for (let i = 0; i < SAMPLES; i++) {
      freqArr[i] = xToFreq(i)
      response[i] = 0
    }

    // For each band, approximate its contribution
    for (const band of bands) {
      for (let i = 0; i < SAMPLES; i++) {
        const f = freqArr[i]
        let mag = 0
        if (band.type === 'peaking') {
          const bw = band.Q
          // Simple bell curve approximation in dB domain
          const semitones = Math.abs(Math.log2(f / band.freq) * 12)
          const rolloff = Math.exp(-semitones * semitones / (2 * bw * bw * 2))
          mag = band.gain * rolloff
        } else if (band.type === 'lowshelf') {
          const rolloff = 1 / (1 + Math.pow(f / band.freq, 2))
          mag = band.gain * rolloff
        } else if (band.type === 'highshelf') {
          const rolloff = 1 / (1 + Math.pow(band.freq / f, 2))
          mag = band.gain * rolloff
        } else if (band.type === 'highpass') {
          const atten = -24 * Math.log10(Math.max(f / band.freq, 0.001))
          mag = f < band.freq ? Math.max(atten, -36) : 0
        } else if (band.type === 'lowpass') {
          const atten = -24 * Math.log10(Math.max(band.freq / f, 0.001))
          mag = f > band.freq ? Math.max(atten, -36) : 0
        }
        response[i] += mag
      }
    }

    // Draw curve
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.9)'
    ctx.lineWidth = 1.5
    ctx.shadowColor = 'rgba(0, 212, 255, 0.4)'
    ctx.shadowBlur = 3
    ctx.beginPath()
    for (let i = 0; i < SAMPLES; i++) {
      const x = i
      const y = gainToY(Math.max(-15, Math.min(15, response[i])))
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    }
    ctx.stroke()
    ctx.shadowBlur = 0

    // Fill below curve
    ctx.fillStyle = 'rgba(0, 212, 255, 0.05)'
    ctx.beginPath()
    for (let i = 0; i < SAMPLES; i++) {
      const x = i
      const y = gainToY(Math.max(-15, Math.min(15, response[i])))
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
    }
    ctx.lineTo(WIDTH, HEIGHT/2); ctx.lineTo(0, HEIGHT/2); ctx.closePath(); ctx.fill()

  }, [bands])

  const handleMouseDown = (e: React.MouseEvent<SVGElement>, idx: number) => {
    e.preventDefault()
    const rect = svgRef.current!.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width * WIDTH
    const y = (e.clientY - rect.top)  / rect.height * HEIGHT
    dragStartRef.current = { x, y, origFreq: bands[idx].freq, origGain: bands[idx].gain }
    setDragging(idx)
  }

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (dragging === null || !dragStartRef.current) return
    const rect = svgRef.current!.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width * WIDTH
    const y = (e.clientY - rect.top)  / rect.height * HEIGHT
    const dx = x - dragStartRef.current.x
    const dy = y - dragStartRef.current.y
    const newFreq = Math.max(20, Math.min(20000, dragStartRef.current.origFreq * Math.pow(10, dx / WIDTH * Math.log10(20000/20))))
    const newGain = Math.max(-12, Math.min(12, dragStartRef.current.origGain - (dy / (HEIGHT/2 - 4)) * 12))
    const updated = bands.map((b, i) => i === dragging ? {
      ...b,
      freq: Math.round(newFreq),
      gain: ['highpass','lowpass'].includes(b.type) ? b.gain : parseFloat(newGain.toFixed(1))
    } : b)
    onChange(updated)
  }

  const handleMouseUp = () => setDragging(null)

  const handleWheel = (e: React.WheelEvent<SVGElement>, idx: number) => {
    e.preventDefault()
    const updated = bands.map((b, i) => i === idx ? {
      ...b,
      Q: Math.max(0.1, Math.min(10, b.Q + (e.deltaY > 0 ? -0.1 : 0.1)))
    } : b)
    onChange(updated)
  }

  return (
    <div className="space-y-2">
      {/* Canvas for frequency response */}
      <div className="relative" style={{ height: HEIGHT }}>
        <canvas ref={canvasRef} width={WIDTH} height={HEIGHT} className="absolute inset-0 w-full h-full rounded-lg" />
        {/* SVG overlay for draggable nodes */}
        <svg
          ref={svgRef}
          className="absolute inset-0 w-full h-full cursor-crosshair"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          {bands.map((band, idx) => {
            const x = freqToX(band.freq)
            const y = ['highpass','lowpass'].includes(band.type) ? HEIGHT/2 : gainToY(band.gain)
            const color = EQ_COLORS[idx % EQ_COLORS.length]
            return (
              <g key={band.id} onMouseDown={e => handleMouseDown(e, idx)} onWheel={e => handleWheel(e, idx)}>
                <circle cx={x} cy={y} r={8} fill={color} fillOpacity={0.85} stroke="white" strokeWidth={1.5}
                  style={{ cursor: 'grab', filter: dragging === idx ? 'brightness(1.3)' : 'none' }} />
                <text x={x} y={y + 4} textAnchor="middle" fontSize={8} fill="black" fontWeight="bold" pointerEvents="none">
                  {idx + 1}
                </text>
              </g>
            )
          })}
        </svg>
      </div>

      {/* Freq/gain labels for each band */}
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${bands.length}, 1fr)` }}>
        {bands.map((band, idx) => {
          const color = EQ_COLORS[idx % EQ_COLORS.length]
          const freqLabel = band.freq >= 1000 ? (band.freq/1000).toFixed(1)+'k' : band.freq+'Hz'
          return (
            <div key={band.id} className="text-center">
              <div className="text-[9px] font-mono" style={{ color }}>{freqLabel}</div>
              {!['highpass','lowpass'].includes(band.type) && (
                <div className="text-[9px] text-forge-muted">{band.gain > 0 ? '+' : ''}{band.gain.toFixed(1)}</div>
              )}
              {/* Band type selector */}
              <select
                value={band.type}
                onChange={e => onChange(bands.map((b, i) => i === idx ? { ...b, type: e.target.value as EQBand['type'] } : b))}
                className="w-full text-[8px] bg-forge-black border border-forge-border rounded text-forge-muted mt-0.5 py-0.5"
                style={{ fontSize: '8px' }}
              >
                <option value="highpass">HP</option>
                <option value="lowshelf">LS</option>
                <option value="peaking">PK</option>
                <option value="highshelf">HS</option>
                <option value="lowpass">LP</option>
              </select>
            </div>
          )
        })}
      </div>

      {/* Freq/gain hint text */}
      <p className="text-[9px] text-forge-border text-center">Drag nodes · Scroll wheel = Q</p>
    </div>
  )
}

// ─── Effects Panel ────────────────────────────────────────────────────────────
function EffectsPanel({ track, onChange, presets, onSavePreset, onLoadPreset, onDeletePreset, onClose, isPro }: {
  track: Track; onChange: (fx: TrackEffects) => void
  presets: Preset[]; onSavePreset: (name: string, fx: TrackEffects) => void; onLoadPreset: (fx: TrackEffects) => void
  onDeletePreset: (id: string) => void; onClose: () => void; isPro: boolean
}) {
  const [activeTab, setActiveTab] = useState<'fx'|'presets'>('fx')
  const [presetName, setPresetName] = useState('')
  const fx = track.effects
  const set = (k: keyof TrackEffects, v: number | string | boolean) => onChange({ ...fx, [k]: v })

  const quickPresets = [
    { name: 'Vocal', fx: VOCAL_PRESET },
    { name: 'Trap',  fx: TRAP_PRESET  },
    { name: 'Lo-Fi', fx: LOFI_PRESET  },
  ]

  return (
    <div className="h-full flex flex-col bg-forge-dark">
      <div className="px-3 py-2.5 border-b border-forge-border flex items-center gap-2 flex-shrink-0 bg-forge-black/40">
        <div className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: track.color }} />
        <span className="text-sm font-semibold text-forge-text truncate flex-1">{track.name}</span>
        <Sliders size={13} className="text-forge-muted" />
        <button onClick={onClose} className="p-1 text-forge-muted hover:text-forge-text transition-colors" title="Close FX panel"><X size={13} /></button>
      </div>

      <div className="flex gap-1 p-2 border-b border-forge-border bg-forge-black/20">
        <button onClick={() => setActiveTab('fx')} className={clsx('flex-1 py-1.5 rounded-lg text-xs font-semibold transition-colors', activeTab==='fx' ? 'bg-forge-accent text-white' : 'text-forge-muted hover:text-forge-text hover:bg-forge-card')}>FX Chain</button>
        <button onClick={() => setActiveTab('presets')} className={clsx('flex-1 py-1.5 rounded-lg text-xs font-semibold transition-colors', activeTab==='presets' ? 'bg-forge-accent text-white' : 'text-forge-muted hover:text-forge-text hover:bg-forge-card')}>Presets</button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-5">
        {activeTab === 'fx' ? (
          <>
            <div>
              <p className="text-[9px] text-forge-muted uppercase tracking-widest mb-2">Quick Presets</p>
              <div className="flex gap-1.5 flex-wrap">
                {quickPresets.map(p => (
                  <button key={p.name} onClick={() => onChange(p.fx)}
                    className="text-[10px] px-2.5 py-1 rounded-lg border border-forge-border text-forge-muted hover:text-forge-accent hover:border-forge-accent transition-colors font-semibold uppercase tracking-widest">
                    {p.name}
                  </button>
                ))}
                <button onClick={() => onChange(DEFAULT_FX)}
                  className="text-[10px] px-2.5 py-1 rounded-lg border border-forge-border/50 text-forge-border hover:text-forge-muted hover:border-forge-border transition-colors uppercase tracking-widest">
                  Reset
                </button>
              </div>
            </div>

            {/* EQ — parametric visual */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 transition-opacity", !fx.eqEnabled && "opacity-50")}>
              <div className="flex items-center gap-1.5 mb-3">
                <button onClick={() => set('eqEnabled', !fx.eqEnabled)}
                  className={clsx('w-4 h-4 rounded-full border-2 transition-colors flex-shrink-0',
                    fx.eqEnabled ? 'bg-forge-gold border-forge-gold' : 'border-forge-border bg-transparent')}>
                </button>
                <p className="text-[10px] font-bold text-forge-gold uppercase tracking-widest">EQ</p>
              </div>
              {fx.eqEnabled && (
                <ParametricEQ bands={fx.eqBands} onChange={v => onChange({ ...fx, eqBands: v })} />
              )}
            </div>

            {/* Space */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 transition-opacity", !(fx.reverbEnabled || fx.delayEnabled) && "opacity-50")}>
              <div className="flex items-center gap-1.5 mb-3">
                <button onClick={() => { onChange({ ...fx, reverbEnabled: !fx.reverbEnabled, delayEnabled: !fx.reverbEnabled }) }}
                  className={clsx('w-4 h-4 rounded-full border-2 transition-colors flex-shrink-0',
                    (fx.reverbEnabled || fx.delayEnabled) ? 'bg-blue-400 border-blue-400' : 'border-forge-border bg-transparent')}>
                </button>
                <p className="text-[10px] font-bold text-blue-400 uppercase tracking-widest">Space</p>
              </div>
              <div className="flex justify-around">
                <Knob label="Reverb"   value={fx.reverbWet}    min={0} max={1}    step={0.01} color="#60a5fa"
                  format={v => Math.round(v*100)+'%'} onChange={v => set('reverbWet',v)} />
                <Knob label="Delay"    value={fx.delayTime}    min={0} max={0.75} step={0.01} color="#60a5fa"
                  format={v => Math.round(v*1000)+'ms'} onChange={v => set('delayTime',v)} />
                <Knob label="Feedback" value={fx.delayFeedback} min={0} max={0.8}  step={0.01} color="#60a5fa"
                  format={v => Math.round(v*100)+'%'} onChange={v => set('delayFeedback',v)} />
              </div>
            </div>

            {/* Compressor */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 transition-opacity", !fx.compEnabled && "opacity-50")}>
              <div className="flex items-center gap-1.5 mb-3">
                <button onClick={() => set('compEnabled', !fx.compEnabled)}
                  className={clsx('w-4 h-4 rounded-full border-2 transition-colors flex-shrink-0',
                    fx.compEnabled ? 'bg-green-400 border-green-400' : 'border-forge-border bg-transparent')}>
                </button>
                <p className="text-[10px] font-bold text-green-400 uppercase tracking-widest">Compressor</p>
              </div>
              <div className="flex justify-around">
                <Knob label="Threshold" value={fx.compThreshold} min={-60} max={0}  step={1}   color="#4ade80"
                  format={v => v+'dB'} onChange={v => set('compThreshold',v)} />
                <Knob label="Ratio"     value={fx.compRatio}     min={1}   max={20} step={0.5} color="#4ade80"
                  format={v => v.toFixed(1)+':1'} onChange={v => set('compRatio',v)} />
              </div>
            </div>

            {/* Autotune */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 space-y-3 transition-opacity", !fx.autotuneEnabled && "opacity-50")}>
              <div className="flex items-center gap-1.5">
                <button onClick={() => set('autotuneEnabled', !fx.autotuneEnabled)}
                  className={clsx('w-4 h-4 rounded-full border-2 transition-colors flex-shrink-0',
                    fx.autotuneEnabled ? 'bg-forge-accent border-forge-accent' : 'border-forge-border bg-transparent')}>
                </button>
                <p className="text-[10px] font-bold text-forge-accent uppercase tracking-widest">Autotune</p>
              </div>
              <div>
                <span className="text-[10px] text-forge-muted uppercase tracking-widest block mb-1.5">Key</span>
                <div className="flex flex-wrap gap-1">
                  {['none', ...NOTE_NAMES].map(k => (
                    <button key={k} onClick={() => set('autotuneKey', k)}
                      className={clsx('px-2 py-0.5 rounded text-[10px] font-bold border transition-colors', fx.autotuneKey === k ? 'bg-forge-accent border-forge-accent text-white' : 'border-forge-border text-forge-muted hover:border-forge-text')}>
                      {k === 'none' ? 'Off' : k}
                    </button>
                  ))}
                </div>
              </div>
              {fx.autotuneKey !== 'none' && (
                <>
                  <div className="flex gap-1">
                    {(['major','minor','pentatonic'] as const).map(s => (
                      <button key={s} onClick={() => set('autotuneScale', s)}
                        className={clsx('flex-1 py-1 rounded text-[10px] font-bold border transition-colors capitalize', fx.autotuneScale===s ? 'bg-forge-accent border-forge-accent text-white' : 'border-forge-border text-forge-muted hover:border-forge-text')}>
                        {s}
                      </button>
                    ))}
                  </div>
                  <div className="flex justify-center">
                    <Knob label="Strength" value={fx.autotuneStrength} min={0} max={1} step={0.01}
                      format={v => Math.round(v*100)+'%'} onChange={v => set('autotuneStrength',v)} />
                  </div>
                </>
              )}
              <div className="flex justify-center">
                <Knob label="Pitch" value={fx.pitchShift} min={-12} max={12} step={1}
                  format={v => (v>0?'+':'')+v+' st'} onChange={v => set('pitchShift',v)} />
              </div>
            </div>

            {/* Distortion */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 space-y-2 transition-opacity", !fx.distortionEnabled && "opacity-50")}>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-forge-muted font-medium">Distortion</span>
                  <button onClick={() => onChange({ ...fx, distortionEnabled: !fx.distortionEnabled })}
                    className={`text-xs px-2 py-0.5 rounded ${fx.distortionEnabled ? 'bg-forge-orange/20 text-forge-orange' : 'bg-forge-black text-forge-muted'}`}>
                    {fx.distortionEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>
                {fx.distortionEnabled && (
                  <input type="range" min={0} max={1} step={0.01} value={fx.distortion}
                    onChange={e => onChange({ ...fx, distortion: parseFloat(e.target.value) })}
                    className="w-full accent-forge-orange" />
                )}
              </div>
            </div>

            {/* Chorus */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 space-y-2 transition-opacity", !fx.chorusEnabled && "opacity-50")}>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-forge-muted font-medium">Chorus</span>
                  <button onClick={() => onChange({ ...fx, chorusEnabled: !fx.chorusEnabled })}
                    className={`text-xs px-2 py-0.5 rounded ${fx.chorusEnabled ? 'bg-forge-orange/20 text-forge-orange' : 'bg-forge-black text-forge-muted'}`}>
                    {fx.chorusEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>
                {fx.chorusEnabled && (<>
                  <div className="flex justify-between text-xs text-forge-muted"><span>Rate</span><span>{fx.chorusRate?.toFixed(1)} Hz</span></div>
                  <input type="range" min={0.1} max={8} step={0.1} value={fx.chorusRate ?? 1.5}
                    onChange={e => onChange({ ...fx, chorusRate: parseFloat(e.target.value) })} className="w-full accent-forge-orange" />
                  <div className="flex justify-between text-xs text-forge-muted"><span>Depth</span><span>{fx.chorusDepth?.toFixed(0)} ms</span></div>
                  <input type="range" min={0} max={30} step={1} value={fx.chorusDepth ?? 5}
                    onChange={e => onChange({ ...fx, chorusDepth: parseFloat(e.target.value) })} className="w-full accent-forge-orange" />
                </>)}
              </div>
            </div>

            {/* Tremolo */}
            <div className={clsx("bg-forge-black/30 rounded-xl p-3 space-y-2 transition-opacity", !fx.tremoloEnabled && "opacity-50")}>
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-forge-muted font-medium">Tremolo</span>
                  <button onClick={() => onChange({ ...fx, tremoloEnabled: !fx.tremoloEnabled })}
                    className={`text-xs px-2 py-0.5 rounded ${fx.tremoloEnabled ? 'bg-forge-orange/20 text-forge-orange' : 'bg-forge-black text-forge-muted'}`}>
                    {fx.tremoloEnabled ? 'ON' : 'OFF'}
                  </button>
                </div>
                {fx.tremoloEnabled && (<>
                  <div className="flex justify-between text-xs text-forge-muted"><span>Rate</span><span>{fx.tremoloRate?.toFixed(1)} Hz</span></div>
                  <input type="range" min={0.1} max={20} step={0.1} value={fx.tremoloRate ?? 4}
                    onChange={e => onChange({ ...fx, tremoloRate: parseFloat(e.target.value) })} className="w-full accent-forge-orange" />
                  <div className="flex justify-between text-xs text-forge-muted"><span>Depth</span><span>{Math.round((fx.tremoloDepth ?? 0) * 100)}%</span></div>
                  <input type="range" min={0} max={1} step={0.01} value={fx.tremoloDepth ?? 0}
                    onChange={e => onChange({ ...fx, tremoloDepth: parseFloat(e.target.value) })} className="w-full accent-forge-orange" />
                </>)}
              </div>
            </div>
          </>
        ) : (
          <div className="relative">
            <div className="space-y-3">
              <div className="flex gap-2">
                <input value={presetName} onChange={e => setPresetName(e.target.value)}
                  placeholder={isPro ? 'Preset name...' : 'Pro required to save presets'} className="input-forge text-sm flex-1 py-1.5"
                  disabled={!isPro}
                  onKeyDown={e => { if (isPro && e.key === 'Enter' && presetName.trim()) { onSavePreset(presetName.trim(), fx); setPresetName('') } }} />
                <button onClick={() => { if (isPro && presetName.trim()) { onSavePreset(presetName.trim(), fx); setPresetName('') } }}
                  disabled={!isPro || !presetName.trim()} className="btn-primary px-3 py-1.5 text-sm disabled:opacity-40">
                  <Save size={14} />
                </button>
              </div>
              {presets.length === 0 && <p className="text-xs text-forge-muted text-center py-6 bg-forge-black/20 rounded-xl">No saved presets yet<br/><span className="opacity-60">{isPro ? 'Name an FX setting and save it' : 'Upgrade to Pro to save presets'}</span></p>}
              {presets.map(p => (
                <div key={p.id} className="flex items-center gap-2 p-3 rounded-xl bg-forge-black/40 border border-forge-border hover:border-forge-accent/40 transition-colors">
                  <span className="text-sm text-forge-text flex-1 font-medium">{p.name}</span>
                  <button onClick={() => onLoadPreset(p.effects)} className="text-xs font-semibold text-forge-accent hover:text-forge-accent/80 px-2 py-1 rounded-lg bg-forge-accent/10 hover:bg-forge-accent/20 transition-colors">Load</button>
                  <button
                    onClick={async () => {
                      try {
                        const res = await presetsApi.share(p.id)
                        const shareUrl = `${window.location.origin}/presets/share/${res.data.token}`
                        navigator.clipboard?.writeText(shareUrl)
                        toast.success('Share link copied!')
                      } catch {
                        toast.error('Failed to share preset')
                      }
                    }}
                    className="p-1 text-forge-muted hover:text-forge-cyan transition-colors"
                    title="Share preset"
                  >
                    <Share2 size={12} />
                  </button>
                  <button onClick={() => onDeletePreset(p.id)} className="p-1 text-forge-muted hover:text-red-400 transition-colors" title="Delete preset"><X size={12} /></button>
                </div>
              ))}
            </div>
            {!isPro && (
              <div className="absolute inset-0 bg-forge-black/80 backdrop-blur-sm rounded-xl flex flex-col items-center justify-center z-10">
                <Lock size={24} className="text-forge-muted mb-2" />
                <p className="text-forge-text font-medium text-sm">Pro Only</p>
                <p className="text-forge-muted text-xs text-center mt-1 px-4">Presets are available for Pro subscribers</p>
                <Link href="/pricing" className="mt-3 btn-primary text-xs px-3 py-1.5">Upgrade to Pro</Link>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Live Recording Waveform ──────────────────────────────────────────────────
// Shows a real-time growing waveform during recording, positioned at the recording clip
function LiveRecordingWave({
  analyser, recStartPct
}: { analyser: AnalyserNode | null; recStartPct: number }) {
  const canvasRef  = useRef<HTMLCanvasElement>(null)
  const rafRef     = useRef<number>(0)
  const samplesRef = useRef<number[]>([])

  useEffect(() => {
    if (!analyser || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx2d  = canvas.getContext('2d')!
    const bufLen = analyser.frequencyBinCount
    const dataArr = new Uint8Array(bufLen)
    samplesRef.current = []

    const drawLoop = () => {
      rafRef.current = requestAnimationFrame(drawLoop)
      analyser.getByteTimeDomainData(dataArr)
      // Compute RMS amplitude for this frame
      let sum = 0
      for (let i = 0; i < bufLen; i++) { const v = (dataArr[i] - 128) / 128; sum += v * v }
      samplesRef.current.push(Math.sqrt(sum / bufLen))

      const W = canvas.width, H = canvas.height
      ctx2d.clearRect(0, 0, W, H)
      const samples = samplesRef.current
      if (samples.length < 2) return
      // Bar width narrows as more data comes in (fixed canvas = growing history)
      const barW = Math.max(1.5, W / samples.length)
      ctx2d.fillStyle = '#ff3c0099'
      for (let i = 0; i < samples.length; i++) {
        const x  = (i / samples.length) * W
        const bh = Math.max(2, samples[i] * H * 2.5)
        ctx2d.fillRect(x, (H - bh) / 2, barW - 0.5, bh)
      }
      // Blinking record dot
      if (Math.floor(Date.now() / 500) % 2 === 0) {
        ctx2d.fillStyle = '#ff0000cc'
        ctx2d.beginPath(); ctx2d.arc(W - 8, 8, 5, 0, Math.PI * 2); ctx2d.fill()
      }
    }
    drawLoop()
    return () => cancelAnimationFrame(rafRef.current)
  }, [analyser])

  if (!analyser) return null
  return (
    <div
      className="absolute top-0 bottom-0 pointer-events-none z-10"
      style={{ left: `${recStartPct}%`, right: 0 }}
    >
      <canvas ref={canvasRef} width={1200} height={88} className="w-full h-full" />
    </div>
  )
}

// ─── Main Studio Page ─────────────────────────────────────────────────────────
export default function StudioPage() {
  const { beatId } = useParams<{ beatId: string }>()
  const router = useRouter()
  const searchParams = useSearchParams()
  const { user, isLoading } = useAuthStore()

  const [beat, setBeat]                 = useState<any>(null)
  const [pageLoading, setPageLoading]   = useState(true)
  const [tracks, setTracks]             = useState<Track[]>([])
  const [selectedTrack, setSelectedTrack] = useState<string | null>(null)
  const [playing, setPlaying]           = useState(false)
  const [recording, setRecording]       = useState(false)
  const [playhead, setPlayhead]         = useState(0)
  const [tool, setTool]                 = useState<Tool>('pointer')
  const [bpm, setBpm]                   = useState(120)
  const [showPianoRoll, setShowPianoRoll]       = useState<string | null>(null)
  const [showDevicePicker, setShowDevicePicker] = useState(false)
  const [showKeybinds, setShowKeybinds]         = useState(false)
  const [micDeviceId, setMicDeviceId]           = useState<string>('')
  const [presets, setPresets]           = useState<Preset[]>([])
  const [exporting, setExporting]       = useState(false)
  const [isMobile, setIsMobile]         = useState(false)
  const [metronomeEnabled, setMetronomeEnabled] = useState(false)
  const [liveAnalyser, setLiveAnalyser] = useState<AnalyserNode | null>(null)
  const [monitorEnabled, setMonitorEnabled] = useState(false)
  const [recStartPct, setRecStartPct] = useState(0)
  const [zoom, setZoom] = useState(1)
  const [showFxPanel, setShowFxPanel] = useState(true)
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; trackId: string; clipId: string; trackType: string } | null>(null)
  const [selectedClip, setSelectedClip] = useState<{ trackId: string; clipId: string } | null>(null)
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [saveBeatTitle, setSaveBeatTitle] = useState('')
  const [savingBeat, setSavingBeat] = useState(false)
  const [inviteModal, setInviteModal] = useState(false)
  const [inviteUsername, setInviteUsername] = useState('')
  const [inviting, setInviting] = useState(false)
  const [collaborators, setCollaborators] = useState<any[]>([])
  const [collabsLoading, setCollabsLoading] = useState(false)
  const [kickingCollaborator, setKickingCollaborator] = useState<string | null>(null)

  // Audio refs
  const audioCtxRef        = useRef<AudioContext | null>(null)
  const trackGraphs        = useRef<Map<string, TrackGraph>>(new Map())
  const audioBufferCache   = useRef<Map<string, AudioBuffer>>(new Map())
  const activeSourcesRef   = useRef<AudioBufferSourceNode[]>([])
  const playStartTimeRef   = useRef<number>(0)
  const playStartOffsetRef = useRef<number>(0)
  const playheadInterval   = useRef<ReturnType<typeof setInterval> | null>(null)
  const impulseRef         = useRef<AudioBuffer | null>(null)

  // Recording refs
  const mediaRecorderRef   = useRef<MediaRecorder | null>(null)
  const recordedChunks     = useRef<Blob[]>([])
  const recordingTrackId   = useRef<string | null>(null)
  const recordingAnalyser  = useRef<AnalyserNode | null>(null)
  const monitorNodeRef     = useRef<AudioNode | null>(null)
  const recordingStartOffset = useRef<number>(0)   // timeline time when record started

  // WaveSurfer
  const wsInstances        = useRef<Map<string, any>>(new Map())
  const waveContainers     = useRef<Map<string, HTMLDivElement>>(new Map())
  const audioFileDurations = useRef<Map<string, number>>(new Map())
  const [, setDurationsVersion] = useState(0) // bumped when a new audio duration is known

  // Clip drag/resize
  const clipInteraction = useRef<{
    type: 'move' | 'resizeRight' | 'resizeLeft' | 'gain'
    trackId: string; clipId: string
    startX: number; startY: number
    origStartTime: number; origDuration: number; origGain: number
  } | null>(null)

  // Fade handle drag
  const fadeDrag = useRef<{ type: 'in' | 'out'; trackId: string; clipId: string; startX: number; origFade: number } | null>(null)

  // Undo stack — stores serialized track states (no blobs)
  const undoStack = useRef<Track[][]>([])

  // Metronome cleanup
  const metronomeCleanupRef = useRef<(() => void) | null>(null)

  const initialTrackCountRef = useRef<number>(-1)

  // Timeline scroll ref (for zoom)
  const timelineScrollRef = useRef<HTMLDivElement>(null)

  // ── Mobile check ────────────────────────────────────────────────────────────
  useEffect(() => {
    setIsMobile(window.innerWidth < 1024)
    const h = () => setIsMobile(window.innerWidth < 1024)
    window.addEventListener('resize', h); return () => window.removeEventListener('resize', h)
  }, [])

  // ── Load beat + saved project ────────────────────────────────────────────────
  useEffect(() => {
    if (!isLoading && !user) { router.push('/login?redirect=/studio/'+beatId); return }
    if (!beatId || !user) return
    beatsApi.get(beatId).then(res => {
      const b = res.data
      // Block studio access while beat is still generating
      if (b.status === 'generating') {
        toast.error('This beat is still being generated. Please wait until it\'s ready.')
        router.push('/dashboard')
        return
      }
      setBeat(b); setBpm(b.bpm || 120)
      const audioUrl = b.wav_url || b.mp3_url || b.preview_url

      const versionId = searchParams?.get('version')
      if (versionId) {
        // Load specific version — skip localStorage, restore from server version snapshot
        studioApi.getVersion(beatId, versionId).then(vRes => {
          const pd = vRes.data?.project_data
          if (pd) {
            const vTracks = (pd.tracks || []).map((t: any) => ({ ...t, audioBlob: undefined, armed: t.armed ?? false }))
            if (initialTrackCountRef.current === -1) initialTrackCountRef.current = vTracks.length
            setTracks(vTracks)
            setBpm(pd.bpm || b.bpm || 120)
            setSelectedTrack(vTracks[0]?.id || null)
            toast.success(`Loaded version v${vRes.data.version_number}`)
          }
        }).catch(() => toast.error('Failed to load version')).finally(() => setPageLoading(false))
        return
      }

      const saved = localStorage.getItem(`studio_project_${beatId}`)
      if (saved) {
        try {
          const proj = JSON.parse(saved)
          const savedTracks = proj.tracks.map((t: any) => ({ ...t, audioBlob: undefined, armed: t.armed ?? false }))
          if (initialTrackCountRef.current === -1) initialTrackCountRef.current = savedTracks.length
          setTracks(savedTracks)
          setBpm(proj.bpm || b.bpm || 120)
          setSelectedTrack(proj.tracks[0]?.id || null)
          toast.success('Project restored')
          return
        } catch {}
      }

      // Load server-saved studio project (e.g. when a buyer opens from library)
      const serverProject = b.studio_project
        ? (typeof b.studio_project === 'string' ? JSON.parse(b.studio_project) : b.studio_project)
        : null

      // Handle both { tracks: [...] } format and bare [] format
      const serverTracks = serverProject?.tracks ?? (Array.isArray(serverProject) ? serverProject : null)
      if (serverTracks && serverTracks.length > 0) {
        const loadedTracks = serverTracks.map((t: any) => ({ ...t, audioBlob: undefined, armed: t.armed ?? false }))
        if (initialTrackCountRef.current === -1) initialTrackCountRef.current = loadedTracks.length
        setTracks(loadedTracks)
        setBpm(serverProject?.bpm || b.bpm || 120)
        setSelectedTrack(loadedTracks[0]?.id || null)
        setPageLoading(false)
        return
      }

      const dur = b.duration_seconds || 120
      const initialTracks: Track[] = []

      // Parse stems — MySQL may return it as already-parsed object or still as string
      const rawStems = b.stems
      const stems: Record<string, string> | null = !rawStems ? null
        : typeof rawStems === 'string' ? (() => { try { return JSON.parse(rawStems) } catch { return null } })()
        : rawStems

      // Stem name → studio instrument type
      const STEM_INSTRUMENTS: Record<string, InstrumentType> = {
        // Rhythm stems
        drums: 'drums', bass: 'bass', percs: 'drums', percussion: 'drums',
        // Melodic stems
        keyboard: 'piano', woodwinds: 'piano', vocals: 'pad',
        melody: 'piano', piano: 'piano', epiano: 'epiano', rhodes: 'epiano',
        synth: 'synth', 'lead synth': 'synth', guitar: 'synth', trumpet: 'synth',
        'acoustic guitar': 'piano', harp: 'piano', flute: 'piano', bells: 'piano',
        marimba: 'piano', clarinet: 'piano', banjo: 'piano', ukulele: 'piano',
        organ: 'epiano', sax: 'epiano', saxophone: 'epiano', vibraphone: 'epiano',
        strings: 'pad', violin: 'pad', cello: 'pad', choir: 'pad', pad: 'pad',
        brass: 'synth', horn: 'synth', other: 'synth',
      }

      // If beat has stems (separate audio files), create one track per stem
      if (stems && Object.keys(stems).length > 0) {
        // Order stems: drums first, then bass, then melodic instruments
        const STEM_ORDER = ['drums', 'percs', 'percussion', 'bass', 'keyboard', 'melody', 'piano', 'synth', 'guitar', 'strings', 'woodwinds', 'brass', 'pad', 'choir', 'vocals']
        const stemEntries = Object.entries(stems).sort(([a], [b]) => {
          const ai = STEM_ORDER.indexOf(a), bi = STEM_ORDER.indexOf(b)
          return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi)
        })
        stemEntries.forEach(([stemName, stemUrl], i) => {
          if (!stemUrl) return
          const label = stemName.charAt(0).toUpperCase() + stemName.slice(1)
          initialTracks.push({
            id: uid(), type: 'audio', name: label,
            color: TRACK_COLORS[i % TRACK_COLORS.length],
            volume: 0.8, pan: 0, muted: false, solo: false, armed: false,
            instrument: STEM_INSTRUMENTS[stemName] || 'piano',
            audioUrl: stemUrl,
            clips: [{ id: uid(), startTime: 0, duration: dur, offset: 0, clipGain: 1 }],
            midiNotes: [], effects: DEFAULT_FX,
          })
        })
      } else {
        // No stems available — show combined audio as one locked beat track
        const beatTrack: Track = {
          id: uid(), type: 'beat', name: b.title || 'Beat', color: TRACK_COLORS[0],
          volume: 0.8, pan: 0, muted: false, solo: false, armed: false, instrument: 'piano',
          audioUrl, clips: audioUrl ? [{ id: uid(), startTime: 0, duration: dur, offset: 0, clipGain: 1 }] : [],
          midiNotes: [], effects: DEFAULT_FX,
        }
        initialTracks.push(beatTrack)
      }

      // If beat has MIDI tracks (from a MIDI generation), import them
      const MIDI_INSTS: InstrumentType[] = ['piano', 'bass', 'synth', 'pad', 'epiano']
      if (Array.isArray(b.midi_tracks) && b.midi_tracks.length > 0) {
        b.midi_tracks.forEach((mt: any, i: number) => {
          initialTracks.push({
            id: uid(), type: 'midi',
            name: mt.name || `MIDI ${i + 1} — ${(mt.instrument || MIDI_INSTS[i % MIDI_INSTS.length])}`,
            color: TRACK_COLORS[(initialTracks.length + i) % TRACK_COLORS.length],
            volume: 0.9, pan: 0, muted: false, solo: false, armed: false,
            instrument: (mt.instrument as InstrumentType) || MIDI_INSTS[i % MIDI_INSTS.length],
            clips: [{ id: uid(), startTime: 0, duration: mt.total_beats || 16, offset: 0, clipGain: 1 }],
            midiNotes: mt.notes || [],
            effects: DEFAULT_FX,
          })
        })
      }

      if (initialTrackCountRef.current === -1) initialTrackCountRef.current = initialTracks.length
      setTracks(initialTracks); setSelectedTrack(initialTracks[0]?.id || null)
    }).catch(() => toast.error('Failed to load beat')).finally(() => setPageLoading(false))

    const saved = localStorage.getItem('studio_presets')
    if (saved) { try { setPresets(JSON.parse(saved)) } catch {} }
  }, [beatId, user, isLoading])

  // ── Audio context helpers ────────────────────────────────────────────────────
  const getAudioCtx = useCallback(() => {
    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext()
    if (audioCtxRef.current.state === 'suspended') audioCtxRef.current.resume()
    return audioCtxRef.current
  }, [])

  const getImpulse = useCallback(() => {
    const ctx = getAudioCtx()
    if (!impulseRef.current) impulseRef.current = createImpulse(ctx)
    return impulseRef.current
  }, [getAudioCtx])

  // ── Init or get persistent track graph ──────────────────────────────────────
  const initTrackGraph = useCallback((track: Track): TrackGraph => {
    const ctx = getAudioCtx()
    const impulse = getImpulse()
    let graph = trackGraphs.current.get(track.id)
    if (!graph) {
      graph = buildTrackGraph(ctx, track.effects, track.muted ? 0 : track.volume, track.pan, impulse)
      graph.output.connect(ctx.destination)
      trackGraphs.current.set(track.id, graph)
    }
    return graph
  }, [getAudioCtx, getImpulse])

  // ── Update all graphs when tracks change (live effect updates) ───────────────
  useEffect(() => {
    if (!playing) return
    tracks.forEach(track => {
      const graph = trackGraphs.current.get(track.id)
      if (graph) updateTrackGraph(graph, track)
    })
  }, [tracks, playing])

  // ── WaveSurfer: keyed by clip.id so each clip (incl. cut clips) gets its own wave ─
  const initWaveSurfer = useCallback(async (clipId: string, container: HTMLDivElement, audioUrl: string, color: string) => {
    if (wsInstances.current.has(clipId)) {
      wsInstances.current.get(clipId)?.destroy()
      wsInstances.current.delete(clipId)
    }
    const WaveSurfer = (await import('wavesurfer.js')).default
    const ws = WaveSurfer.create({
      container, waveColor: color + '55', progressColor: color + 'aa',
      height: 54, barWidth: 2, barGap: 1, barRadius: 2,
      normalize: true, interact: false,
    })
    ws.on('ready', () => {
      const dur = ws.getDuration()
      if (dur > 0 && !audioFileDurations.current.has(audioUrl)) {
        audioFileDurations.current.set(audioUrl, dur)
        setDurationsVersion(v => v + 1)
      }
    })
    ws.load(audioUrl)
    wsInstances.current.set(clipId, ws)
  }, [])

  // Watch for new clips / audio URL changes (e.g. after recording or cut)
  useEffect(() => {
    const allClipIds = new Set(tracks.flatMap(t => t.clips.map(c => c.id)))
    tracks.forEach(track => {
      if (track.type === 'midi' || !track.audioUrl) return
      track.clips.forEach(clip => {
        const container = waveContainers.current.get(clip.id)
        if (!container) return
        if (!wsInstances.current.has(clip.id)) {
          initWaveSurfer(clip.id, container, track.audioUrl!, track.color)
        }
      })
    })
    // Destroy WaveSurfer instances for clips that no longer exist
    wsInstances.current.forEach((ws, id) => {
      if (!allClipIds.has(id)) { ws.destroy(); wsInstances.current.delete(id) }
    })
    waveContainers.current.forEach((_, id) => {
      if (!allClipIds.has(id)) { waveContainers.current.delete(id) }
    })
    // Destroy audio graphs for removed tracks
    trackGraphs.current.forEach((_, id) => {
      if (!tracks.find(t => t.id === id)) {
        const g = trackGraphs.current.get(id)
        if (g) { try { g.output.disconnect() } catch {} }
        trackGraphs.current.delete(id)
      }
    })
  }, [tracks, initWaveSurfer])

  // ── Load audio buffer (cached) ───────────────────────────────────────────────
  const loadBuffer = useCallback(async (url: string): Promise<AudioBuffer | null> => {
    const ctx = getAudioCtx()
    if (audioBufferCache.current.has(url)) return audioBufferCache.current.get(url)!
    try {
      const res = await fetch(url); const ab = await res.arrayBuffer()
      const buf = await ctx.decodeAudioData(ab)
      audioBufferCache.current.set(url, buf); return buf
    } catch { return null }
  }, [getAudioCtx])

  // ── Undo ─────────────────────────────────────────────────────────────────────
  const pushUndo = useCallback((currentTracks: Track[]) => {
    const snapshot = currentTracks.map(({ audioBlob, ...t }) => ({ ...t })) as Track[]
    undoStack.current = [...undoStack.current.slice(-29), snapshot]
  }, [])

  const undo = useCallback(() => {
    if (undoStack.current.length === 0) { toast('Nothing to undo'); return }
    const prev = undoStack.current[undoStack.current.length - 1]
    undoStack.current = undoStack.current.slice(0, -1)
    setTracks(prev)
    toast('Undo')
  }, [])

  // ── Metronome ─────────────────────────────────────────────────────────────────
  const startMetronome = useCallback((ctx: AudioContext, startOffset: number, ctxStartTime: number, bpmValue: number) => {
    if (metronomeCleanupRef.current) { metronomeCleanupRef.current(); metronomeCleanupRef.current = null }
    let stopped = false
    const secPerBeat = 60 / bpmValue
    // Figure out which beat we're on
    let beatIndex = Math.floor(startOffset / secPerBeat)
    let scheduledUpTo = ctxStartTime

    const schedule = () => {
      if (stopped) return
      const scheduleAhead = 0.2 // seconds ahead to schedule
      const now = ctx.currentTime
      while (scheduledUpTo < now + scheduleAhead) {
        const beatTime = ctxStartTime + (beatIndex * secPerBeat - startOffset)
        if (beatTime >= now - 0.01) {
          const osc = ctx.createOscillator()
          const gain = ctx.createGain()
          osc.connect(gain); gain.connect(ctx.destination)
          const isBar = beatIndex % 4 === 0
          osc.frequency.value = isBar ? 1000 : 600
          gain.gain.setValueAtTime(0.3, beatTime)
          gain.gain.exponentialRampToValueAtTime(0.001, beatTime + 0.05)
          osc.start(beatTime); osc.stop(beatTime + 0.06)
        }
        beatIndex++
        scheduledUpTo = ctxStartTime + (beatIndex * secPerBeat - startOffset)
      }
      if (!stopped) setTimeout(schedule, 50)
    }
    schedule()
    metronomeCleanupRef.current = () => { stopped = true }
  }, [])

  // ── Playback ─────────────────────────────────────────────────────────────────
  const handlePlay = useCallback(async () => {
    const ctx     = getAudioCtx()
    const hasSolo = tracks.some(t => t.solo)
    const offset  = playhead

    activeSourcesRef.current.forEach(src => { try { src.stop() } catch {} })
    activeSourcesRef.current = []

    await Promise.all(tracks.map(async track => {
      const shouldPlay = hasSolo ? track.solo : !track.muted
      if (!shouldPlay) return

      const graph = initTrackGraph(track)
      // Sync graph with current track state
      updateTrackGraph(graph, track)

      if (track.type === 'midi') {
        const secPerBeat = 60 / bpm
        for (const note of track.midiNotes) {
          const noteStart = note.startBeat * secPerBeat
          if (noteStart < offset) continue
          const duration  = note.durationBeats * secPerBeat
          const startAt   = ctx.currentTime + (noteStart - offset)
          synthesizeMidiNote(ctx, note.note, note.velocity, duration, track.instrument, graph.input, startAt)
        }
        return
      }

      if (!track.audioUrl) return
      const buf = await loadBuffer(track.audioUrl)
      if (!buf) return

      for (const clip of track.clips) {
        const clipEnd = clip.startTime + clip.duration
        if (clipEnd <= offset) continue

        const src = ctx.createBufferSource()
        src.buffer = buf
        // Apply pitch shift + autotune as detune (cents)
        {
          const fx = track.effects
          if (fx.autotuneEnabled && fx.autotuneKey !== 'none' && fx.pitchShift !== undefined) {
            src.detune.value = (fx.pitchShift || 0) * 100  // pitchShift in semitones → cents
          }
          let detuneValue = track.effects.pitchShift * 100
          if (track.effects.autotuneEnabled && track.effects.autotuneKey !== 'none' && track.effects.autotuneStrength > 0) {
            const hz = detectPitch(buf)
            if (hz) detuneValue += calcAutotuneDetune(hz, track.effects.autotuneKey, track.effects.autotuneScale, track.effects.autotuneStrength)
          }
          if (detuneValue !== 0) src.detune.value = detuneValue
        }

        // Apply clip gain via GainNode
        const clipGainNode = ctx.createGain()
        clipGainNode.gain.value = clip.clipGain ?? 1
        src.connect(clipGainNode); clipGainNode.connect(graph.input)

        const bufOffset    = Math.max(0, offset - clip.startTime) + clip.offset
        const startAt      = Math.max(0, ctx.currentTime + clip.startTime - offset)
        const playDuration = clip.duration - Math.max(0, offset - clip.startTime)

        // Loop stems when the clip is longer than the audio (e.g. 32s stem looped over 120s beat)
        const shouldLoop = clip.duration > buf.duration + 0.5
        if (shouldLoop) {
          src.loop      = true
          src.loopStart = 0
          src.loopEnd   = buf.duration
        }

        // Fade in/out automation
        if ((clip.fadeIn && clip.fadeIn > 0) || (clip.fadeOut && clip.fadeOut > 0)) {
          const fadeNode = ctx.createGain()
          clipGainNode.disconnect(); src.connect(fadeNode); fadeNode.connect(graph.input)
          const t0 = startAt
          if (clip.fadeIn && clip.fadeIn > 0) {
            const fadeDur = Math.min(clip.fadeIn, playDuration)
            fadeNode.gain.setValueAtTime(0, t0)
            fadeNode.gain.linearRampToValueAtTime(1, t0 + fadeDur)
          }
          if (clip.fadeOut && clip.fadeOut > 0) {
            const fStart = t0 + playDuration - clip.fadeOut
            if (fStart > t0) {
              fadeNode.gain.setValueAtTime(1, fStart)
              fadeNode.gain.linearRampToValueAtTime(0, t0 + playDuration)
            }
          }
        }

        src.start(startAt, bufOffset % buf.duration, shouldLoop ? undefined : Math.max(0.01, playDuration))
        if (shouldLoop) {
          // Stop the looping source after playDuration
          src.stop(startAt + Math.max(0.01, playDuration))
        }
        activeSourcesRef.current.push(src)
      }
    }))

    playStartTimeRef.current   = ctx.currentTime
    playStartOffsetRef.current = offset
    setPlaying(true)

    if (playheadInterval.current) clearInterval(playheadInterval.current)
    playheadInterval.current = setInterval(() => {
      const elapsed = audioCtxRef.current ? audioCtxRef.current.currentTime - playStartTimeRef.current : 0
      setPlayhead(playStartOffsetRef.current + elapsed)
    }, 50)

    if (metronomeEnabled) {
      startMetronome(ctx, offset, ctx.currentTime, bpm)
    }
  }, [tracks, playhead, bpm, getAudioCtx, initTrackGraph, loadBuffer, metronomeEnabled, startMetronome])

  const handlePause = useCallback(() => {
    setPlaying(false)
    if (playheadInterval.current) clearInterval(playheadInterval.current)
    activeSourcesRef.current.forEach(src => { try { src.stop() } catch {} })
    activeSourcesRef.current = []
    if (metronomeCleanupRef.current) { metronomeCleanupRef.current(); metronomeCleanupRef.current = null }
  }, [])

  const handleStop = useCallback(() => {
    handlePause(); setPlayhead(0)
  }, [handlePause])

  // ── Seek ──────────────────────────────────────────────────────────────────────
  const handleRulerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = (e.clientX - rect.left) / rect.width
    const newTime = ratio * TIMELINE_DURATION
    const wasPlaying = playing
    if (wasPlaying) handlePause()
    setPlayhead(newTime)
    if (wasPlaying) setTimeout(() => handlePlay(), 50)
  }

  // ── Recording ─────────────────────────────────────────────────────────────────
  const handleRecord = useCallback(async () => {
    if (recording) {
      mediaRecorderRef.current?.stop()
      // Disconnect monitoring
      if (monitorNodeRef.current) {
        try { monitorNodeRef.current.disconnect() } catch {}
        monitorNodeRef.current = null
      }
      setRecording(false)
      setLiveAnalyser(null)
      recordingAnalyser.current = null
      return
    }
    try {
      const armedTrack = tracks.find(t => t.armed)

      // Clean audio constraints — disable browser DSP for studio quality
      const audioConstraints: MediaTrackConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl:  false,
        sampleRate:       48000,
        channelCount:     1,
        ...(micDeviceId ? { deviceId: { exact: micDeviceId } } : {}),
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints })
      const ctx    = getAudioCtx()

      // Analyser for live waveform — lightweight, no processing
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 1024
      analyser.smoothingTimeConstant = 0
      const micSrc = ctx.createMediaStreamSource(stream)
      micSrc.connect(analyser)
      recordingAnalyser.current = analyser
      setLiveAnalyser(analyser)

      // Monitoring — route through the armed track's full effects chain (EQ, comp, reverb, delay)
      if (monitorEnabled) {
        const armedTrk = tracks.find(t => t.armed)
        const monFx  = armedTrk?.effects ?? VOCAL_PRESET
        const monVol = armedTrk?.volume  ?? 0.8
        const monPan = armedTrk?.pan     ?? 0
        const impulse = impulseRef.current || createImpulse(ctx)
        const monGraph = buildTrackGraph(ctx, monFx, monVol, monPan, impulse)
        monGraph.output.connect(ctx.destination)
        micSrc.connect(monGraph.input)
        monitorNodeRef.current = monGraph.output
      }

      let targetTrackId: string
      if (armedTrack) {
        targetTrackId = armedTrack.id
      } else {
        targetTrackId = uid()
        const vocalTrack: Track = {
          id: targetTrackId, type: 'vocal',
          name: 'Vocal ' + (tracks.filter(t => t.type === 'vocal').length + 1),
          color: TRACK_COLORS[tracks.length % TRACK_COLORS.length],
          volume: 0.9, pan: 0, muted: false, solo: false, armed: true, instrument: 'piano',
          clips: [], midiNotes: [], effects: VOCAL_PRESET,
        }
        setTracks(prev => [...prev, vocalTrack])
        setSelectedTrack(targetTrackId)
      }
      recordingTrackId.current = targetTrackId

      // Snapshot the timeline position when recording starts
      const recStart = playhead
      recordingStartOffset.current = recStart
      setRecStartPct((recStart / TIMELINE_DURATION) * 100)

      // Use Opus codec for better quality; fall back gracefully
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/webm')
          ? 'audio/webm'
          : ''
      const mr = new MediaRecorder(stream, {
        ...(mimeType ? { mimeType } : {}),
        audioBitsPerSecond: 128000,
      })
      mediaRecorderRef.current = mr
      recordedChunks.current = []

      mr.ondataavailable = e => { if (e.data.size > 0) recordedChunks.current.push(e.data) }
      mr.onstop = () => {
        stream.getTracks().forEach(t => t.stop())
        const blobType = mimeType || 'audio/webm'
        const blob = new Blob(recordedChunks.current, { type: blobType })
        const url  = URL.createObjectURL(blob)

        // Duration = how much time passed in the AudioContext since recording started
        const ctxElapsed = audioCtxRef.current
          ? audioCtxRef.current.currentTime - playStartTimeRef.current
          : 0
        const recEnd = recordingStartOffset.current + ctxElapsed
        const dur    = Math.max(0.5, recEnd - recordingStartOffset.current)
        const recStartSnap = recordingStartOffset.current

        setTracks(prev => prev.map(t => {
          if (t.id !== recordingTrackId.current) return t
          // Build new clips list:
          // 1. Keep clips fully before recStart (unchanged)
          // 2. Trim clips that straddle recStart (cut at recStart)
          // 3. Delete clips fully inside [recStart, recEnd]
          // 4. Trim clips that straddle recEnd (cut at recEnd, keep right part)
          // 5. Add the new recording clip
          const newClips: Clip[] = []
          for (const c of t.clips) {
            const cEnd = c.startTime + c.duration
            if (cEnd <= recStartSnap) {
              newClips.push(c)  // entirely before — keep as-is
            } else if (c.startTime >= recEnd) {
              newClips.push(c)  // entirely after — keep as-is
            } else if (c.startTime < recStartSnap && cEnd > recStartSnap) {
              // Straddles start — trim to end at recStart
              newClips.push({ ...c, duration: recStartSnap - c.startTime })
            }
            // Clips fully inside [recStart, recEnd] are deleted (replaced by new recording)
            // Clips straddling recEnd: keep right part (trim from recEnd)
            if (c.startTime < recEnd && cEnd > recEnd) {
              const trimAmt = recEnd - c.startTime
              newClips.push({
                ...c, id: uid(),
                startTime: recEnd,
                duration: cEnd - recEnd,
                offset: c.offset + trimAmt,
              })
            }
          }
          // Add the new recording clip
          newClips.push({ id: uid(), startTime: recStartSnap, duration: dur, offset: 0, clipGain: 1 })

          return { ...t, audioUrl: url, audioBlob: blob, armed: false, clips: newClips }
        }))
        audioBufferCache.current.delete(url)
        toast.success('Recording saved!')
      }

      mr.start(100) // collect data every 100ms for smoother waveform
      setRecording(true)
      if (!playing) handlePlay()
    } catch (err: any) {
      toast.error('Microphone error: ' + (err.message || 'Permission denied'))
    }
  }, [recording, tracks, micDeviceId, monitorEnabled, playing, playhead, getAudioCtx, handlePlay])

  // ── Track limit ────────────────────────────────────────────────────────────────
  const FREE_EXTRA_TRACKS = 1  // free users can add 1 track beyond initial beat tracks
  const isPro = user?.subscription_plan === 'pro' && user?.subscription_status === 'active'
  const canAddTrack = isPro || (initialTrackCountRef.current === -1 ? true : tracks.length < initialTrackCountRef.current + FREE_EXTRA_TRACKS)

  // ── Import audio ──────────────────────────────────────────────────────────────
  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]; if (!file) return
    if (!canAddTrack) {
      toast.error(`Free plan allows 1 extra track. Upgrade to Pro for unlimited.`)
      e.target.value = ''
      return
    }
    const url = URL.createObjectURL(file)
    const newId = uid()
    const track: Track = {
      id: newId, type: 'audio', name: file.name.replace(/\.[^.]+$/, ''),
      color: TRACK_COLORS[tracks.length % TRACK_COLORS.length],
      volume: 0.8, pan: 0, muted: false, solo: false, armed: false, instrument: 'piano',
      audioUrl: url, audioBlob: file,
      clips: [{ id: uid(), startTime: 0, duration: 0, offset: 0, clipGain: 1 }],
      midiNotes: [], effects: DEFAULT_FX,
    }
    const audio = new Audio(url)
    audio.onloadedmetadata = () => {
      setTracks(prev => prev.map(t => t.id === newId ? { ...t, clips: [{ id: uid(), startTime: 0, duration: audio.duration, offset: 0, clipGain: 1 }] } : t))
    }
    setTracks(prev => [...prev, track]); setSelectedTrack(newId)
    e.target.value = ''
    toast.success('Imported: ' + track.name)
  }

  // ── Add Track helpers ─────────────────────────────────────────────────────────
  const addAudioTrack = () => {
    if (!canAddTrack) { toast.error(`Free plan allows 1 extra track. Upgrade to Pro for unlimited.`); return }
    const inputEl = document.getElementById('global-import-input') as HTMLInputElement | null
    if (inputEl) inputEl.click()
  }

  const MIDI_INSTRUMENTS: InstrumentType[] = ['piano', 'bass', 'synth', 'pad', 'epiano']
  const addMidiTrack = () => {
    if (!canAddTrack) { toast.error(`Free plan allows 1 extra track. Upgrade to Pro for unlimited.`); return }
    const midiCount = tracks.filter(t => t.type === 'midi').length
    const instrument = MIDI_INSTRUMENTS[midiCount % MIDI_INSTRUMENTS.length]
    const track: Track = {
      id: uid(), type: 'midi', name: `MIDI ${midiCount + 1} — ${instrument.charAt(0).toUpperCase() + instrument.slice(1)}`,
      color: TRACK_COLORS[(tracks.length) % TRACK_COLORS.length],
      volume: 0.9, pan: 0, muted: false, solo: false, armed: false, instrument,
      clips: [{ id: uid(), startTime: 0, duration: 16, offset: 0, clipGain: 1 }],
      midiNotes: [], effects: DEFAULT_FX,
    }
    setTracks(prev => [...prev, track]); setSelectedTrack(track.id); setShowPianoRoll(track.id)
  }

  const addVocalTrack = () => {
    if (!canAddTrack) { toast.error(`Free plan allows 1 extra track. Upgrade to Pro for unlimited.`); return }
    const track: Track = {
      id: uid(), type: 'vocal', name: 'Vocal ' + (tracks.filter(t=>t.type==='vocal').length + 1),
      color: TRACK_COLORS[tracks.length % TRACK_COLORS.length],
      volume: 0.9, pan: 0, muted: false, solo: false, armed: true, instrument: 'piano',
      clips: [], midiNotes: [], effects: VOCAL_PRESET,
    }
    setTracks(prev => [...prev, track]); setSelectedTrack(track.id)
    toast('Vocal track added. Click R to arm and record.')
  }

  const createMidiRegion = (trackId: string) => {
    const clipStart = playhead
    setTracks(prev => prev.map(t => t.id !== trackId ? t : {
      ...t,
      clips: [...t.clips, { id: uid(), startTime: clipStart, duration: 16, offset: 0, clipGain: 1 }]
    }))
    toast('MIDI region created — open Piano Roll to add notes')
  }

  // ── Timeline tools ────────────────────────────────────────────────────────────
  const handleTimelineClick = (e: React.MouseEvent<HTMLDivElement>, trackId: string) => {
    if (tool !== 'cut') return
    const rect = e.currentTarget.getBoundingClientRect()
    const track = tracks.find(t => t.id === trackId); if (!track || !track.clips[0]) return
    const clip = track.clips[0]
    const cutTime = ((e.clientX - rect.left) / rect.width) * TIMELINE_DURATION
    if (cutTime <= clip.startTime || cutTime >= clip.startTime + clip.duration) return
    pushUndo(tracks)
    const relCut = cutTime - clip.startTime
    const newClip: Clip = { id: uid(), startTime: cutTime, duration: clip.duration - relCut, offset: clip.offset + relCut, clipGain: clip.clipGain }
    setTracks(prev => prev.map(t => t.id === trackId ? { ...t, clips: [...t.clips.filter(c => c.id !== clip.id), { ...clip, duration: relCut }, newClip] } : t))
    toast.success('Clip cut')
  }

  const handleClipClick = (_e: React.MouseEvent<HTMLDivElement>, _trackId: string, _clip: Clip) => {
    // Fade now handled by draggable handles; selection handled inline
  }

  // Clip mouse down: pointer = move/resize, gain = drag up/down
  const handleClipMouseDown = (
    e: React.MouseEvent, trackId: string, clip: Clip,
    resizeType: 'none' | 'resizeRight' | 'resizeLeft'
  ) => {
    e.stopPropagation(); e.preventDefault()
    if (tool === 'gain') {
      clipInteraction.current = {
        type: 'gain', trackId, clipId: clip.id,
        startX: e.clientX, startY: e.clientY,
        origStartTime: 0, origDuration: 0, origGain: clip.clipGain ?? 1,
      }
      return
    }
    if (tool !== 'pointer') return
    const type = resizeType === 'none' ? 'move' : resizeType
    clipInteraction.current = {
      type: type as 'move' | 'resizeRight' | 'resizeLeft',
      trackId, clipId: clip.id,
      startX: e.clientX, startY: e.clientY,
      origStartTime: clip.startTime, origDuration: clip.duration, origGain: 0,
    }
  }

  const handleTimelineMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (fadeDrag.current) {
      const { type, trackId, clipId, startX, origFade } = fadeDrag.current
      const rect = e.currentTarget.getBoundingClientRect()
      const dt = ((e.clientX - startX) / rect.width) * TIMELINE_DURATION
      setTracks(prev => prev.map(t => t.id !== trackId ? t : {
        ...t, clips: t.clips.map(c => {
          if (c.id !== clipId) return c
          if (type === 'in') return { ...c, fadeIn: Math.max(0, Math.min(c.duration * 0.95, origFade + dt)) }
          return { ...c, fadeOut: Math.max(0, Math.min(c.duration * 0.95, origFade - dt)) }
        })
      }))
      return
    }
    if (!clipInteraction.current) return
    const { type, trackId, clipId, startX, startY, origStartTime, origDuration, origGain } = clipInteraction.current
    if (type === 'gain') {
      const dy = startY - e.clientY
      const newGain = Math.max(0.05, Math.min(4, origGain + dy * 0.015))
      setTracks(prev => prev.map(t => t.id !== trackId ? t : {
        ...t, clips: t.clips.map(c => c.id !== clipId ? c : { ...c, clipGain: newGain })
      }))
      return
    }
    const rect = e.currentTarget.getBoundingClientRect()
    const dx = e.clientX - startX
    const dt = (dx / rect.width) * TIMELINE_DURATION
    setTracks(prev => prev.map(t => t.id !== trackId ? t : {
      ...t,
      clips: t.clips.map(c => {
        if (c.id !== clipId) return c
        if (type === 'move') return { ...c, startTime: Math.max(0, origStartTime + dt) }
        if (type === 'resizeRight') return { ...c, duration: Math.max(0.1, origDuration + dt) }
        if (type === 'resizeLeft') {
          const newStart = Math.max(0, origStartTime + dt)
          const delta = newStart - origStartTime
          return { ...c, startTime: newStart, duration: Math.max(0.1, origDuration - delta), offset: Math.max(0, c.offset + delta) }
        }
        return c
      })
    }))
  }

  // Clip gain via scroll
  const handleClipWheel = (e: React.WheelEvent, trackId: string, clipId: string) => {
    if (tool !== 'gain') return
    e.preventDefault()
    const delta = e.deltaY < 0 ? 0.05 : -0.05
    setTracks(prev => prev.map(t => t.id !== trackId ? t : {
      ...t,
      clips: t.clips.map(c => c.id !== clipId ? c : { ...c, clipGain: Math.max(0.05, Math.min(4, (c.clipGain ?? 1) + delta)) })
    }))
  }

  // Timeline wheel: Ctrl+scroll = zoom, else pass through
  const handleTimelineWheel = (e: React.WheelEvent) => {
    if (!e.ctrlKey) return
    e.preventDefault()
    setZoom(prev => Math.max(1, Math.min(8, prev * (e.deltaY < 0 ? 1.15 : 0.87))))
  }

  // ── Invite collaborator ───────────────────────────────────────────────────────
  const loadCollaborators = async () => {
    setCollabsLoading(true)
    try {
      const res = await studioApi.collaborators(beatId)
      setCollaborators(res.data ?? [])
    } catch {
      // non-fatal
    } finally {
      setCollabsLoading(false)
    }
  }

  const handleInvite = async () => {
    if (!inviteUsername.trim()) return
    setInviting(true)
    try {
      await studioApi.invite(beatId, inviteUsername.trim())
      toast.success(`Invitation sent to @${inviteUsername}`)
      setInviteUsername('')
      loadCollaborators()
    } catch (err: any) {
      if (err.response?.data?.requiresPro) {
        toast.error('Studio collaboration requires Pro subscription')
      } else {
        toast.error(err.response?.data?.error || 'Failed to send invitation')
      }
    } finally {
      setInviting(false)
    }
  }

  const handleKickCollaborator = async (userId: string, username: string) => {
    setKickingCollaborator(userId)
    try {
      await studioApi.kickCollaborator(beatId, userId)
      toast.success(`@${username} removed from collaboration`)
      setCollaborators(prev => prev.filter(c => c.id !== userId))
    } catch (err: any) {
      toast.error(err.response?.data?.error || 'Failed to remove collaborator')
    } finally {
      setKickingCollaborator(null)
    }
  }

  // ── Save project ──────────────────────────────────────────────────────────────
  const saveProject = useCallback(async () => {
    const proj = { beatId, bpm, tracks: tracks.map(({ audioBlob, ...t }) => t), savedAt: new Date().toISOString() }
    localStorage.setItem(`studio_project_${beatId}`, JSON.stringify(proj))
    // Also persist to server as a named version (version history)
    try {
      const res = await studioApi.saveVersion(beatId, {
        project_data: proj,
        label: `Saved by ${user?.display_name || user?.username || 'user'}`,
      })
      toast.success(`Project saved — v${res.data.version_number}`)
    } catch {
      // Server save failed (e.g. no auth) — local save still worked
      toast.success('Project saved locally')
    }
  }, [beatId, bpm, tracks, user])

  // ── Export ────────────────────────────────────────────────────────────────────
  const handleExport = useCallback(async () => {
    if (exporting) return
    if (user?.subscription_plan !== 'pro') {
      toast.error('WAV export requires a Pro plan. Upgrade to unlock.')
      return
    }
    setExporting(true)
    toast.loading('Bouncing mix...', { id: 'export' })
    try {
      const totalDuration = Math.max(30, Math.max(...tracks.flatMap(t => t.clips.map(c => c.startTime + c.duration)), 30))
      const offCtx  = new OfflineAudioContext(2, Math.ceil(44100 * totalDuration), 44100)
      const impulse = createImpulse(offCtx)
      const hasSolo = tracks.some(t => t.solo)

      await Promise.all(tracks.filter(t => hasSolo ? t.solo : !t.muted).map(async (track) => {
        if (track.type === 'midi') {
          const secPerBeat = 60 / bpm
          const { input, output } = buildFxChainOffline(offCtx, track.effects, track.volume, track.pan, impulse)
          output.connect(offCtx.destination)
          for (const note of track.midiNotes) {
            synthesizeMidiNote(offCtx, note.note, note.velocity, note.durationBeats * secPerBeat, track.instrument, input, note.startBeat * secPerBeat)
          }
          return
        }
        if (!track.audioUrl) return
        const res = await fetch(track.audioUrl); const ab = await res.arrayBuffer()
        let audioBuf: AudioBuffer
        try { audioBuf = await offCtx.decodeAudioData(ab) } catch { return }

        for (const clip of track.clips) {
          const src = offCtx.createBufferSource()
          src.buffer = audioBuf
          {
            let detuneValue = track.effects.pitchShift * 100
            if (track.effects.autotuneEnabled && track.effects.autotuneKey !== 'none' && track.effects.autotuneStrength > 0) {
              const hz = detectPitch(audioBuf)
              if (hz) detuneValue += calcAutotuneDetune(hz, track.effects.autotuneKey, track.effects.autotuneScale, track.effects.autotuneStrength)
            }
            if (detuneValue !== 0) src.detune.value = detuneValue
          }

          const { input, output } = buildFxChainOffline(offCtx, track.effects, track.volume * (clip.clipGain ?? 1), track.pan, impulse)
          src.connect(input); output.connect(offCtx.destination)

          const shouldLoop = clip.duration > audioBuf.duration + 0.5
          if (shouldLoop) { src.loop = true; src.loopStart = 0; src.loopEnd = audioBuf.duration }

          if ((clip.fadeIn && clip.fadeIn > 0) || (clip.fadeOut && clip.fadeOut > 0)) {
            const fadeGain = offCtx.createGain()
            src.disconnect(); src.connect(fadeGain); fadeGain.connect(input)
            const t0 = clip.startTime
            if (clip.fadeIn && clip.fadeIn > 0)  { fadeGain.gain.setValueAtTime(0, t0); fadeGain.gain.linearRampToValueAtTime(1, t0 + clip.fadeIn) }
            if (clip.fadeOut && clip.fadeOut > 0) { const fEnd = t0 + clip.duration; fadeGain.gain.setValueAtTime(1, fEnd - clip.fadeOut); fadeGain.gain.linearRampToValueAtTime(0, fEnd) }
          }

          if (shouldLoop) {
            src.start(clip.startTime, clip.offset % audioBuf.duration)
            src.stop(clip.startTime + clip.duration)
          } else {
            src.start(clip.startTime, clip.offset, clip.duration)
          }
        }
      }))

      const rendered = await offCtx.startRendering()
      const wav = encodeWAV(rendered)
      const url = URL.createObjectURL(wav)
      const a   = document.createElement('a')
      a.href = url; a.download = `${beat?.title || 'studio-mix'}.wav`
      document.body.appendChild(a); a.click(); document.body.removeChild(a)
      URL.revokeObjectURL(url)
      toast.success('Mix exported!', { id: 'export' })
    } catch (err: any) {
      toast.error('Export failed: ' + err.message, { id: 'export' })
    } finally {
      setExporting(false)
    }
  }, [exporting, tracks, bpm, beat])

  // ── Helpers ───────────────────────────────────────────────────────────────────
  const updateTrack = useCallback((id: string, patch: Partial<Track>) => {
    setTracks(prev => prev.map(t => t.id === id ? { ...t, ...patch } : t))
  }, [])

  const deleteTrack = useCallback((id: string) => {
    if (tracks.find(t => t.id === id)?.type === 'beat') return toast.error('Cannot delete the beat track')
    pushUndo(tracks)
    setTracks(prev => prev.filter(t => t.id !== id))
    wsInstances.current.get(id)?.destroy(); wsInstances.current.delete(id)
    const g = trackGraphs.current.get(id)
    if (g) { try { g.output.disconnect() } catch {} }
    trackGraphs.current.delete(id)
    if (selectedTrack === id) setSelectedTrack(tracks[0]?.id || null)
  }, [tracks, selectedTrack, pushUndo])

  const deleteClip = useCallback((trackId: string, clipId: string) => {
    pushUndo(tracks)
    setTracks(prev => prev.map(t => t.id === trackId ? { ...t, clips: t.clips.filter(c => c.id !== clipId) } : t))
  }, [tracks, pushUndo])

  const savePreset = useCallback((name: string, effects: TrackEffects) => {
    if (user?.subscription_plan !== 'pro') {
      toast.error('Saving presets requires a Pro plan. Upgrade to unlock.')
      return
    }
    const updated = [...presets, { id: uid(), name, effects }]
    setPresets(updated); localStorage.setItem('studio_presets', JSON.stringify(updated))
    toast.success('Preset saved: ' + name)
  }, [presets, user])

  const deletePreset = useCallback((id: string) => {
    const updated = presets.filter(p => p.id !== id)
    setPresets(updated)
    localStorage.setItem('studio_presets', JSON.stringify(updated))
  }, [presets])

  // ── Save as Beat + Publish ────────────────────────────────────────────────────
  const getStudioProject = useCallback(() => {
    return tracks.map(({ audioBlob, ...t }) => t)
  }, [tracks])

  const handleSaveAsBeat = useCallback(async () => {
    if (!saveBeatTitle.trim()) return
    setSavingBeat(true)
    try {
      await beatsApi.create({
        title: saveBeatTitle.trim(),
        bpm,
        genre: beat?.genre || 'trap',
        key: beat?.musical_key || 'C',
        mood: beat?.mood || '',
        status: 'personal',
        wav_url: beat?.wav_url,
        mp3_url: beat?.mp3_url,
        preview_url: beat?.preview_url,
        waveform_data: beat?.waveform_data,
        duration_seconds: beat?.duration_seconds,
        studio_project: getStudioProject(),
      })
      toast.success('Saved to My Beats!')
      setShowSaveDialog(false)
      setSaveBeatTitle('')
    } catch {
      toast.error('Failed to save beat')
    } finally {
      setSavingBeat(false)
    }
  }, [saveBeatTitle, bpm, beat, getStudioProject])

  // Mix all tracks down to a WAV blob using OfflineAudioContext
  const mixdownToBlob = useCallback(async (): Promise<Blob> => {
    const totalDuration = Math.max(30, Math.max(...tracks.flatMap(t => t.clips.map(c => c.startTime + c.duration)), 30))
    const offCtx  = new OfflineAudioContext(2, Math.ceil(44100 * totalDuration), 44100)
    const impulse = createImpulse(offCtx)
    const hasSolo = tracks.some(t => t.solo)

    await Promise.all(tracks.filter(t => hasSolo ? t.solo : !t.muted).map(async (track) => {
      if (track.type === 'midi') {
        const secPerBeat = 60 / bpm
        const { input, output } = buildFxChainOffline(offCtx, track.effects, track.volume, track.pan, impulse)
        output.connect(offCtx.destination)
        for (const note of track.midiNotes) {
          synthesizeMidiNote(offCtx, note.note, note.velocity, note.durationBeats * secPerBeat, track.instrument, input, note.startBeat * secPerBeat)
        }
        return
      }
      if (!track.audioUrl) return
      const res = await fetch(track.audioUrl); const ab = await res.arrayBuffer()
      let audioBuf: AudioBuffer
      try { audioBuf = await offCtx.decodeAudioData(ab) } catch { return }
      for (const clip of track.clips) {
        const src = offCtx.createBufferSource()
        src.buffer = audioBuf
        {
          let detuneValue = track.effects.pitchShift * 100
          if (track.effects.autotuneEnabled && track.effects.autotuneKey !== 'none' && track.effects.autotuneStrength > 0) {
            const hz = detectPitch(audioBuf)
            if (hz) detuneValue += calcAutotuneDetune(hz, track.effects.autotuneKey, track.effects.autotuneScale, track.effects.autotuneStrength)
          }
          if (detuneValue !== 0) src.detune.value = detuneValue
        }
        const { input, output } = buildFxChainOffline(offCtx, track.effects, track.volume * (clip.clipGain ?? 1), track.pan, impulse)
        src.connect(input); output.connect(offCtx.destination)
        const shouldLoop = clip.duration > audioBuf.duration + 0.5
        if (shouldLoop) {
          src.loop = true; src.loopStart = 0; src.loopEnd = audioBuf.duration
          src.start(clip.startTime, clip.offset % audioBuf.duration)
          src.stop(clip.startTime + clip.duration)
        } else {
          src.start(clip.startTime, clip.offset, clip.duration)
        }
      }
    }))

    const rendered = await offCtx.startRendering()
    return encodeWAV(rendered)
  }, [tracks, bpm])

  const handlePublishBeat = useCallback(async () => {
    if (user?.subscription_plan !== 'pro') {
      toast.error('Pro plan required to publish beats')
      return
    }
    try {
      toast.loading('Mixing down tracks...', { id: 'publish' })

      // Mix all tracks → WAV → upload → update beat audio
      const wavBlob = await mixdownToBlob()
      const token = localStorage.getItem('token')
      await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/beats/${beatId}/audio`, {
        method: 'POST',
        headers: {
          'Content-Type': 'audio/wav',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: wavBlob,
      })

      await beatsApi.update(beatId, { studio_project: getStudioProject() })
      await beatsApi.publish(beatId)
      setBeat((prev: any) => ({ ...prev, status: 'published' }))
      toast.success('Beat published to marketplace!', { id: 'publish' })
    } catch (err: any) {
      toast.error('Failed to publish: ' + (err.message || 'unknown error'), { id: 'publish' })
    }
  }, [beatId, user, getStudioProject, mixdownToBlob, beat, bpm])

  // ── Keyboard shortcuts ────────────────────────────────────────────────────────
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA') return
      if (e.ctrlKey || e.metaKey) {
        if (e.key === 's') { e.preventDefault(); saveProject() }
        if (e.key === 'e') { e.preventDefault(); handleExport() }
        if (e.key === 'z') { e.preventDefault(); undo() }
        return
      }
      switch (e.code) {
        case 'Space':   e.preventDefault(); playing ? handlePause() : handlePlay(); break
        case 'KeyR':    handleRecord(); break
        case 'Escape':  handleStop(); break
        case 'Digit1': case 'KeyQ': setTool('pointer'); break
        case 'Digit2': case 'KeyW': setTool('cut'); break
        case 'Digit3': case 'KeyE': setTool('select'); break
        case 'Digit4': case 'KeyA': setTool('fade'); break
        case 'Digit5': case 'KeyS': setTool('gain'); break
        case 'Backspace':
          if (selectedClip) {
            e.preventDefault()
            deleteClip(selectedClip.trackId, selectedClip.clipId)
            setSelectedClip(null)
          }
          break
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [playing, handlePlay, handlePause, handleStop, handleRecord, saveProject, handleExport, undo])

  const selTrack = tracks.find(t => t.id === selectedTrack)

  // ── Mobile block ──────────────────────────────────────────────────────────────
  if (isMobile) return (
    <div className="min-h-screen bg-forge-black flex flex-col items-center justify-center px-6 text-center">
      <div className="w-20 h-20 rounded-2xl bg-forge-card border border-forge-border flex items-center justify-center mb-6">
        <Sliders size={36} className="text-forge-accent" />
      </div>
      <h1 className="font-display text-3xl text-white mb-3 tracking-wider">DESKTOP ONLY</h1>
      <p className="text-forge-muted max-w-xs mb-8">The studio requires a desktop browser with keyboard and mouse. Open this on your laptop or desktop.</p>
      <Link href="/dashboard?tab=beats" className="btn-secondary flex items-center gap-2"><ArrowLeft size={14} /> Back to Dashboard</Link>
    </div>
  )

  if (pageLoading || isLoading) return (
    <div className="min-h-screen bg-forge-black flex items-center justify-center">
      <Loader2 size={32} className="animate-spin text-forge-accent" />
    </div>
  )

  const toolCursor: Record<Tool, string> = {
    pointer: 'default', cut: 'crosshair', select: 'text', fade: 'col-resize', gain: 'ns-resize'
  }
  const toolIcons = [
    { id: 'pointer' as Tool, icon: <MousePointer2 size={14} />, label: 'Pointer (1)' },
    { id: 'cut'     as Tool, icon: <Scissors size={14} />,      label: 'Cut (2)' },
    { id: 'select'  as Tool, icon: <ZoomIn size={14} />,        label: 'Select (3)' },
    { id: 'fade'    as Tool, icon: <Wand2 size={14} />,         label: 'Fade (4)' },
    { id: 'gain'    as Tool, icon: <TrendingUp size={14} />,    label: 'Gain (5)' },
  ]

  return (
    <div
      className="min-h-screen bg-forge-black flex flex-col select-none"
      onMouseUp={() => { clipInteraction.current = null; fadeDrag.current = null }}
    >
      {/* Hidden import input */}
      <input id="global-import-input" type="file" accept="audio/*" className="hidden" onChange={handleImport} />

      {showPianoRoll && (() => {
        const t = tracks.find(tt => tt.id === showPianoRoll)!
        return (
          <PianoRoll
            notes={t?.midiNotes || []}
            instrument={t?.instrument || 'piano'}
            bpm={bpm}
            onChange={notes => updateTrack(showPianoRoll, { midiNotes: notes })}
            onInstrumentChange={inst => updateTrack(showPianoRoll, { instrument: inst })}
            onClose={() => setShowPianoRoll(null)}
          />
        )
      })()}
      {showDevicePicker && <DevicePicker current={micDeviceId} onSelect={setMicDeviceId} onClose={() => setShowDevicePicker(false)} />}
      {showKeybinds     && <KeybindsOverlay onClose={() => setShowKeybinds(false)} />}

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-4 bg-forge-dark border-b border-forge-border flex-shrink-0" style={{ paddingTop: 68, paddingBottom: 10 }}>
        <Link href="/dashboard?tab=beats" className="text-forge-muted hover:text-forge-text p-1"><ArrowLeft size={16} /></Link>
        <Music2 size={15} className="text-forge-accent" />
        <span className="font-display text-white text-sm truncate max-w-[200px]">{beat?.title || 'Studio'}</span>
        <span className="text-forge-muted text-xs">— STUDIO</span>

        {/* Transport */}
        <div className="flex items-center gap-1.5 ml-3 bg-forge-black rounded-xl px-3 py-1.5 border border-forge-border">
          <button onClick={handleStop} className="text-forge-muted hover:text-forge-text p-1" title="Stop (Esc)"><SkipBack size={13} /></button>
          <button onClick={playing ? handlePause : handlePlay}
            className="w-7 h-7 rounded-full bg-forge-accent flex items-center justify-center text-white hover:bg-forge-accent/80 transition-colors">
            {playing ? <Pause size={12} /> : <Play size={12} fill="white" />}
          </button>
          <button onClick={handleRecord}
            className={clsx('w-7 h-7 rounded-full flex items-center justify-center transition-colors', recording ? 'bg-red-500 animate-pulse text-white' : 'bg-forge-card text-red-400 hover:bg-red-500/20 border border-forge-border')}
            title="Record (R)">
            <Circle size={11} fill={recording ? 'white' : 'currentColor'} />
          </button>
          {/* Monitor toggle — hear yourself while recording (no effects to avoid latency) */}
          <button
            onClick={() => setMonitorEnabled(v => !v)}
            title={monitorEnabled ? 'Monitoring ON — click to mute yourself' : 'Monitoring OFF — click to hear yourself (no effects, low latency)'}
            className={clsx('flex items-center gap-1 px-2 py-1 rounded-lg border text-[9px] font-bold transition-colors',
              monitorEnabled ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-forge-black border-forge-border text-forge-muted hover:text-forge-text'
            )}>
            <Mic size={10} />MON
          </button>
          <span className="text-xs font-mono text-forge-accent w-20 ml-1">{fmt(playhead)}</span>
        </div>

        {/* BPM */}
        <div className="flex items-center gap-1 bg-forge-black rounded-lg px-3 py-1.5 border border-forge-border">
          <span className="text-[10px] text-forge-muted uppercase">BPM</span>
          <input type="number" value={bpm} onChange={e => setBpm(parseInt(e.target.value)||120)}
            className="w-10 bg-transparent text-forge-accent text-sm font-mono text-center outline-none" min={60} max={300} />
        </div>

        {/* Metronome */}
        <button
          onClick={() => setMetronomeEnabled(v => !v)}
          title="Toggle Metronome"
          className={clsx('flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-semibold transition-colors',
            metronomeEnabled ? 'bg-forge-accent/20 border-forge-accent text-forge-accent' : 'bg-forge-black border-forge-border text-forge-muted hover:text-forge-text'
          )}>
          <Timer size={12} />
          <span className="text-[10px]">Click</span>
        </button>

        {/* Tools */}
        <div className="flex items-center gap-0.5 bg-forge-black rounded-lg p-1 border border-forge-border">
          {toolIcons.map(({ id, icon, label }) => (
            <button key={id} onClick={() => setTool(id)} title={label}
              className={clsx('p-1.5 rounded transition-colors', tool===id ? 'bg-forge-accent/20 text-forge-accent' : 'text-forge-muted hover:text-forge-text')}>
              {icon}
            </button>
          ))}
        </div>

        <div className="flex-1" />

        {/* Zoom */}
        <div className="flex items-center gap-0.5 bg-forge-black rounded-lg p-1 border border-forge-border">
          <button onClick={() => setZoom(z => Math.max(1, z / 1.5))} className="p-1 text-forge-muted hover:text-forge-text transition-colors" title="Zoom out"><ZoomOut size={12} /></button>
          <span className="text-[10px] text-forge-muted font-mono w-8 text-center">{zoom.toFixed(1)}x</span>
          <button onClick={() => setZoom(z => Math.min(8, z * 1.5))} className="p-1 text-forge-muted hover:text-forge-text transition-colors" title="Zoom in (Ctrl+scroll)"><ZoomIn size={12} /></button>
        </div>

        {/* FX panel toggle */}
        <button onClick={() => setShowFxPanel(v => !v)}
          className={clsx('p-1.5 rounded transition-colors', showFxPanel ? 'text-forge-accent' : 'text-forge-muted hover:text-forge-text')}
          title="Toggle FX panel">
          <Sliders size={14} />
        </button>

        {/* Undo */}
        <button onClick={undo} className="p-1.5 text-forge-muted hover:text-forge-text transition-colors" title="Undo (Ctrl+Z)">
          <RotateCcw size={14} />
        </button>

        <button onClick={() => setShowKeybinds(true)} className="p-1.5 text-forge-muted hover:text-forge-text" title="Keyboard shortcuts"><Keyboard size={14} /></button>
        <button onClick={() => setShowDevicePicker(true)} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-forge-black border border-forge-border text-forge-muted hover:text-forge-text text-xs transition-colors" title="Select microphone">
          <Settings size={12} /> Mic {micDeviceId ? <span className="text-green-400">●</span> : ''}
        </button>
        <button onClick={saveProject} className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-forge-black border border-forge-border text-forge-muted hover:text-forge-text text-xs transition-colors" title="Save project (Ctrl+S)">
          <Save size={12} /> Save
        </button>
        <button onClick={handleExport} disabled={exporting}
          className={clsx(
            'py-1.5 px-4 text-sm flex items-center gap-2 disabled:opacity-50',
            user?.subscription_plan === 'pro' ? 'btn-primary' : 'flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-forge-black border border-forge-border text-forge-muted hover:text-forge-accent text-xs transition-colors'
          )}
          title={user?.subscription_plan === 'pro' ? 'Export WAV (Ctrl+E)' : 'WAV Export — Pro only'}>
          {exporting ? <Loader2 size={13} className="animate-spin" /> : <Download size={13} />}
          Export WAV {user?.subscription_plan !== 'pro' && <span className="text-[9px] border border-forge-gold/50 text-forge-gold px-1 rounded">PRO</span>}
        </button>
        <button
          onClick={() => { setSaveBeatTitle(beat?.title ? beat.title + ' (copy)' : 'My Beat'); setShowSaveDialog(true) }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-forge-black border border-forge-border text-forge-muted hover:text-forge-accent text-xs transition-colors"
          title="Save as new beat in My Beats">
          <Save size={12} /> Save as Beat
        </button>
        {user?.subscription_plan === 'pro' && (
          <button
            onClick={handlePublishBeat}
            disabled={beat?.status === 'published' || beat?.status === 'generating'}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-forge-black border border-forge-border text-forge-muted hover:text-green-400 text-xs transition-colors disabled:opacity-40"
            title="Publish beat to marketplace">
            <Globe size={12} /> {beat?.status === 'published' ? 'Published' : beat?.status === 'generating' ? 'Generating...' : 'Publish'}
          </button>
        )}
        {isPro && (
          <button
            onClick={() => { setInviteModal(true); loadCollaborators() }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-forge-black border border-forge-border text-forge-muted hover:text-forge-cyan text-xs transition-colors"
            title="Invite collaborator">
            <Users size={12} /> Invite
          </button>
        )}
      </div>

      {/* ── Main Layout ─────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">
        {/* Single scrollable timeline container */}
        <div
          className="flex-1 overflow-x-auto overflow-y-auto"
          ref={timelineScrollRef}
          onWheel={handleTimelineWheel}
        >
          <div style={{ minWidth: `${zoom * 100}%` }}>

            {/* Ruler row — sticky top */}
            <div className="flex bg-[#0e0e0e] border-b border-forge-border sticky top-0 z-20">
              {/* Track label cell — sticky left */}
              <div className="w-56 flex-shrink-0 sticky left-0 z-30 bg-[#0e0e0e] border-r border-forge-border px-2 py-2 flex items-center justify-between">
                <span className="text-[10px] text-forge-muted uppercase tracking-widest">Tracks</span>
                <div className="flex items-center gap-1">
                  <button onClick={addAudioTrack} className="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] text-forge-muted hover:text-forge-accent border border-forge-border/60 hover:border-forge-accent/60 transition-colors">
                    <Plus size={8} /> Audio
                  </button>
                  <button onClick={addMidiTrack} className="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] text-forge-muted hover:text-forge-gold border border-forge-border/60 hover:border-forge-gold/60 transition-colors">
                    <Plus size={8} /> MIDI
                  </button>
                  <button onClick={addVocalTrack} className="flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[9px] text-forge-muted hover:text-red-400 border border-forge-border/60 hover:border-red-400/60 transition-colors">
                    <Plus size={8} /> Vocal
                  </button>
                </div>
              </div>
              {/* Ruler ticks */}
              <div className="flex-1 relative h-8 cursor-pointer" onClick={handleRulerClick}>
                <div className="absolute inset-0 flex items-end pb-1">
                  {Array.from({ length: 13 }, (_, i) => i * 15).map(t => (
                    <div key={t} className="absolute flex flex-col items-center" style={{ left: `${(t/TIMELINE_DURATION)*100}%` }}>
                      <div className="w-px h-2 bg-forge-border/60" />
                      <span className="text-[9px] text-forge-muted font-mono mt-0.5">{fmt(t)}</span>
                    </div>
                  ))}
                  {Array.from({ length: TIMELINE_DURATION/5 + 1 }, (_, i) => i * 5).filter(t => t % 15 !== 0).map(t => (
                    <div key={t} className="absolute bottom-1" style={{ left: `${(t/TIMELINE_DURATION)*100}%` }}>
                      <div className="w-px h-1 bg-forge-border/30" />
                    </div>
                  ))}
                </div>
                <div className="absolute top-0 bottom-0 flex flex-col items-center pointer-events-none" style={{ left: `${(playhead/TIMELINE_DURATION)*100}%`, transform: 'translateX(-50%)' }}>
                  <div className="w-2 h-2 bg-forge-accent rotate-45 mt-1" />
                  <div className="w-px flex-1 bg-forge-accent" />
                </div>
              </div>
            </div>

            {/* Track rows */}
            {tracks.map(track => (
              <div key={track.id}
                className={clsx('flex border-b border-forge-border/40', selectedTrack === track.id ? 'bg-[#1a1410]/60' : 'hover:bg-[#111]/40')}
                onClick={() => setSelectedTrack(track.id)}>

                {/* Track header — sticky left */}
                <div className="w-56 flex-shrink-0 sticky left-0 z-10 bg-[#0a0a0a] border-r border-forge-border/40 p-2 space-y-1.5">
                  <div className="flex items-center gap-1.5">
                    <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: track.color }} />
                    <span className="text-xs text-forge-text font-medium truncate flex-1">{track.name}</span>
                    <div className="flex gap-0.5 flex-shrink-0">
                      {track.type === 'midi' && (
                        <button onClick={e => { e.stopPropagation(); setShowPianoRoll(track.id) }}
                          className="p-0.5 text-forge-muted hover:text-forge-gold transition-colors text-[10px]" title="Piano Roll">
                          ⊞
                        </button>
                      )}
                      <button onClick={e => { e.stopPropagation(); updateTrack(track.id, { muted: !track.muted }) }}
                        className={clsx('px-1 py-0.5 rounded text-[9px] font-bold border transition-colors',
                          track.muted ? 'bg-forge-accent border-forge-accent text-white' : 'border-forge-border text-forge-muted hover:text-forge-text'
                        )} title="Mute">M</button>
                      <button onClick={e => { e.stopPropagation(); updateTrack(track.id, { solo: !track.solo }) }}
                        className={clsx('px-1 py-0.5 rounded text-[9px] font-bold border transition-colors',
                          track.solo ? 'bg-forge-gold border-forge-gold text-black' : 'border-forge-border text-forge-muted hover:text-forge-text'
                        )} title="Solo">S</button>
                      <button onClick={e => { e.stopPropagation(); updateTrack(track.id, { armed: !track.armed }) }}
                        className={clsx('px-1 py-0.5 rounded text-[9px] font-bold border transition-colors',
                          track.armed ? 'bg-red-500 border-red-500 text-white animate-pulse' : 'border-forge-border text-forge-muted hover:text-red-400'
                        )} title="Arm for recording">R</button>
                      {track.type !== 'beat' && (
                        <button onClick={e => { e.stopPropagation(); deleteTrack(track.id) }}
                          className="p-0.5 text-forge-muted hover:text-red-400 transition-colors" title="Delete track">
                          <Trash2 size={10} />
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <Volume2 size={8} className="text-forge-muted flex-shrink-0" />
                    <input type="range" min={0} max={1} step={0.01} value={track.volume}
                      onChange={e => { e.stopPropagation(); updateTrack(track.id, { volume: parseFloat(e.target.value) }) }}
                      onClick={e => e.stopPropagation()} className="flex-1 accent-forge-accent h-1" />
                    <span className="text-[9px] text-forge-muted w-5 text-right">{Math.round(track.volume*100)}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-[9px] text-forge-muted w-6">PAN</span>
                    <input type="range" min={-1} max={1} step={0.01} value={track.pan}
                      onChange={e => { e.stopPropagation(); updateTrack(track.id, { pan: parseFloat(e.target.value) }) }}
                      onClick={e => e.stopPropagation()} className="flex-1 accent-forge-accent h-1" />
                    <span className={clsx('text-[9px] w-6 text-right', track.pan < -0.05 ? 'text-blue-400' : track.pan > 0.05 ? 'text-forge-accent' : 'text-forge-muted')}>
                      {track.pan === 0 ? 'C' : track.pan < 0 ? `L${Math.round(-track.pan*100)}` : `R${Math.round(track.pan*100)}`}
                    </span>
                  </div>
                </div>

                {/* Timeline area */}
                <div
                  className="flex-1 relative overflow-hidden"
                  style={{ height: 88, cursor: toolCursor[tool] }}
                  onClick={e => handleTimelineClick(e, track.id)}
                  onMouseMove={e => handleTimelineMouseMove(e)}
                >
                  <div className="absolute inset-0" style={{ background: `${track.color}06` }} />
                  {/* BPM grid lines */}
                  <div className="absolute inset-0 pointer-events-none overflow-hidden">
                    {(() => {
                      const secPerBeat = 60 / bpm
                      const bars: JSX.Element[] = []
                      let t = 0; let idx = 0
                      while (t < TIMELINE_DURATION) {
                        const left = (t / TIMELINE_DURATION) * 100
                        const isBar = idx % 4 === 0
                        bars.push(<div key={t} className="absolute top-0 bottom-0" style={{ left: `${left}%`, width: 1, background: isBar ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.03)' }} />)
                        t += secPerBeat; idx++
                      }
                      return bars
                    })()}
                  </div>

                  {recording && recordingTrackId.current === track.id && liveAnalyser && (
                    <LiveRecordingWave analyser={liveAnalyser} recStartPct={recStartPct} />
                  )}

                  {track.clips.map(clip => {
                    const left  = (clip.startTime / TIMELINE_DURATION) * 100
                    const width = (clip.duration  / TIMELINE_DURATION) * 100
                    return (
                      <div key={clip.id}
                        className="absolute top-1.5 bottom-1.5 rounded-md border"
                        style={{ left: `${left}%`, width: `${Math.max(width, 0.3)}%`, minWidth: 4, borderColor: track.color+'50', background: track.color+'14', overflow: 'visible' }}
                        onMouseDown={e => handleClipMouseDown(e, track.id, clip, 'none')}
                        onClick={e => { handleClipClick(e, track.id, clip); if (tool === 'select') setSelectedClip({ trackId: track.id, clipId: clip.id }) }}
                        onContextMenu={e => { e.preventDefault(); setContextMenu({ x: e.clientX, y: e.clientY, trackId: track.id, clipId: clip.id, trackType: track.type }) }}
                        onWheel={e => handleClipWheel(e, track.id, clip.id)}
                      >
                        {track.type !== 'midi' && (
                          <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
                            {(() => {
                              const totalDur = track.audioUrl
                                ? (audioFileDurations.current.get(track.audioUrl) ?? (clip.offset + clip.duration))
                                : clip.duration
                              const wPct = totalDur > 0 ? (totalDur / clip.duration) * 100 : 100
                              const lPct = totalDur > 0 ? -(clip.offset / clip.duration) * 100 : 0
                              return (
                                <div
                                  ref={el => {
                                    if (el && track.audioUrl) {
                                      const existing = waveContainers.current.get(clip.id)
                                      if (!existing || existing !== el) {
                                        waveContainers.current.set(clip.id, el)
                                        if (!wsInstances.current.has(clip.id)) {
                                          initWaveSurfer(clip.id, el, track.audioUrl, track.color)
                                        }
                                      }
                                    }
                                  }}
                                  style={{ position: 'absolute', top: 0, bottom: 0, left: `${lPct}%`, width: `${wPct}%`, minHeight: 56 }}
                                />
                              )
                            })()}
                          </div>
                        )}

                        {track.type === 'midi' && (
                          <div className="absolute inset-0 overflow-hidden" onDoubleClick={e => { e.stopPropagation(); setShowPianoRoll(track.id) }} title="Double-click to open Piano Roll">
                            {track.midiNotes.length === 0 ? (
                              <button onClick={e => { e.stopPropagation(); setShowPianoRoll(track.id) }} className="absolute inset-0 flex items-center justify-center text-[10px] text-forge-muted hover:text-forge-text">
                                Open Piano Roll
                              </button>
                            ) : track.midiNotes.slice(0, 80).map(n => (
                              <div key={n.id} className="absolute rounded-sm" style={{
                                left: `${(n.startBeat / 16) * 100}%`,
                                width: `${Math.max((n.durationBeats / 16) * 100, 0.5)}%`,
                                top: `${100 - ((n.note - 24) / 72) * 100}%`,
                                height: 3, background: track.color,
                              }} />
                            ))}
                          </div>
                        )}

                        {clip.fadeIn != null && clip.fadeIn > 0 && (
                          <div className="absolute left-0 top-0 bottom-0 pointer-events-none" style={{ width: `${(clip.fadeIn/clip.duration)*100}%`, background: `linear-gradient(to right, ${track.color}70, transparent)` }} />
                        )}
                        {clip.fadeOut != null && clip.fadeOut > 0 && (
                          <div className="absolute right-0 top-0 bottom-0 pointer-events-none" style={{ width: `${(clip.fadeOut/clip.duration)*100}%`, background: `linear-gradient(to left, ${track.color}70, transparent)` }} />
                        )}

                        {tool === 'gain' ? (
                          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                            <span className="text-[10px] font-mono text-forge-gold bg-black/60 rounded px-1">{(clip.clipGain ?? 1).toFixed(2)}x ↕</span>
                          </div>
                        ) : clip.clipGain !== undefined && clip.clipGain !== 1 && (
                          <div className="absolute top-1 left-1 text-[9px] font-mono text-forge-gold">{clip.clipGain.toFixed(2)}x</div>
                        )}

                        <div className="absolute left-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-white/20 rounded-l z-10"
                          onMouseDown={e => { e.stopPropagation(); handleClipMouseDown(e, track.id, clip, 'resizeLeft') }} />
                        <div className="absolute right-0 top-0 bottom-0 w-2 cursor-ew-resize hover:bg-white/20 rounded-r z-10"
                          onMouseDown={e => { e.stopPropagation(); handleClipMouseDown(e, track.id, clip, 'resizeRight') }} />

                        {/* Fade handles — draggable circles visible when fade tool active */}
                        {tool === 'fade' && (
                          <>
                            {/* Fade-in handle: positioned at end of fade-in zone */}
                            <div
                              className="absolute z-20 w-4 h-4 rounded-full border-2 border-white shadow-lg"
                              style={{
                                background: '#60a5fa',
                                left: `${((clip.fadeIn || 0) / clip.duration) * 100}%`,
                                top: '50%',
                                transform: 'translate(-50%, -50%)',
                                cursor: 'ew-resize',
                              }}
                              title={`Fade in: ${((clip.fadeIn || 0)).toFixed(1)}s — drag to adjust`}
                              onMouseDown={e => {
                                e.stopPropagation(); e.preventDefault()
                                fadeDrag.current = { type: 'in', trackId: track.id, clipId: clip.id, startX: e.clientX, origFade: clip.fadeIn || 0 }
                              }}
                            />
                            {/* Fade-out handle: positioned at start of fade-out zone */}
                            <div
                              className="absolute z-20 w-4 h-4 rounded-full border-2 border-white shadow-lg"
                              style={{
                                background: '#fb923c',
                                left: `${((clip.duration - (clip.fadeOut || 0)) / clip.duration) * 100}%`,
                                top: '50%',
                                transform: 'translate(-50%, -50%)',
                                cursor: 'ew-resize',
                              }}
                              title={`Fade out: ${((clip.fadeOut || 0)).toFixed(1)}s — drag to adjust`}
                              onMouseDown={e => {
                                e.stopPropagation(); e.preventDefault()
                                fadeDrag.current = { type: 'out', trackId: track.id, clipId: clip.id, startX: e.clientX, origFade: clip.fadeOut || 0 }
                              }}
                            />
                          </>
                        )}
                      </div>
                    )
                  })}

                  <div className="absolute top-0 bottom-0 w-px bg-forge-accent/70 pointer-events-none"
                    style={{ left: `${(playhead/TIMELINE_DURATION)*100}%` }} />

                  {recording && recordingTrackId.current === track.id && (
                    <div className="absolute top-1.5 right-2 flex items-center gap-1 z-20">
                      <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                      <span className="text-[9px] text-red-400 font-bold">REC</span>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Add track row */}
            <div className="flex items-center gap-4 px-4 py-3 border-b border-forge-border/20 opacity-50 hover:opacity-100 transition-opacity">
              <button onClick={addAudioTrack} className="flex items-center gap-2 text-xs text-forge-muted hover:text-forge-accent transition-colors">
                <FileAudio size={13} /> Import Audio
              </button>
              <span className="text-forge-border">·</span>
              <button onClick={addMidiTrack} className="flex items-center gap-2 text-xs text-forge-muted hover:text-forge-gold transition-colors">
                <span className="text-sm">⊞</span> Add MIDI Track
              </button>
              <span className="text-forge-border">·</span>
              <button onClick={addVocalTrack} className="flex items-center gap-2 text-xs text-forge-muted hover:text-red-400 transition-colors">
                <Mic size={13} /> Add Vocal Track
              </button>
              <span className="text-forge-border">·</span>
              <button onClick={handleRecord} className={clsx('flex items-center gap-2 text-xs transition-colors', recording ? 'text-red-400 animate-pulse' : 'text-forge-muted hover:text-red-400')}>
                <Circle size={11} fill="currentColor" /> {recording ? 'Stop Recording' : 'Record Now'}
              </button>
            </div>

          </div>
        </div>

        {/* ── Effects Sidebar ─────────────────────────────────────────────── */}
        {showFxPanel && selTrack && (
          <div className="w-72 flex-shrink-0 border-l border-forge-border overflow-hidden flex flex-col">
            <EffectsPanel
              track={selTrack}
              onChange={fx => updateTrack(selTrack.id, { effects: fx })}
              presets={presets}
              onSavePreset={savePreset}
              onLoadPreset={fx => updateTrack(selTrack.id, { effects: fx })}
              onDeletePreset={deletePreset}
              onClose={() => setShowFxPanel(false)}
              isPro={user?.subscription_plan === 'pro'}
            />
          </div>
        )}
      </div>

      {/* ── Save as Beat Dialog ─────────────────────────────────────────────── */}
      {showSaveDialog && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70" onClick={() => setShowSaveDialog(false)}>
          <div className="bg-forge-card border border-forge-border rounded-2xl p-6 w-full max-w-sm shadow-2xl" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display text-white flex items-center gap-2"><Save size={16} className="text-forge-accent" /> Save as New Beat</h3>
              <button onClick={() => setShowSaveDialog(false)}><X size={18} className="text-forge-muted" /></button>
            </div>
            <input
              value={saveBeatTitle}
              onChange={e => setSaveBeatTitle(e.target.value)}
              placeholder="Beat title..."
              className="input-forge w-full mb-3"
              autoFocus
              onKeyDown={e => { if (e.key === 'Enter') handleSaveAsBeat() }}
            />
            <p className="text-xs text-forge-muted mb-4">Creates a new entry in My Beats using this beat's audio. Studio edits (MIDI, clips, FX) are saved separately in your browser.</p>
            <div className="flex gap-2">
              <button onClick={() => setShowSaveDialog(false)} className="flex-1 py-2 rounded-xl border border-forge-border text-forge-muted text-sm hover:text-forge-text transition-colors">Cancel</button>
              <button onClick={handleSaveAsBeat} disabled={!saveBeatTitle.trim() || savingBeat}
                className="btn-primary flex-1 py-2 text-sm disabled:opacity-40 flex items-center justify-center gap-2">
                {savingBeat ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />} Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Context Menu ────────────────────────────────────────────────────── */}
      {inviteModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60" onClick={() => setInviteModal(false)}>
          <div className="bg-forge-card border border-forge-border rounded-2xl p-6 w-full max-w-md shadow-xl mx-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-display text-lg text-white flex items-center gap-2"><Users size={18} className="text-forge-orange" />Collaborators</h3>
              <button onClick={() => setInviteModal(false)} className="text-forge-muted hover:text-forge-text"><X size={18} /></button>
            </div>

            {/* Current collaborators */}
            <div className="mb-4">
              {collabsLoading ? (
                <div className="flex items-center gap-2 text-forge-muted text-sm py-2">
                  <Loader2 size={14} className="animate-spin" /> Loading…
                </div>
              ) : collaborators.length === 0 ? (
                <p className="text-forge-muted text-sm py-2">No collaborators yet.</p>
              ) : (
                <ul className="space-y-2 mb-2">
                  {collaborators.map(c => (
                    <li key={c.id} className="flex items-center justify-between gap-3 bg-forge-black/50 rounded-lg px-3 py-2">
                      <div className="flex items-center gap-2 min-w-0">
                        {c.avatar_url ? (
                          <img src={c.avatar_url} alt="" className="w-7 h-7 rounded-full object-cover flex-shrink-0" />
                        ) : (
                          <div className="w-7 h-7 rounded-full bg-forge-orange/20 flex items-center justify-center flex-shrink-0 text-xs text-forge-orange font-bold">
                            {(c.display_name || c.username || '?')[0].toUpperCase()}
                          </div>
                        )}
                        <span className="text-white text-sm truncate">@{c.username || c.display_name}</span>
                      </div>
                      <button
                        onClick={() => handleKickCollaborator(c.id, c.username || c.display_name)}
                        disabled={kickingCollaborator === c.id}
                        className="flex items-center gap-1 px-2 py-1 rounded text-xs text-red-400 hover:bg-red-500/10 border border-red-500/20 hover:border-red-500/40 transition-colors disabled:opacity-50 flex-shrink-0"
                        title="Remove collaborator"
                      >
                        {kickingCollaborator === c.id ? <Loader2 size={12} className="animate-spin" /> : <X size={12} />}
                        Kick
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="border-t border-forge-border pt-4">
              <p className="text-forge-muted text-xs mb-3">Invite someone by username</p>
              <input
                autoFocus
                value={inviteUsername}
                onChange={e => setInviteUsername(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleInvite()}
                className="input-forge w-full mb-3"
                placeholder="@username"
              />
              <div className="flex gap-2">
                <button onClick={() => setInviteModal(false)} className="flex-1 btn-secondary py-2">Close</button>
                <button onClick={handleInvite} disabled={inviting} className="flex-1 btn-primary py-2 flex items-center justify-center gap-2 disabled:opacity-50">
                  {inviting ? <Loader2 size={14} className="animate-spin" /> : <UserPlus size={14} />}
                  Send Invite
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {contextMenu && (
        <div
          className="fixed z-[9999] bg-forge-card border border-forge-border rounded-xl shadow-2xl py-1 min-w-[160px]"
          style={{ top: contextMenu.y, left: contextMenu.x }}
          onMouseLeave={() => setContextMenu(null)}
        >
          <button className="w-full px-4 py-2 text-xs text-forge-text hover:bg-forge-accent/10 text-left transition-colors"
            onClick={() => { deleteClip(contextMenu.trackId, contextMenu.clipId); setContextMenu(null) }}>
            🗑 Delete Clip
          </button>
          <button className="w-full px-4 py-2 text-xs text-forge-text hover:bg-forge-accent/10 text-left transition-colors"
            onClick={() => {
              setTracks(prev => prev.map(t => t.id !== contextMenu.trackId ? t : {
                ...t, clips: t.clips.map(c => c.id !== contextMenu.clipId ? c : { ...c, fadeIn: c.duration * 0.1 })
              })); setContextMenu(null)
            }}>
            ↗ Fade In (10%)
          </button>
          <button className="w-full px-4 py-2 text-xs text-forge-text hover:bg-forge-accent/10 text-left transition-colors"
            onClick={() => {
              setTracks(prev => prev.map(t => t.id !== contextMenu.trackId ? t : {
                ...t, clips: t.clips.map(c => c.id !== contextMenu.clipId ? c : { ...c, fadeOut: c.duration * 0.1 })
              })); setContextMenu(null)
            }}>
            ↘ Fade Out (10%)
          </button>
          <button className="w-full px-4 py-2 text-xs text-forge-text hover:bg-forge-accent/10 text-left transition-colors"
            onClick={() => {
              setTracks(prev => prev.map(t => t.id !== contextMenu.trackId ? t : {
                ...t, clips: t.clips.map(c => c.id !== contextMenu.clipId ? c : { ...c, fadeIn: 0, fadeOut: 0 })
              })); setContextMenu(null)
            }}>
            ✕ Remove Fades
          </button>
          {contextMenu.trackType === 'midi' && (
            <button className="w-full px-4 py-2 text-xs text-forge-gold hover:bg-forge-gold/10 text-left transition-colors"
              onClick={() => { createMidiRegion(contextMenu.trackId); setContextMenu(null) }}>
              ⊞ Create MIDI Region
            </button>
          )}
        </div>
      )}
    </div>
  )
}
