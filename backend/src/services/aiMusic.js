/**
 * BeatHole AI Music Generation Service
 *
 * Priority order:
 *   1. Stability AI Stable Audio v2  (AI_MODEL=stability, needs STABILITY_API_KEY)
 *      → Best quality, official REST API, ~€0.05/beat, no RunPod needed
 *   2. MusicGen via RunPod           (AI_MODEL=musicgen, needs BARK_API_URL + RUNPOD_API_KEY)
 *      → Open-source fallback, decent quality
 *   3. Mock                          (development / no keys set)
 */

const axios   = require('axios');
const fs      = require('fs');
const path    = require('path');
const { v4: uuidv4 } = require('uuid');
const { uploadBuffer } = require('./storage');

const UPLOADS_DIR = process.env.UPLOADS_DIR || path.join(__dirname, '../../uploads');
const BASE_URL    = process.env.BASE_URL    || 'https://api.beathole.com';

// ─── Prompt builders ─────────────────────────────────────────────────────────

const GENRE_DESCRIPTORS = {
  'trap':      'trap beat, hard 808 bass, rolling trap hi-hats, snare on the 3, trap drums, dark synths',
  'drill':     'UK drill beat, dark 808 bass slides, aggressive triplet hi-hats, minor key, drill percussion',
  'hip-hop':   'hip hop beat, boom bap drums, punchy kick and snare, sampled bass line, hip hop groove',
  'hip hop':   'hip hop beat, boom bap drums, punchy kick and snare, sampled bass line, hip hop groove',
  'boom bap':  'classic boom bap, vinyl-textured drums, jazz bass, golden era hip hop groove, NYC style',
  'r&b':       'R&B beat, smooth soulful groove, live-sounding drums, warm bass, lush chords, neo soul',
  'afrobeats': 'afrobeats, West African percussion groove, talking drum pattern, afro bass line, vibrant rhythm',
  'dancehall': 'dancehall riddim, one-drop rhythm, reggae-electronic bass, caribbean percussion groove',
  'lo-fi':     'lo-fi hip hop, vinyl crackle, mellow jazz chords, relaxed boom bap drums, warm analog tone',
  'pop':       'pop music beat, catchy groove, polished modern production, four-on-the-floor rhythm',
  'electronic':'electronic music, synthesizer bass, programmed drums, EDM production, electronic groove',
};

const MOOD_DESCRIPTORS = {
  'dark':       'dark atmosphere, ominous minor key, menacing brooding tone',
  'energetic':  'high energy, driving powerful rhythm, intense electrifying feel',
  'chill':      'chill relaxed vibe, mellow laid-back groove, smooth calm atmosphere',
  'aggressive': 'aggressive hard-hitting feel, intense forceful energy, raw power',
  'emotional':  'emotional melancholic mood, heartfelt expressive tone, introspective',
  'uplifting':  'uplifting positive energy, bright feel-good melody, motivational',
  'mysterious': 'mysterious suspenseful atmosphere, eerie haunting undertones',
  'romantic':   'romantic smooth warm tone, sensual groove, intimate feel',
};

/**
 * Build a rich prompt for Stability AI Stable Audio
 * Stability responds best to dense, music-production-vocabulary text
 */
const buildStabilityPrompt = ({ genre, mood, bpm, style, key, prompt }) => {
  const parts = [];
  const g = (genre || '').toLowerCase();
  const m = (mood  || '').toLowerCase();

  // Genre expansion
  if (GENRE_DESCRIPTORS[g]) {
    parts.push(GENRE_DESCRIPTORS[g]);
  } else if (genre) {
    parts.push(`${genre} beat, ${genre} instrumental music`);
  }

  // Mood expansion
  if (MOOD_DESCRIPTORS[m]) {
    parts.push(MOOD_DESCRIPTORS[m]);
  } else if (mood) {
    parts.push(`${mood} mood`);
  }

  if (style)  parts.push(`${style} production style`);
  if (bpm)    parts.push(`${bpm} BPM`);
  if (key)    parts.push(`key of ${key}`);
  if (prompt) parts.push(prompt);

  // Quality + instrumental enforcement
  parts.push(
    'instrumental, no vocals, no singing, no spoken words, ' +
    'professional studio quality beat, clear crisp mix, ' +
    'well-produced music, singable catchy melody, radio-ready production'
  );

  return parts.join(', ');
};


// ─── Main entry point ─────────────────────────────────────────────────────────

const generateBeat = async ({
  genre, mood, bpm, style, title, key, prompt, duration, beatType, beatId,
  referenceAudio, referenceStrength,
  promptMode, caption, lyrics, inferSteps, guidanceScale,
  schedulerType, cfgType, omegaScale, guidanceInterval,
  guidanceIntervalDecay, minGuidanceScale, useErgTag,
  useErgLyric, useErgDiffusion, ossSteps,
  guidanceScaleText, guidanceScaleLyric, seed,
}) => {
  // null/undefined = user chose "Random" → pick 120–240s
  const targetDuration = duration || (Math.floor(Math.random() * 121) + 120); // 120–240s random
  console.log(`🎵 Generating: "${title}" | genre=${genre} | ${targetDuration}s | mode=${beatType || 'audio'}`);

  // 1. Stability AI (best quality)
  if (process.env.STABILITY_API_KEY) {
    return await callStabilityAudio({ genre, mood, bpm, style, key, prompt, duration: targetDuration });
  }

  // 2. ACE-Step / MusicGen via RunPod
  if (process.env.BARK_API_URL) {
    return await callMusicGenRunPod({
      genre, mood, bpm, style, key, prompt, duration: targetDuration, beatType, beatId,
      referenceAudio, referenceStrength,
      promptMode, caption, lyrics, inferSteps, guidanceScale,
      schedulerType, cfgType, omegaScale, guidanceInterval,
      guidanceIntervalDecay, minGuidanceScale, useErgTag,
      useErgLyric, useErgDiffusion, ossSteps,
      guidanceScaleText, guidanceScaleLyric, seed,
    });
  }

  // 3. Mock (dev)
  console.warn('⚠️  No AI configured — using mock. Set STABILITY_API_KEY for real beats.');
  return await mockGenerate({ title });
};

// ─── Stability AI Stable Audio v2 ─────────────────────────────────────────────

const callStabilityAudio = async ({ genre, mood, bpm, style, key, prompt, duration }) => {
  const apiKey = process.env.STABILITY_API_KEY;
  const generationId = uuidv4();
  const textPrompt = buildStabilityPrompt({ genre, mood, bpm, style, key, prompt });

  // Stability AI Stable Audio max is 190s
  const seconds = Math.min(duration, 180);

  console.log(`🎵 Stability AI | "${textPrompt.slice(0, 80)}..." | ${seconds}s`);

  // Use FormData (built-in Node 18+)
  const FormData = global.FormData || require('form-data');
  const form = new FormData();
  form.append('prompt', textPrompt);
  form.append('output_format', 'mp3');
  form.append('seconds_total', String(seconds));
  form.append('steps', '100');

  const response = await axios.post(
    'https://api.stability.ai/v2beta/audio/stable-audio-v2-generate',
    form,
    {
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: 'audio/*',
        ...(form.getHeaders ? form.getHeaders() : {}),
      },
      responseType: 'arraybuffer',
      timeout: 300000, // 5 min
    }
  );

  // Save the mp3 to uploads
  const filename  = `${generationId}.mp3`;
  const beatDir   = path.join(UPLOADS_DIR, 'beats');
  if (!fs.existsSync(beatDir)) fs.mkdirSync(beatDir, { recursive: true });
  const filePath  = path.join(beatDir, filename);
  fs.writeFileSync(filePath, Buffer.from(response.data));

  const mp3Url = `${BASE_URL}/uploads/beats/${filename}`;
  console.log(`✅ Stability AI done | ${mp3Url}`);

  return {
    generationId,
    wavUrl:      null,
    mp3Url,
    previewUrl:  mp3Url,
    duration:    seconds,
    waveformData: generateMockWaveform(),
  };
};

// ─── MusicGen via RunPod ──────────────────────────────────────────────────────

const callMusicGenRunPod = async ({
  genre, mood, bpm, style, key, prompt, duration, beatType, beatId,
  referenceAudio, referenceStrength,
  promptMode, caption, lyrics, inferSteps, guidanceScale,
  schedulerType, cfgType, omegaScale, guidanceInterval,
  guidanceIntervalDecay, minGuidanceScale, useErgTag,
  useErgLyric, useErgDiffusion, ossSteps,
  guidanceScaleText, guidanceScaleLyric, seed,
}) => {
  const baseUrl = process.env.BARK_API_URL;
  const apiKey  = process.env.RUNPOD_API_KEY;
  const generationId = uuidv4();

  // Clamp to 1:30–4:00; handler does chunked multi-pass generation internally
  const clampedDuration = Math.max(90, Math.min(duration, 240));
  const outputMode = beatType === 'midi' ? 'midi' : 'audio';

  console.log(`🎵 MusicGen RunPod | genre=${genre} style="${style}" mood=${mood} bpm=${bpm} | ${clampedDuration}s | mode=${outputMode}`);

  const response = await axios.post(
    `${baseUrl}/run`,
    {
      input: {
        genre:       genre  || '',
        style:       style  || '',
        mood:        mood   || '',
        bpm:         bpm    || undefined,
        key:         key    || '',
        prompt:      prompt || '',   // user's extra free-text only
        duration:    clampedDuration,
        output_mode:       outputMode,
        beatId:            beatId || '',
        referenceAudio:    referenceAudio    || undefined,
        referenceStrength: referenceStrength || undefined,
        promptMode:        promptMode || process.env.ACESTEP_PROMPT_MODE || 'native',
        caption:           caption || undefined,
        lyrics:            lyrics || undefined,
        inferSteps:        inferSteps ?? undefined,
        guidanceScale:     guidanceScale ?? undefined,
        schedulerType:     schedulerType || undefined,
        cfgType:           cfgType || undefined,
        omegaScale:        omegaScale ?? undefined,
        guidanceInterval:  guidanceInterval ?? undefined,
        guidanceIntervalDecay: guidanceIntervalDecay ?? undefined,
        minGuidanceScale:  minGuidanceScale ?? undefined,
        useErgTag:         useErgTag ?? undefined,
        useErgLyric:       useErgLyric ?? undefined,
        useErgDiffusion:   useErgDiffusion ?? undefined,
        ossSteps:          ossSteps || undefined,
        guidanceScaleText: guidanceScaleText ?? undefined,
        guidanceScaleLyric: guidanceScaleLyric ?? undefined,
        seed:              seed ?? undefined,
      },
    },
    {
      headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
      timeout: 30000,
    }
  );

  const jobId = response.data?.id;
  if (!jobId) throw new Error('RunPod did not return a job ID');
  console.log(`⏳ RunPod job queued: ${jobId}`);
  return await pollRunPodJob(baseUrl, apiKey, jobId, generationId, outputMode);
};

// Poll up to 40 minutes (480 × 5s) — enough for 3-pass 5-min generation
const pollRunPodJob = async (baseUrl, apiKey, jobId, generationId, outputMode = 'audio', maxAttempts = 480, intervalMs = 5000) => {
  for (let i = 0; i < maxAttempts; i++) {
    await sleep(intervalMs);
    const statusRes = await axios.get(`${baseUrl}/status/${jobId}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
      timeout: 15000,
    });
    const { status, output, error } = statusRes.data;
    if (i % 6 === 0) console.log(`⏳ Job ${jobId}: ${status} (${Math.round((i * intervalMs) / 60000)}min elapsed)`);
    if (status === 'COMPLETED') {
      if (outputMode === 'midi') {
        if (!output?.midi_tracks) throw new Error('Job complete but no MIDI tracks');
      } else {
        if (!output?.wav_url) throw new Error('Job complete but no audio URL');
      }
      return await decodeAndStore(output, generationId, outputMode);
    }
    if (status === 'FAILED' || status === 'CANCELLED') throw new Error(`RunPod job ${status}: ${error || 'unknown'}`);
  }
  throw new Error('RunPod job timeout after 40 minutes');
};

const decodeAndStore = async (output, generationId, outputMode = 'audio') => {
  // MIDI mode — upload the preview audio, pass through MIDI track data
  if (outputMode === 'midi') {
    let wavUrl = null;
    if (output.wav_base64) {
      const wavBuffer = Buffer.from(output.wav_base64, 'base64');
      wavUrl = await uploadBuffer(wavBuffer, `${generationId}_preview.wav`, 'audio/wav', 'beats');
    }
    return {
      generationId,
      wavUrl,
      mp3Url:      null,
      previewUrl:  wavUrl,
      duration:    output.duration_seconds || (output.total_bars ? (output.total_bars * 4 * 60) / (output.tempo_bpm || 120) : 60),
      waveformData: generateMockWaveform(),
      midiTracks:  output.midi_tracks || [],
      stems:       null,
    };
  }

  // Audio mode — handler already uploaded main WAV directly to avoid RunPod payload limit
  const wavUrl = output.wav_url;

  // stem_urls are already uploaded by the handler directly to the backend
  const stems = (output.stem_urls && Object.keys(output.stem_urls).length > 0)
    ? output.stem_urls
    : null;

  return {
    generationId,
    wavUrl,
    mp3Url:      null,
    previewUrl:  wavUrl,
    duration:    output.duration_seconds || 60,
    waveformData: generateMockWaveform(),
    stems,
    midiTracks:  null,
  };
};

// ─── Mock ─────────────────────────────────────────────────────────────────────

const mockGenerate = async ({ title }) => {
  await sleep(1500);
  return {
    generationId: uuidv4(),
    wavUrl:       null,
    mp3Url:       null,
    previewUrl:   null,
    duration:     120,
    waveformData: generateMockWaveform(),
    isMock:       true,
  };
};

// ─── Utilities ────────────────────────────────────────────────────────────────

const generateMockWaveform = (points = 200) => {
  const data = [];
  for (let i = 0; i < points; i++) {
    const base     = Math.sin(i * 0.1) * 0.3;
    const noise    = (Math.random() - 0.5) * 0.7;
    const envelope = Math.sin((i / points) * Math.PI);
    data.push(Math.abs(base + noise) * envelope);
  }
  return data;
};

const generateTitle = ({ genre, mood, style }) => {
  const adjectives = ['Dark', 'Epic', 'Smooth', 'Hard', 'Chill', 'Fire', 'Cold', 'Raw', 'Deep', 'Vibrant'];
  const nouns      = ['Wave', 'Night', 'Storm', 'Dream', 'Flow', 'Pulse', 'Rush', 'Vibe', 'Zone', 'Cipher'];
  const adj  = adjectives[Math.floor(Math.random() * adjectives.length)];
  const noun = nouns[Math.floor(Math.random() * nouns.length)];
  // Only include the first word/token of genre to avoid long prompts ending up in the title
  const genreLabel = genre ? genre.split(/[\s,]+/)[0] : '';
  return `${adj} ${noun}${genreLabel ? ` (${genreLabel})` : ''}`;
};

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

module.exports = { generateBeat, generateTitle, generateMockWaveform };
