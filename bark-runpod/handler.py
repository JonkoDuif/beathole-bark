"""
BeatHole AI — ACE-Step 1.5 RunPod Handler

Audio mode : ACE-Step 1.5 text2music → stem extraction → Basic Pitch MIDI transcription
MIDI mode  : Programmatic MIDI generation + ACE-Step preview audio
"""
import sys, base64, io, os, math, random, json as _json, urllib.request, uuid
print(f"[startup] Python {sys.version}", flush=True)

try:
    import torch
    print(f"[startup] torch {torch.__version__} | CUDA: {torch.cuda.is_available()}", flush=True)
except Exception as e:
    print(f"[startup] TORCH ERROR: {e}", flush=True); sys.exit(1)

try:
    import runpod
    print("[startup] runpod OK", flush=True)
except Exception as e:
    print(f"[startup] RUNPOD ERROR: {e}", flush=True); sys.exit(1)

try:
    import numpy as np
    import soundfile as sf
    print("[startup] numpy/soundfile OK", flush=True)
except Exception as e:
    print(f"[startup] NUMPY/SOUNDFILE ERROR: {e}", flush=True); sys.exit(1)

try:
    from acestep.inference import GenerationParams, GenerationConfig, generate_music
    print("[startup] ACE-Step inference OK", flush=True)
except Exception as e:
    print(f"[startup] ACE-STEP INFERENCE ERROR: {e}", flush=True); sys.exit(1)

try:
    # ACE-Step 1.5 handler classes
    try:
        from acestep.handler import AceStepHandler as _DitHandlerClass
        from acestep.llm_inference import LLMHandler as _LLMHandlerClass
        print("[startup] ACE-Step 1.5 handlers OK", flush=True)
    except ImportError:
        from acestep.handlers import DitHandler as _DitHandlerClass, LLMHandler as _LLMHandlerClass
        print("[startup] ACE-Step 1.0 handlers OK", flush=True)
except Exception as e:
    print(f"[startup] ACE-STEP HANDLERS ERROR: {e}", flush=True); sys.exit(1)

try:
    from basic_pitch.inference import predict as _bp_predict
    _BASIC_PITCH_OK = True
    print("[startup] basic-pitch OK", flush=True)
except Exception as _bp_err:
    _BASIC_PITCH_OK = False
    print(f"[startup] basic-pitch not available (MIDI transcription disabled): {_bp_err}", flush=True)

try:
    from demucs.pretrained import get_model as _demucs_get_model
    from demucs.apply import apply_model as _demucs_apply
    _DEMUCS_OK = True
    print("[startup] demucs OK", flush=True)
except Exception as _demucs_err:
    _DEMUCS_OK = False
    print(f"[startup] demucs not available: {_demucs_err}", flush=True)

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000   # ACE-Step native output sample rate
STEM_SR     = 22050   # stem upload sample rate — good quality, manageable size

_dit_handler       = None   # turbo — text2music generation
_dit_base_handler  = None   # base  — extract task (turbo does NOT support extract)
_llm_handler       = None

_demucs_model      = None
_DEMUCS_MODEL_NAME = "htdemucs_6s"
_DEMUCS_SR         = 44100   # htdemucs native sample rate

def _patch_low_cpu_mem():
    """
    Monkey-patch PreTrainedModel.from_pretrained to force low_cpu_mem_usage=False.

    ACE-Step's DiT model calls Tensor.item() during __init__ (e.g. for positional
    encodings or layer-norm params). With low_cpu_mem_usage=True (the HF default),
    tensors are placed on the 'meta' device first — meaning they have shape but no
    data. Calling .item() on a meta tensor raises:
      RuntimeError: Tensor.item() cannot be called on meta tensors

    Forcing low_cpu_mem_usage=False makes from_pretrained load weights directly on
    CPU with real data, so .item() works. ACE-Step then moves the model to CUDA.
    """
    try:
        from transformers import modeling_utils as _mu
        _orig = _mu.PreTrainedModel.from_pretrained

        @classmethod
        def _patched(cls, pretrained_model_name_or_path, *args, **kwargs):
            kwargs.setdefault("low_cpu_mem_usage", False)
            return _orig.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)

        _mu.PreTrainedModel.from_pretrained = _patched
        print("[patch] from_pretrained patched: low_cpu_mem_usage=False default", flush=True)
        return _orig  # caller can restore if needed
    except Exception as e:
        print(f"[patch] WARNING: could not patch from_pretrained: {e}", flush=True)
        return None


def _restore_pretrained(orig):
    if orig is None:
        return
    try:
        from transformers import modeling_utils as _mu
        _mu.PreTrainedModel.from_pretrained = orig
    except Exception:
        pass


def get_handlers():
    """
    Initialize AceStepHandler + LLMHandler.

    ACE-Step 1.5 uses project_root to locate its checkpoints:
      {project_root}/checkpoints/acestep-v15-turbo/     <- DiT model
      {project_root}/checkpoints/vae/                   <- VAE
      {project_root}/checkpoints/Qwen3-Embedding-0.6B/  <- text encoder

    initialize_service() downloads these from HuggingFace automatically on
    first run. Set ACESTEP_PROJECT_ROOT in the environment (Dockerfile) so
    files persist at a known location across pod restarts.
    """
    global _dit_handler, _dit_base_handler, _llm_handler
    if _dit_handler is not None:
        return _dit_handler, _dit_base_handler, _llm_handler

    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    lm_model     = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-0.6B")

    print(f"[acestep] Loading handlers (device={device}, project_root={project_root})...", flush=True)

    orig_fp = _patch_low_cpu_mem()

    def _init_dit(config_name: str):
        """Init a fresh DitHandler with the given config. Returns handler or None."""
        h = _DitHandlerClass()
        if not hasattr(h, "initialize_service"):
            return h  # ACE-Step 1.0 — no config_name concept
        print(f"[acestep] initialize_service(config_path={config_name})...", flush=True)
        try:
            h.initialize_service(project_root=project_root, config_path=config_name, device=device)
        except Exception as e:
            print(f"[acestep] initialize_service({config_name}) raised: {e}", flush=True)
        if getattr(h, "model", None) is None:
            print(f"[acestep] {config_name}: model is None after init", flush=True)
            return None
        missing = [c for c in ("model", "vae", "text_encoder", "text_tokenizer")
                   if getattr(h, c, None) is None]
        if missing:
            print(f"[acestep] {config_name}: missing components {missing}", flush=True)
            return None
        print(f"[acestep] {config_name} handler ready", flush=True)
        return h

    # ── Turbo model: text2music, cover, repaint (fastest) ─────────────────────
    _dit_handler = _init_dit("acestep-v15-turbo")
    if _dit_handler is None:
        _dit_handler = _init_dit("acestep-v15-base")  # fallback
    if _dit_handler is None:
        _restore_pretrained(orig_fp)
        raise RuntimeError("Could not initialize any ACE-Step DiT model")

    # ── Base model: extract task (turbo does NOT support extract per docs) ─────
    # Only load if turbo succeeded — they share VAE/text-encoder so VRAM cost
    # is mostly just the DiT weights (~4 GB extra).
    _dit_base_handler = _init_dit("acestep-v15-base")
    if _dit_base_handler is None:
        print("[acestep] Base model unavailable — extract will use turbo (may be lower quality)", flush=True)
        _dit_base_handler = _dit_handler  # graceful fallback

    _restore_pretrained(orig_fp)

    # ── LLM (optional — enables chain-of-thought captioning) ──────────────────
    _llm_handler = _LLMHandlerClass()
    lm_checkpoint_dir = os.path.join(project_root, "checkpoints")
    try:
        _llm_handler.initialize(checkpoint_dir=lm_checkpoint_dir, lm_model_path=lm_model, device=device)
        print("[acestep] LLM handler initialized", flush=True)
    except Exception as lm_err:
        print(f"[acestep] LLM init failed (non-fatal): {lm_err}", flush=True)
        _llm_handler = None

    print("[acestep] All handlers ready", flush=True)
    return _dit_handler, _dit_base_handler, _llm_handler


# ── Utility: audio → WAV base64 ───────────────────────────────────────────────
def np_to_wav_b64(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T          # (channels, samples) → (samples, channels)
    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    sf.write(buf, audio_i16, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def resample_mono(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Naive linear resample — no scipy dependency."""
    if audio.ndim == 2:
        audio = audio.mean(axis=0) if audio.shape[0] <= audio.shape[1] else audio.mean(axis=1)
    if src_sr == dst_sr:
        return audio
    n = int(len(audio) * dst_sr / src_sr)
    return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


# ── Utility: upload stem WAV to backend ───────────────────────────────────────
def upload_main_audio(backend_url: str, api_key: str, beat_id: str, wav_b64: str) -> str:
    payload = _json.dumps({
        "beatId":    beat_id,
        "wavBase64": wav_b64,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/internal/beats/upload",
        data=payload,
        headers={"Content-Type": "application/json", "x-internal-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return _json.loads(resp.read())["url"]


def upload_stem(backend_url: str, api_key: str, beat_id: str, stem_name: str, wav_b64: str) -> str:
    payload = _json.dumps({
        "beatId":    beat_id,
        "stemName":  stem_name,
        "wavBase64": wav_b64,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{backend_url.rstrip('/')}/api/internal/stems",
        data=payload,
        headers={"Content-Type": "application/json", "x-internal-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return _json.loads(resp.read())["url"]


# ═══════════════════════════════════════════════════════════════════════════════
# ── Genre / mood / instrument vocabulary ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Multiple variants per genre — each variant leads with the DRUM/RHYTHM character
# so ACE-Step always generates the right percussion foundation first.
GENRE_VARIANTS = {
    "trap": [
        "dark Atlanta trap, thunderous 808 sub kick with pitch slides, rolling 32nd-note triplet hi-hats, hard cracking snare on beat 3, chopped soul sample, ominous minor melody",
        "hard trap banger, heavy distorted 808 bass drum, rapid hi-hat triplet rolls with flams, crisp snare beat 3, pitched vocal chop, menacing dark atmosphere",
        "melodic trap, emotional piano loop over heavy 808 kicks and rolling hi-hat triplets, hard snare beat 3, introspective dark energy, sad atmospheric trap",
        "cinematic trap, orchestral strings over thunderous 808 sub kick, rapid hi-hat rolls, hard snare crack beat 3, epic dark production, cold dramatic atmosphere",
        "midnight trap, massive detuned 808 kick, haunting piano sample, 32nd-note hi-hat rolls building tension, punchy dry snare beat 3, dark brooding groove",
        "aggressive trap, explosive 808 bass drum with sub rumble, rapid triplet hi-hat patterns, sharp snare, dark threatening atmosphere, heavy trap production",
    ],
    "drill": [
        "UK drill, deep sliding 808 kick drum, sparse cold 16th-note hi-hats, sharp crisp snare short decay, off-beat 808 sub hits, ominous dark gritty street sound",
        "Brooklyn drill, heavy booming 808 bass kick with pitch glide, minimal sinister hi-hat pattern, tight hard snare, eerie piano loop, dark aggressive production",
        "Chicago drill, menacing minor chords, booming 808 kick with slides, staccato cold hi-hats, hard punchy snare, thunderous off-beat percussion",
        "UK drill banger, deep sliding 808 bass, sparse aggressive hi-hat pattern, chilling string stabs, hard snare crack, cold brooding atmosphere",
        "melodic drill, haunting flute loop over sliding 808 kicks, minimal cold hi-hats, crisp hard snare, cold emotional energy, dark drill groove",
    ],
    "hip hop": [
        "soulful hip hop, punchy sampled kick on beats 1 and 3, crisp snare on 2 and 4, swinging 16th-note hi-hats, warm vinyl-sampled piano, deep punchy groove",
        "dusty boom bap-influenced hip hop, hard knocking kick and snare, loose swinging hi-hats, muffled soul chop, warm analog texture, head-nodding groove",
        "dark hip hop, boom bap kick and snare pattern, cinematic orchestral sample, swinging hi-hats, underground raw feel",
        "smooth hip hop, laid-back kick groove on 1 and 3, crisp snare 2 and 4, shuffled hi-hats, lush Rhodes melody, relaxed head-nodding vibe",
        "raw hip hop, heavy punchy kick, cracking snare, gritty sampled horn stab, swinging hi-hat pattern, street authenticity, deep bass pocket",
    ],
    "hip-hop": [
        "soulful hip hop, punchy sampled kick on beats 1 and 3, crisp snare on 2 and 4, swinging 16th-note hi-hats, warm vinyl-sampled piano, deep groove",
        "dark hip hop, hard knocking kick and snare, loose swinging hi-hats, cinematic orchestral sample, boom bap rhythm, underground feel",
        "smooth hip hop, laid-back kick and snare pocket, shuffled hi-hats, lush Rhodes melody, relaxed head-nodding vibe",
        "raw gritty hip hop, heavy punchy kick, cracking sampled snare, gritty horn stab, swinging hi-hats, street authenticity",
    ],
    "boom bap": [
        "classic boom bap, deep sampled kick drum on beats 1 and 3, hard rimshot snare on 2 and 4, loose swinging hi-hats with ghost notes, dusty vinyl soul sample",
        "golden era boom bap, heavy sampled kick and snare break, swinging hi-hat groove with choke, soulful horn sample, jazz-infused dusty groove",
        "gritty boom bap, punchy compressed kick, cracking sampled snare, loose triplet hi-hat swing, chopped vocal sample, crackly vinyl texture, underground rawness",
        "cinematic boom bap, deep sampled drum break, swinging hi-hats with fills, orchestral brass sample, booming kick, timeless old-school feel",
        "hard boom bap, heavy knocking kick on 1 and 3, loud cracking snare 2 and 4, tight hi-hat pattern, aggressive underground hip hop energy",
    ],
    "r&b": [
        "silky smooth R&B, smooth kick and snare pocket groove, shuffled hi-hat with ghost notes, warm Fender Rhodes chords, lush reverb, intimate late-night feel",
        "neo-soul R&B, laid-back kick on 1 and 3, snare ghost notes on 2 and 4, swinging hi-hats, live electric guitar licks, sensual atmosphere",
        "dark R&B, deep kick groove, snare on 2 and 4 with fills, shuffled hi-hats, minor key piano, deep bass pulse, moody emotional tension",
        "contemporary R&B, punchy modern kick pattern, sharp snare, 16th-note hi-hat drive, lush pad harmonies, polished feel",
        "soulful R&B, warm pocket kick and snare, swinging hi-hats, gospel-influenced chord progressions, warm vintage keys, heartfelt",
    ],
    "afrobeats": [
        "vibrant afrobeats, syncopated kick pattern, talking drum accents, layered conga and shaker, off-beat hi-hat groove, bright guitar melody, joyful West African rhythm",
        "afrobeats banger, complex afro polyrhythm, layered hand percussion, syncopated kick, catchy synth hook, infectious high-energy arrangement",
        "amapiano-influenced afrobeats, log drum bass hits, hypnotic layered percussion groove, talking drum, shaker, South African sound",
        "afro fusion, vibrant syncopated percussion layers, kora-inspired melody, rich polyrhythm, conga and bongo groove, warm organic feel",
    ],
    "dancehall": [
        "riddim dancehall, punchy kick on beat 1 and and-beat-3, snare on 2 and 4, offbeat hi-hat chop, deep digital bass, Caribbean energy",
        "modern dancehall, trap-influenced drum pattern, melodic synth hook, punchy kick, snare, danceable Caribbean groove",
        "lovers rock dancehall, smooth reggae-inspired kick and snare, offbeat hi-hat, warm bass, romantic Caribbean vibe",
    ],
    "lo-fi": [
        "lo-fi hip hop, muffled kick with soft attack, brushed snare with vinyl grain, loose dusty hi-hats, mellow Rhodes melody, late night studying atmosphere",
        "rainy day lo-fi, dusty muffled kick drum, soft brushed snare decay, gentle hi-hats, warm tape saturation, sleepy piano chords, nostalgic haze",
        "jazzy lo-fi, laid-back kick and brushed snare groove, dusty vinyl hi-hats, muted guitar chords, vinyl grain texture, cozy introspective atmosphere",
        "lo-fi soul, gentle kick drum, soft brushed snare, warm upright bass, dusty hi-hats, Rhodes chords, slow meditative groove",
        "lo-fi chill, relaxed kick and snare pattern, soft hi-hats, vocal chop sample, mellow analog warmth, drifting daydream feel",
    ],
    "electronic": [
        "cutting-edge electronic, driving kick on every beat, aggressive 16th-note hi-hats, layered synthesizer arpeggios, explosive driving pulse, massive synth stabs",
        "experimental electronic, complex mechanical percussion pattern, textured modular synth, evolving complex pads, intense futuristic atmosphere, relentless forward motion",
        "dark electronic, cold industrial kick pattern, hypnotic relentless percussion, heavy bass punch, industrial synth, underground club power",
        "melodic electronic, four-on-the-floor kick drive, crisp clap on backbeats, soaring emotional synth lead, lush atmospheric depth, euphoric energy",
        "hard electronic, brutal crushing percussion hits, distorted bass pressure, relentless drum pattern, intense aggressive energy, raw club force",
    ],
    "house": [
        "classic Chicago house, stomping four-on-the-floor kick hitting every beat, open hi-hat on every offbeat, crisp clap on 2 and 4, punchy soulful piano chords, warm analog bassline, peak-hour dancefloor energy",
        "funky house, driving four-on-the-floor kick, syncopated open hi-hat groove, punchy clap on backbeats, chopped vocal stab, thick deep bass line, lively dancefloor energy",
        "melodic house, powerful four-on-the-floor kick drive, hi-hat offbeat groove, euphoric emotional piano progression, soaring lush pads, crowd-lifting energy",
        "afro house, driving four-on-the-floor kick, hypnotic tribal percussion layers, open hi-hat offbeats, heavy groove, driving bassline, spiritual dancefloor power",
        "peak-hour house, aggressive four-on-the-floor kick drive, syncopated clap and hi-hat, relentless bass stab, intense club energy, full powerful production",
    ],
    "deep house": [
        "deep house, soft four-on-the-floor kick with warm attack, subtle open hi-hat on offbeats, soft clap on backbeats, powerful warm sub bass, rich atmospheric pads, late-night groove",
        "soulful deep house, rolling four-on-the-floor kick, gentle swinging hi-hat groove, dusty organ chords, rolling punchy deep bass, emotional intimate dancefloor energy",
        "minimal deep house, soft kick on every beat, subtle hi-hat offbeat, hypnotic groove, deep resonant bass thump, dark underground tension",
        "vocal deep house, rolling kick pattern, gentle hi-hat swing, emotional chord progression, lush reverb depth, late night warmth",
    ],
    "tech house": [
        "driving tech house, hard punchy kick on every beat, percussive hi-hat groove with rim shot accents, menacing hypnotic bassline, dark relentless underground groove, peak-hour intensity",
        "funky tech house, heavy mechanical four-on-the-floor kick, aggressive syncopated hi-hat, thick filtered bass stab, hard rim percussion, high-energy arrangement",
        "industrial tech house, crushing distorted kick every beat, industrial hi-hat pattern, heavy percussion hits, raw club energy, intense mechanical force",
        "hard tech house, aggressive bass pressure, relentless four-on-the-floor groove, dystopian underground energy, full powerful production",
    ],
    "techno": [
        "dark Berlin techno, crushing industrial kick drum on every beat at 140 BPM, aggressive 16th-note hi-hats, heavy metallic percussion, relentless hypnotic rhythm, cold mechanical energy",
        "hard techno, massive distorted kick every beat, rapid mechanical hi-hat rhythm, industrial percussion texture, aggressive distortion, brutal driving intensity",
        "melodic techno, relentless industrial kick pattern, 16th-note hi-hat drive, soaring emotional synth lead, building cathartic peak-hour energy",
        "minimal techno, hard pounding kick drum on every beat, hypnotic hi-hat groove, deep resonant bass, rising tension, dark underground drive",
        "peak techno, massive crushing kick pressure, relentless hi-hat drive, sweeping filter automation, overwhelming forward momentum",
    ],
    "edm": [
        "festival EDM, four-on-the-floor kick at 128 BPM, punchy clap on 2 and 4, build-up snare roll into massive drop, massive soaring synth lead, explosive epic stadium production",
        "progressive EDM, driving kick on every beat, clap on backbeats, pre-drop snare fill, emotional chord progression, huge soaring synth lead, euphoric breakdown into massive drop",
        "electro EDM, punchy kick every beat, sharp clap, retro-futuristic synth stab, high-energy peak-hour intensity, full energetic production",
        "big room EDM, crushing four-on-the-floor kick drop, punchy clap, huge chord stabs, massive build tension release, overwhelming crowd energy",
        "electro house EDM, driving four-on-the-floor kick, punchy distorted bass, relentless clap and hi-hat, explosive drop energy, high-intensity production",
    ],
    "dubstep": [
        "heavy riddim dubstep, half-time kick on beat 1 at 140 BPM, massive reverb-soaked snare on beat 3, sparse hi-hats, crushing reese bass wobble, devastating sub-bass drop, filthy modulated bass growl",
        "brostep dubstep, half-time drum pattern, enormous snare crack on beat 3, massive distorted growl bass, face-melting drop intensity, full-frequency destruction",
        "melodic dubstep, half-time kick and massive snare pattern, soaring emotional synth lead, cinematic orchestral build, explosive huge cathartic drop",
        "dark dubstep, half-time kick pattern at 140 BPM, bone-crushing reverb snare beat 3, ominous cinematic tension build, industrial bass growl, dystopian overwhelming energy",
        "neuro dubstep, half-time drum pattern, complex bass sound design, technical rhythmic precision, intense mechanical energy, deep sub pressure",
    ],
    "drum and bass": [
        "liquid drum and bass, rapid 170 BPM syncopated breakbeat, rolling kick and sharp snare fills, fast 16th-note hi-hats, soulful Rhodes melody, smooth atmospheric feel",
        "dark drum and bass, frantic amen break at 170 BPM, syncopated kick and snare, rapid hi-hat drive, ominous aggressive bass wobble, intense underground energy",
        "neurofunk drum and bass, precise breakbeat at 170 BPM, technical syncopated percussion, complex bass modulation, futuristic intense mechanical energy",
        "jump up drum and bass, energetic amen-style break, fast syncopated kick and snare at 170 BPM, heavyweight punchy bass stab, crowd-moving energy",
        "hard drum and bass, relentless fast breakbeat, rapid syncopated percussion at 170 BPM, crushing bass pressure, intense raw underground power",
    ],
    "dnb": [
        "liquid DnB, rolling 170 BPM amen break, syncopated kick and snare, rapid hi-hats, warm atmospheric pads, soulful melodic forward momentum",
        "dark DnB, frantic breakbeat at 170 BPM, aggressive syncopated percussion, rapid amen break, relentless underground raw energy",
        "neurofunk DnB, precise 170 BPM breakbeat, technical syncopated percussion, clinical bass design, cold futuristic atmosphere",
        "jump up DnB, energetic fast breakbeat, syncopated kick pattern at 170 BPM, heavy punchy bass, high-intensity dancefloor power",
    ],
    "jungle": [
        "classic jungle, frantic chopped amen break, complex syncopated kick and snare, reggae bass wobble, raw 90s underground energy, rhythmic intensity",
        "dark jungle, ominous frantic amen break pattern, rapid syncopated percussion complexity, intense rhythmic drive, powerful underground energy",
    ],
    "reggaeton": [
        "Latin reggaeton, dembow kick on beat 1 and the-and-of-beat-3, snare on beats 2 and 4, shuffled hi-hat, bright synth melody, urban Caribbean groove",
        "dark reggaeton, aggressive dembow kick pattern, snare on backbeats, shuffled hi-hat, minor key melody, street urban Latin energy",
        "romantic reggaeton, punchy dembow kick and snare, smooth hi-hat groove, smooth melody, warm production, sensual Latin groove",
    ],
    "latin": [
        "vibrant Latin beat, layered conga and bongo patterns, timbales, shaker and claves, complex polyrhythmic Latin percussion, bright brass stabs, festive rhythmic energy",
        "Latin jazz fusion, complex polyrhythm with congas, timbales, and piano, complex chord voicings, rich harmonic groove",
        "dark Latin, minor key guitar over layered Latin percussion, conga and shaker, moody atmosphere, intense rhythmic tension",
    ],
    "jazz": [
        "late-night jazz, swinging jazz ride cymbal, brushed snare with ghost notes, kick on downbeats, hi-hat on 2 and 4, smoky walking bass, complex chord voicings, intimate atmosphere",
        "jazz fusion, live jazz drum kit with swing feel, ride cymbal pattern, ghost note snare, electric piano, intricate rhythmic interplay, improvisational energy",
        "dark jazz, loose swinging brushed drums, dissonant chords, sparse minor melody, noir cinematic atmosphere",
        "upbeat jazz, driving bebop kick and snare, swinging ride cymbal, brass section, lively groove, classic bebop energy",
    ],
    "soul": [
        "deep soul, tight groove kick on 1 and 3, snare on 2 and 4 with pocket, shuffled hi-hat, warm organ chords, gospel-influenced melody, heartfelt emotional expression",
        "classic soul, punchy Motown kick and snare groove, swinging hi-hats, lush string arrangement, timeless emotional depth",
        "neo-soul, laid-back kick and snare pocket, swinging hi-hats, complex jazz chords, live bass groove, introspective warm atmosphere",
        "southern soul, driving kick and snare, syncopated hi-hat, raw emotional guitar licks, deep groove",
    ],
    "funk": [
        "classic funk, syncopated kick on the one, tight ghost-note snare, open hi-hat choke pattern, slapped bass groove, wah guitar, deep pocket, James Brown influence",
        "jazz funk, driving funky kick pattern, crisp snare with ghost notes, open hi-hat rhythm, complex chord progression, electric piano, sophisticated groove",
        "dark funk, syncopated minor key kick groove, tight snare, heavy bass line, tense rhythmic interplay",
        "P-funk inspired, deep funky kick and snare groove, layered percussion, spacey synth over heavy groove, cosmic feel",
    ],
    "pop": [
        "polished pop, punchy four-on-the-floor kick, sharp snare on 2 and 4, 16th-note hi-hat drive, catchy piano hook, modern production, hook-driven radio sound",
        "dark pop, minor key drum groove, emotional kick and snare pattern, hi-hat drive, minor key emotional melody, lush production",
        "indie pop, live drum feel with organic kick and snare, natural hi-hat groove, warm melody, bittersweet emotional atmosphere",
        "electropop, driving electronic kick every beat, punchy clap, bright synth melody, contemporary high-energy production",
    ],
    "cinematic": [
        "epic cinematic score, orchestral percussion with timpani and taiko drums, dramatic snare rolls, sweeping orchestral strings, powerful tension and release",
        "dark cinematic, dramatic percussion build, heavy timpani hits, dissonant brass stabs, ominous cello, tense suspenseful atmosphere",
        "emotional cinematic, soft orchestral percussion, delicate strings, solo piano over lush strings, heartbreaking melodic theme",
        "action cinematic, driving taiko and snare rolls, aggressive orchestral percussion, heroic brass, relentless forward momentum",
        "ambient cinematic, soft atmospheric percussion, evolving pads, sparse piano motif, vast spacious sound",
    ],
    "ambient": [
        "deep ambient, minimal soft percussion texture, slowly evolving pad layers, vast spacious dreamscape",
        "dark ambient, ominous drone, sparse percussion texture, unsettling atmospheric tension, cinematic darkness",
        "ambient electronic, soft rhythmic pulse, crystalline synth arpeggios, peaceful floating atmosphere",
        "nature-inspired ambient, organic textural rhythm, gentle melodic motif, serene evolving soundscape",
    ],
    "phonk": [
        "dark Memphis phonk, heavy distorted kick with 808 sub, cowbell accents on upbeats, aggressive rolling hi-hats, chopped soul vocal, distorted 808, vintage cassette texture",
        "aggressive phonk, crunchy distorted 808 kick, rapid hi-hat rolls with cowbell, hard snare slam, dark chord sample, raw underground energy",
        "drift phonk, hard distorted 808 bass drum, rolling hi-hats and cowbell, dark synth atmosphere, high-energy intense momentum",
        "melodic phonk, heavy distorted kick, cowbell hi-hat pattern, haunting vocal chop, distorted bass, eerie dark production",
    ],
    "cloud rap": [
        "hazy cloud rap, slow minimal trap drums, sparse reverb-soaked kick and snare, soft hi-hats, washed-out ambient trap, atmospheric reverb-drenched melody, ethereal feel",
        "dark cloud rap, slow muffled kick, distant reverb snare, ominous pad texture, soft hi-hats, cold atmospheric trap",
        "melodic cloud rap, slow hazy drum pattern, muffled kick and soft snare, emotional piano under heavy reverb, dreamy introspective atmosphere",
    ],
    "grime": [
        "raw UK grime, staccato 140 BPM kick pattern, sharp metallic hi-hats, hard snare hit, staccato synth stabs, rolling 8-bar beat, aggressive dark energy",
        "dark grime, aggressive 140 BPM drum pattern, cold hi-hat rhythm, hard snare, minor key synth riff, sparse industrial rhythm, cold London sound",
        "melodic grime, driving 140 BPM percussion, rapid hi-hat, hard snare, emotional keyboard lead, intense urban energy",
    ],
    "uk garage": [
        "UK garage, 2-step shuffle kick on beat 1 and and-beat-3, snare on 2 and 4, syncopated swinging hi-hat groove, pitched vocal chop, warm bass, soulful house influence",
        "speed garage, shuffled 2-step kick, swinging hi-hat, deep sub bass, skippy hi-hat rhythm, late-night underground groove",
        "dark UK garage, shuffled 2-step drum pattern, minor key synth, swinging hi-hats, moody introspective atmosphere",
    ],
    "synthwave": [
        "retro synthwave, driving electronic drum machine pattern, four-on-the-floor kick, massive gated reverb snare on 2 and 4, 16th-note closed hi-hats, pulsing analog bass, neon-lit arpeggios, 80s cinematic atmosphere",
        "dark synthwave, cold electronic drum pattern, hard gated reverb snare, driving kick, cold minor key synth, dystopian Blade Runner atmosphere",
        "melodic synthwave, electronic drum machine groove, gated snare reverb, driving kick, emotional lead synth, lush reverb, romantic 80s nostalgia",
        "outrun synthwave, powerful electronic drum pattern, four-on-the-floor kick, gated reverb snare, powerful bass pulse, heroic lead melody",
    ],
    "vaporwave": [
        "vaporwave, slow hazy drum machine pattern, slowed pitched-down melody, lush reverb wash, nostalgic 80s corporate dream",
        "dark vaporwave, unsettling slow drum pattern, slowed sample, dreamlike distortion, lonely atmosphere",
        "mallsoft vaporwave, ambient minimal drum texture, surreal nostalgic sound, deconstructed elevator music",
    ],
    "hyperpop": [
        "hyperpop, distorted glitchy drum pattern, heavy distorted 808 kick, glitched snare effects, chaotic hi-hat fills, maximalist layered production, chaotic energy",
        "dark hyperpop, aggressive glitched-out drums, heavy distorted kick, stutter snare, pitched vocal glitch, intense hyper-saturated contrast",
        "melodic hyperpop, distorted drum pattern, heavy glitch kick, emotional lead over glitchy production, bittersweet energy",
    ],
    "pluggnb": [
        "dark pluggnb, slow melodic trap drums, heavy 808 sub kick, soft snare on beat 3, gentle hi-hat pattern, smooth minor key melody, emotional introspective feel",
        "pluggnb, laid-back slow trap groove, deep 808 kick, minimal hi-hats, lush pad chords, slow dark 808 bass, hazy romantic atmosphere",
        "melodic pluggnb, slow trap drum groove, heavy 808 kick, minimal percussion, warm piano over slow trap rhythm, intimate emotional depth",
    ],
    "jersey club": [
        "jersey club, frantic 4x4 kick pattern at 130+ BPM, rapid hi-hat chopping, syncopated snare hits, chopped vocal sample, high-energy dancefloor groove",
        "dark jersey club, fast four-on-the-floor kick, chopped percussion pattern, rapid hi-hat, ominous bass stab, intense underground club energy",
    ],
    "reggae": [
        "classic reggae, one-drop kick on beat 3 only, rim shot on beats 2 and 4, open hi-hat on offbeats, offbeat guitar skank, warm bass, roots vibration",
        "dub reggae, sparse one-drop kick pattern, echo-soaked snare, heavy bass, delay effects, hypnotic trippy feel",
        "dancehall reggae, punchy digital riddim kick and snare, offbeat hi-hat chop, modern Caribbean production energy",
    ],
    "gospel": [
        "powerful gospel, driving kick and snare groove, full choir harmony, driving rhythm, uplifting spiritual energy",
        "contemporary gospel, modern punchy drum production, emotional piano, soulful choir texture",
        "dark gospel, driving minor key drum groove, kick and snare pattern, raw emotional power, building drama",
    ],
    "blues": [
        "delta blues, shuffle rhythm kick and snare, driving 12-bar groove, raw electric guitar riff, gritty soulful expression",
        "modern blues, steady kick and snare groove, swinging hi-hats, emotional lead guitar, deep heartfelt feel",
        "blues jazz fusion, swinging jazz-influenced drum groove, complex chord progression, expressive guitar, sophisticated feel",
    ],
    "rock": [
        "hard rock, powerful live drum kit, hard kick on beats 1 and 3, crashing snare on 2 and 4, driving 8th-note hi-hats, driving electric guitar riff, raw energy",
        "indie rock, live drum kit feel, punchy kick and snare groove, natural hi-hat rhythm, warm guitar tone, melodic hook",
        "dark rock, heavy live drums, hard kick and crashing snare, aggressive hi-hat, heavy distorted guitar, minor key tension, brooding atmosphere",
    ],
    "rap": [
        "hard rap beat, heavy punchy kick on 1 and 3, cracking snare on 2 and 4, hi-hat drive, dark sample, street energy, raw production",
        "melodic rap, deep kick groove, crisp snare pattern, swinging hi-hats, emotional piano loop, introspective atmosphere",
    ],
    "metal": [
        "heavy metal, thunderous double-kick drum blast, crashing snare on 2 and 4, rapid 16th-note hi-hats, crushing distorted guitar riffs, massive wall of sound",
        "death metal, blastbeat double-kick pattern, alternating kick and snare, rapid percussion, crushing heavy distorted guitars, overwhelming power",
        "melodic metal, driving kick pattern, heavy snare, fast hi-hat drive, melodic guitar leads over crushing rhythm guitars, epic power",
    ],
    "amapiano": [
        "amapiano, deep log drum bass hits on every beat, layered piano chords, shuffled hi-hat groove, syncopated percussion, South African house energy",
        "amapiano banger, punchy log drum kick pattern, piano stab chords, rolling percussion, deep sub bass, vibrant South African groove",
    ],
    "afro house": [
        "afro house, driving four-on-the-floor kick, layered tribal percussion, congas and shakers, deep bassline, spiritual dancefloor energy",
        "afro house banger, stomping kick every beat, complex afro percussion layers, hypnotic groove, driving bassline, peak-hour energy",
    ],
    "future bass": [
        "future bass, driving electronic kick, powerful clap on backbeats, lush chord swells, bright supersaw synth, emotional atmospheric build, festival energy",
        "melodic future bass, punchy electronic drums, emotive chord progressions, soaring synth leads, massive euphoric drop, beautiful emotional energy",
    ],
    "trap soul": [
        "trap soul, slow trap drums with heavy 808 sub kick, minimal hi-hats, soft snare on beat 3, warm RnB-influenced chords, emotional dark melody, intimate atmosphere",
        "melodic trap soul, slow heavy 808 kicks, gentle rolling hi-hats, RnB chord progressions, sad piano melody, deep emotional introspective vibe",
    ],
}

# Rich mood descriptions — multiple variants per mood, one picked randomly
MOOD_VARIANTS = {
    "dark": [
        "dark ominous minor key, menacing brooding tension, cold threatening atmosphere",
        "pitch-black darkness, slow-burning dread, unsettling harmonic dissonance, shadowy",
        "deeply dark production, haunting minor chords, shadowy oppressive atmosphere, dark energy",
        "noir darkness, tense cinematic atmosphere, cold emotionless menace, dark brooding feel",
        "evil dark atmosphere, dissonant minor harmony, cold sinister production, oppressive dark sound",
    ],
    "sad": [
        "heartbreaking melancholy, tearful minor key melody, deep emotional sorrow, bittersweet pain",
        "devastating sadness, lonely isolated atmosphere, raw emotional vulnerability, quiet grief",
        "melancholic introspection, aching emotional melody, soulful heartbreak, late-night loneliness",
        "mournful sadness, weeping lead melody, slow emotional decay, poignant sorrowful atmosphere",
        "deeply sad emotion, crying emotional melody, hollow lonely feeling, heavy sorrow",
    ],
    "emotional": [
        "deeply emotional, raw vulnerability, heartfelt honest expression, moving touching atmosphere",
        "overwhelming emotion, intense personal feeling, powerful inner journey, emotionally resonant",
        "cathartic emotional release, deeply touching, resonant human connection, moving",
        "soul-stirring emotion, deeply felt personal expression, genuine raw vulnerability",
    ],
    "energetic": [
        "explosive high energy, adrenaline rush, relentless forward momentum, kinetic power",
        "driving intense energy, unstoppable force, electrifying rhythm, powerful momentum",
        "frenetic high-octane energy, fast and furious rhythm, heart-pounding intensity",
        "full-throttle energy, aggressive driving power, maximum intensity",
    ],
    "chill": [
        "mellow laid-back groove, smooth relaxed atmosphere, easygoing Sunday afternoon feel",
        "soft chill vibe, gentle floating melody, unhurried peaceful mood, calm tranquility",
        "warm chill atmosphere, hazy comfort, slow-breathing relaxation, soft and easy",
        "relaxed groovy vibe, smooth mellow flow, comfortable unhurried feel",
    ],
    "aggressive": [
        "brutally aggressive, hard-hitting in-your-face, relentless raw force, confrontational",
        "savage intensity, violent powerful energy, crushing heavy production, overwhelming aggression",
        "menacing aggression, cold calculated force, dominant threatening energy",
        "hostile aggressive attack, intense confrontational sound, raw brutal force",
    ],
    "uplifting": [
        "uplifting hopeful energy, soaring major key, inspiring triumph, bright optimism",
        "euphoric uplift, joyful emotional release, spiritual elevation, warm radiant energy",
        "motivating uplifting power, victorious momentum, hopeful bright atmosphere",
        "inspirational rising energy, triumphant major key mood, bright uplifting feeling",
    ],
    "mysterious": [
        "mysterious eerie tension, haunting unknown, suspenseful cinematic atmosphere",
        "enigmatic dark mystery, unsettling curiosity, strange other-worldly feel",
        "deep mysterious groove, hypnotic trance-like unknown, dark compelling atmosphere",
        "cryptic mysterious mood, eerie unsettling tension, dark curious atmosphere",
    ],
    "romantic": [
        "intimate romantic warmth, sensual smooth groove, tender emotional closeness",
        "passionate romance, deep feeling, lush warm atmosphere, heartfelt connection",
        "late-night romantic mood, silky smooth production, soft emotional intimacy",
        "sensual romantic atmosphere, warm intimate feeling, tender loving mood",
    ],
    "hard": [
        "hard-hitting heavy production, thick punishing bass, aggressive powerful drums, hard-knocking",
        "heavy knocking, dense brutal sound, walls of bass, relentless hard energy",
        "hard street sound, heavy low-end weight, tough uncompromising production",
        "hard aggressive sound, heavy hitting production, raw tough energy",
    ],
    "melodic": [
        "richly melodic, expressive lead instrument, deep harmonic layers, strong memorable hooks",
        "beautiful melodic composition, emotional lead melody, lush harmonic depth, melodic richness",
        "soulful melodic expression, singing lead line, rich emotional musical development",
        "deeply melodic, strong hooky lead, emotional harmonic progression",
    ],
    "happy": [
        "joyful bright happiness, feel-good major key, playful cheerful energy, sunny positivity",
        "carefree happy groove, light bouncy energy, warm smiling atmosphere, optimistic",
        "celebratory happiness, bright upbeat mood, lively joyful production",
    ],
    "angry": [
        "raw angry aggression, distorted heavy energy, explosive tension, furious intensity",
        "boiling rage, crushing heavy production, dark violent energy, uncontrolled force",
        "seething anger, tense grinding distortion, intense hostile atmosphere",
        "furious explosive anger, violent heavy production, aggressive dark intensity",
    ],
    "nostalgic": [
        "warm nostalgic longing, vintage analog texture, bittersweet memory, timeless past feel",
        "deep nostalgia, faded photograph sound, warm crackle and grain, lost time feeling",
        "melancholic nostalgia, old memory resurfacing, warm vintage atmosphere, wistful longing",
    ],
    "epic": [
        "massive epic scale, sweeping orchestral power, cinematic dramatic build, larger than life",
        "towering epic grandeur, heroic tension, powerful build and release, overwhelming scale",
        "epic cinematic intensity, dramatic momentum, colossal sound design, unforgettable impact",
    ],
    "dreamy": [
        "ethereal dreamy float, hazy reverb-drenched atmosphere, hypnotic soft blur, sleep state",
        "lush dreamy landscape, soft shimmering textures, weightless floating atmosphere",
        "dreamy surreal warmth, blurred soft edges, hypnotic drifting meditation",
    ],
    "bouncy": [
        "playful bouncy groove, catchy rhythmic energy, fun upbeat swing, head-bobbing momentum",
        "lively bouncy feel, energetic playful rhythm, infectious catchy movement",
    ],
    "raw": [
        "raw unpolished grit, underground authentic street energy, lo-fi dirty texture",
        "crude raw production, unprocessed honest sound, underground rough authenticity",
        "gritty street raw energy, imperfect human feel, honest unfiltered expression",
    ],
    "heavy": [
        "heavy crushing bass weight, thick dense low-end, massive punishing sub frequencies",
        "heavy powerful sound, crushing bass pressure, thick weighty production",
    ],
    "intense": [
        "intense relentless energy, driving forceful power, gripping overwhelming atmosphere",
        "maximum intensity, gripping forceful production, intense non-stop energy",
    ],
    "lonely": [
        "lonely isolated atmosphere, cold distant sound, hollow empty feeling, desolate",
        "deeply lonely, isolated cold atmosphere, distant reverb, hollow emotional space",
    ],
    "pain": [
        "raw emotional pain, aching sorrow, deep hurt and vulnerability, painful honest expression",
        "emotional anguish, deep pain and sorrow, heavy heartbreak, raw vulnerability",
    ],
    "hyped": [
        "hype energy, crowd-moving aggressive momentum, energetic exciting sound, hyped up feel",
        "maximum hype, aggressive high-energy, exciting crowd-pumping production",
    ],
    "trap": [  # "trap" used as mood/style keyword
        "trap-influenced rhythm, 808 sub bass groove, rolling hi-hat energy, dark atmospheric feel",
        "trap style energy, heavy 808 hits, rolling hi-hat patterns, dark menacing atmosphere",
    ],
}

# Random production texture tags added to every beat for variety
_PRODUCTION_TEXTURES = {
    "trap":          ["pitched vocal chop sample", "chopped soul loop", "granular pad texture",
                      "warm tape saturation", "vinyl crackle underneath", "lush reverb tail on snare",
                      "distant pad atmosphere", "orchestral string sample"],
    "drill":         ["cold atmospheric pad", "eerie string stabs", "pitched vocal sample",
                      "distant choir texture", "dark reverb-drenched atmosphere"],
    "hip hop":       ["dusty vinyl sample", "warm analog compression", "muffled soul chop",
                      "jazz piano sample", "horn stab", "vocal ad-lib chop"],
    "boom bap":      ["vinyl crackle texture", "dusty soul sample", "jazz horn stab",
                      "muffled guitar chop", "record scratch", "warm analog warmth"],
    "r&b":           ["lush string arrangement", "warm Rhodes comping", "breathy vocal texture",
                      "smooth guitar lick", "rich harmonic pads"],
    "lo-fi":         ["vinyl record crackle", "tape wobble", "rain sounds in background",
                      "warm tube saturation", "dusty filtered texture", "muffled distant feel"],
    "house":         ["soulful vocal chop", "percussive shaker groove", "warm organ stab",
                      "funky bass fill", "lush pad swell"],
    "deep house":    ["jazzy chord stab", "breathy vocal texture", "warm sub bass pulse",
                      "distant piano motif"],
    "techno":        ["metallic percussion hit", "industrial noise texture", "filter sweep",
                      "cold robotic atmosphere", "heavy reverb decay"],
    "phonk":         ["Memphis vocal sample", "cowbell accent", "distorted cassette texture",
                      "dark chord stab", "chopped rap vocal"],
    "cloud rap":     ["pitched reversed vocal", "ambient pad wash", "dreamy melody fragment",
                      "distant bell texture"],
    "synthwave":     ["analog synth arpeggios", "gated reverb snare texture", "neon pad atmosphere",
                      "chorus-drenched guitar", "lush stereo field"],
    "cinematic":     ["solo cello motif", "French horn swell", "tension string tremolo",
                      "timpani accent", "choir breath texture"],
    "ambient":       ["field recording texture", "granular synthesis pad", "evolving drone layer",
                      "shimmering overtone texture"],
    "_default":      ["warm reverb atmosphere", "subtle pad layer", "textured background element",
                      "rhythmic accent detail", "harmonic depth layer"],
}

# Atmospheric feel tags — added randomly to give ACE-Step creative direction
_FEEL_TAGS = [
    "late night 3am energy", "empty city streets", "rain on the window",
    "introspective inner thoughts", "staring at the ceiling", "driving alone at night",
    "neon lights and fog", "cold winter silence", "summer heat haze",
    "headphone listening music", "bedroom studio intimacy", "underground club basement",
    "rooftop sunset session", "cinematic storytelling", "raw emotional honesty",
    "street corner atmosphere", "luxury late night", "spiritual transcendence",
]

# High-energy electronic genres — use electronic section labels + higher guidance
_ELECTRONIC_GENRES = {
    "dubstep", "drum and bass", "dnb", "jungle", "techno", "tech house",
    "edm", "electronic", "house", "deep house", "hyperpop", "jersey club",
    "grime", "uk garage",
}

# Mix fullness tags — always injected to push ACE-Step toward richer arrangements
_FULLNESS_TAGS = [
    "multiple layered instruments, full rich arrangement, every frequency filled",
    "dense layered production, bass, mid, and high elements all present, no empty space",
    "full arrangement with drums, bass, melody, and atmosphere all active simultaneously",
    "richly layered mix, sub bass, mid bass, synth leads, pads, and percussion all present",
    "full-bodied production, powerful kick, deep bass, melodic layers, textured atmosphere",
    "packed arrangement, multiple simultaneous instrument layers, full frequency spectrum",
    "dense energetic mix, driving rhythm section with melodic and harmonic layers on top",
    "full production with prominent drums, deep bass groove, lead melody, and supporting pads",
]

# ── Genre-specific drum tags — ALWAYS injected so ACE-Step never forgets the rhythm ──
# These describe the EXACT percussion character of each genre in enough detail
# that the model knows the kick pattern, hi-hat style, and snare placement.
_GENRE_DRUM_TAGS = {
    "trap": [
        "thunderous 808 sub kick with pitch slides, rolling 32nd-note triplet hi-hats building tension, hard cracking snare on beat 3, sparse powerful kick pattern",
        "heavy distorted 808 bass drum, rapid triplet hi-hat rolls with flams and 32nd-note fills, punchy dry snare crack on beat 3, dark trap percussion",
        "massive sliding 808 kick sub, rapid-fire hi-hat triplets alternating 16th and 32nd patterns, sharp snare beat 3, aggressive trap hi-hat rolls",
    ],
    "drill": [
        "deep sliding 808 kick drum, sparse cold 16th-note hi-hats, sharp crisp snare with short tight decay, off-beat 808 sub hits, menacing UK drill drum pattern",
        "booming 808 bass kick with pitch glide, minimal sinister hi-hat rhythm, tight hard snare, dark cold drill percussion arrangement",
        "heavy off-beat 808 kicks with slides, staccato cold minimal hi-hats, hard punchy snare, sparse aggressive drill drum groove",
    ],
    "hip hop": [
        "punchy sampled kick on beats 1 and 3, crisp snare on beats 2 and 4, swinging 16th-note hi-hats with groove, deep hip hop drum pocket",
        "hard knocking boom bap-influenced kick and snare, loose swinging hi-hats with shuffle, organic drum feel, deep pocket groove",
        "heavy sampled kick, cracking snare with snap, swinging syncopated hi-hats, boom bap hip hop rhythm section",
    ],
    "hip-hop": [
        "punchy sampled kick on beats 1 and 3, crisp snare on beats 2 and 4, swinging 16th-note hi-hats, hip hop drum groove",
        "hard knocking kick and snare, loose swinging hi-hats, deep pocket hip hop rhythm section",
    ],
    "boom bap": [
        "deep sampled kick on beats 1 and 3, hard rimshot snare on beats 2 and 4, loose swinging hi-hats with ghost notes, dusty boom bap drum break",
        "heavy sampled drum break, hard punchy kick, cracking rimshot snare, swinging hi-hat choke pattern, classic golden era boom bap groove",
    ],
    "phonk": [
        "heavy distorted 808 kick with sub rumble, rapid rolling hi-hat triplets, hard snare slam, cowbell accents on upbeats, Memphis phonk percussion",
        "crunchy distorted 808 bass drum, rapid hi-hat rolls with cowbell pattern, punchy snare hit, raw distorted phonk percussion arrangement",
    ],
    "house": [
        "four-on-the-floor kick hitting every single beat, open hi-hat on every offbeat sixteenth note, crisp clap or snare on beats 2 and 4, driving house drum pattern",
        "stomping kick drum on every quarter note, syncopated open hi-hat accents, punchy clap on backbeats, classic house music percussion",
    ],
    "deep house": [
        "soft four-on-the-floor kick with warm attack, subtle open hi-hat offbeats, soft clap on beats 2 and 4, deep house groove",
        "rolling kick drum with soft attack, gentle swinging hi-hat groove, subtle clap, hypnotic deep house drum pattern",
    ],
    "tech house": [
        "driving punchy kick every single beat, percussive hi-hat groove with rim shot accents, industrial mechanical rhythm, relentless tech house percussion",
        "heavy four-on-the-floor kick, aggressive syncopated hi-hat and rim percussion, relentless driving tech house drum arrangement",
    ],
    "techno": [
        "crushing industrial kick drum on every beat at 140+ BPM, aggressive 16th-note hi-hats, heavy metallic percussion, relentless mechanical techno rhythm",
        "hard pounding kick every beat, rapid mechanical hi-hat rhythm, industrial percussion, brutal techno drum arrangement",
    ],
    "edm": [
        "four-on-the-floor kick at 128 BPM, sharp clap on beats 2 and 4, build-up snare roll, driving EDM drum pattern into explosive drop",
        "stomping kick every quarter note, punchy clap on backbeats, pre-drop snare fill, powerful festival EDM percussion",
    ],
    "dubstep": [
        "half-time kick on beat 1 at 140 BPM, enormous reverb-soaked snare on beat 3 only, sparse syncopated hi-hats, half-time dubstep drum groove",
        "massive booming snare hit on beat 3, half-time kick drum pattern, minimal sparse hi-hats, devastating dubstep half-time percussion",
    ],
    "drum and bass": [
        "170 BPM syncopated amen breakbeat, fast rolling kick and sharp snare in syncopated pattern, rapid 16th-note hi-hat drive, relentless DnB momentum",
        "frantic breakbeat at 170 BPM, syncopated kick on beat 1 and off-beat, sharp fast snare rolls, amen break-influenced drum and bass",
    ],
    "dnb": [
        "170 BPM rolling amen-style breakbeat, syncopated kick and snare, rapid hi-hat, relentless DnB drum pattern",
        "fast amen break at 170 BPM, syncopated kick pattern, sharp snare hits, rapid hi-hat rolls, energetic DnB percussion",
    ],
    "jungle": [
        "frantic chopped amen break, complex syncopated kick and snare layering, rapid hi-hat patterns, raw 90s jungle drum arrangement",
        "breakbeat complexity with chopped amen samples, fast syncopated percussion, raw frantic jungle rhythm",
    ],
    "afrobeats": [
        "syncopated afrobeats kick pattern, talking drum accents, layered conga and shaker groove, off-beat hi-hat, vibrant West African percussion",
        "complex afrobeats polyrhythm, layered hand percussion and talking drum, syncopated kick, tambourine and shaker, vibrant afro rhythm",
    ],
    "reggaeton": [
        "dembow kick on beat 1 and the-and-of-beat-3, snare on beats 2 and 4, shuffled hi-hat groove, classic reggaeton rhythm section",
        "punchy dembow pattern, kick on 1 and and-3, hard snare on backbeats, shuffled hi-hat, aggressive Latin percussion",
    ],
    "r&b": [
        "smooth kick and snare pocket groove, shuffled hi-hat with ghost notes, polished R&B drum arrangement",
        "laid-back kick on 1 and 3, snare on 2 and 4 with ghost note fills, swinging hi-hat groove, soulful R&B percussion",
    ],
    "lo-fi": [
        "dusty muffled kick drum with soft attack, soft brushed snare with vinyl grain, loose dusty hi-hats, laid-back lo-fi drum groove",
        "muted kick with soft attack, brushed snare decay, dusty vinyl hi-hats, relaxed lo-fi drum pattern with human feel",
    ],
    "funk": [
        "syncopated funk kick on the one, tight ghost-note snare hits, open hi-hat choke pattern, deep pocket groove, James Brown funk percussion",
        "driving funky kick drum with syncopation, crisp snare with ghost notes, open hi-hat rhythm, tight pocket funk arrangement",
    ],
    "reggae": [
        "one-drop kick on beat 3 only, rim shot on beats 2 and 4, open hi-hat on offbeats, classic reggae one-drop drum pattern",
        "roots reggae one-drop kick, syncopated rim shot, offbeat hi-hat chop, deep reggae groove",
    ],
    "grime": [
        "staccato 140 BPM kick pattern, sharp metallic hi-hats, hard snare hit, cold rhythmic grime percussion",
        "aggressive 140 BPM drum arrangement, rapid hi-hat rhythm, punchy cold snare, dark grime beat percussion",
    ],
    "uk garage": [
        "2-step garage shuffle kick on beat 1 and and-beat-3, snare on 2 and 4 with shuffle, syncopated swinging hi-hat, UK garage 2-step drum pattern",
        "shuffled 2-step kick pattern, swinging hi-hats, backbeat snare, bouncy UK garage percussion arrangement",
    ],
    "synthwave": [
        "driving electronic drum machine, four-on-the-floor kick, massive gated reverb snare on beats 2 and 4, 16th-note closed hi-hats, 80s drum machine pattern",
        "electronic drum machine pattern, punchy kick every beat, heavy gated reverb snare, 16th-note hi-hat drive, retro synthwave percussion",
    ],
    "hyperpop": [
        "distorted glitchy drum pattern, heavy distorted 808 kick, glitched stutter snare effects, chaotic hi-hat fills, hyperpop percussion",
        "aggressive glitched-out drums, heavy distorted kick, stutter snare, chaotic energetic drum arrangement",
    ],
    "pluggnb": [
        "slow melodic trap drums, heavy 808 sub kick, soft snare on beat 3, gentle minimal hi-hat pattern, slow pluggnb percussion",
        "laid-back slow trap groove, deep 808 kick, minimal hi-hats, slow relaxed pluggnb drum arrangement",
    ],
    "jersey club": [
        "frantic 4x4 kick at 130+ BPM, rapid hi-hat chopping, syncopated snare hits, chaotic jersey club percussion",
        "fast four-on-the-floor kick, chopped percussion pattern, rapid hi-hat groove, aggressive jersey club drum arrangement",
    ],
    "cloud rap": [
        "slow minimal trap drums, sparse reverb-soaked kick and distant snare, soft hi-hats, dreamy cloud rap percussion",
        "minimal slow trap pattern, muffled kick, distant reverb snare, airy cloud rap drum groove",
    ],
    "jazz": [
        "swinging jazz ride cymbal pattern, brushed snare with ghost notes, kick on downbeats, hi-hat on 2 and 4, bebop drum groove",
        "live jazz drum kit with swing feel, ride cymbal rhythm, brushed snare, loose swing hi-hat choke, jazz percussion",
    ],
    "soul": [
        "tight soul groove kick on 1 and 3, snare on 2 and 4, shuffled hi-hat, gospel-influenced drum pocket",
        "soulful drum pocket, punchy kick and snare with swing, layered percussion, warm soul music drum arrangement",
    ],
    "pop": [
        "punchy kick on 1 and 3, sharp snare on 2 and 4, driving 16th-note hi-hats, modern pop drum arrangement",
        "four-on-the-floor or 1-and-3 kick, crisp clap on backbeats, energetic pop percussion",
    ],
    "cinematic": [
        "orchestral percussion with powerful timpani hits, taiko drums, dramatic snare rolls, cinematic tension percussion",
        "epic timpani and taiko drum hits, dramatic orchestral percussion build, powerful cinematic rhythm section",
    ],
    "rock": [
        "powerful live drum kit, hard kick on beats 1 and 3, crashing snare on 2 and 4, driving 8th-note hi-hats, rock drum groove",
        "driving rock drums, punchy kick, cracking snare on 2 and 4, aggressive hi-hat pattern, live drum sound",
    ],
    "metal": [
        "thunderous double-kick drum blast beats, crashing snare on 2 and 4, rapid 16th-note hi-hats, metal drum arrangement",
        "blastbeat double-kick pattern, alternating kick and snare, rapid percussion, crushing metal drum groove",
    ],
    "dancehall": [
        "digital riddim kick on beat 1 and and-3, snare on backbeats, offbeat hi-hat chop, Caribbean dancehall percussion",
        "punchy dancehall kick and snare rhythm, offbeat hi-hat, energetic Caribbean drum arrangement",
    ],
    "amapiano": [
        "deep log drum bass kick on every beat, layered percussive piano chords, shuffled hi-hat, syncopated percussion, amapiano drum groove",
        "punchy log drum kick pattern, rolling hi-hat groove, syncopated percussion layers, South African amapiano rhythm",
    ],
    "afro house": [
        "driving four-on-the-floor kick, layered tribal conga and shaker percussion, open hi-hat offbeats, afro house drum pattern",
        "stomping kick every beat, complex afro percussion layers, tribal drum groove, afro house rhythm section",
    ],
    "future bass": [
        "driving electronic kick, powerful clap on backbeats, energetic hi-hat drive, festival future bass drum pattern",
        "punchy electronic drums with driving kick, emotive clap on backbeats, aggressive hi-hat, future bass percussion",
    ],
    "trap soul": [
        "slow trap drums, heavy 808 sub kick, minimal gentle hi-hats, soft snare on beat 3, slow trap soul percussion",
        "slow heavy 808 kicks, gentle rolling hi-hats, minimal snare, slow intimate trap soul drum groove",
    ],
    "latin": [
        "layered conga and bongo patterns, timbales, shaker and claves, complex Latin polyrhythmic percussion",
        "vibrant Latin percussion section: congas, timbales, shakers, bass drum, energetic layered Latin rhythm",
    ],
    "blues": [
        "shuffle rhythm kick and snare, driving 12-bar groove, swinging hi-hats, blues drum arrangement",
        "steady kick and snare groove, swinging hi-hats with shuffle feel, blues percussion",
    ],
    "gospel": [
        "driving kick and snare groove, powerful gospel percussion, uplifting rhythm section",
        "punchy gospel kick and snare, driving rhythm, powerful spiritual drum arrangement",
    ],
    "funk": [
        "syncopated funk kick on the one, tight ghost-note snare, open hi-hat choke, deep pocket groove",
        "funky kick drum with syncopation, crisp snare ghost notes, open hi-hat rhythm, tight funk groove",
    ],
    "vaporwave": [
        "slow hazy drum machine pattern, soft muffled kick, gentle snare, minimal vaporwave percussion",
        "slow drum machine beat, soft kick and snare, minimal dreamy vaporwave drum texture",
    ],
}

# ── Genre+mood combination tags ───────────────────────────────────────────────
# When the user types e.g. "sad trap" or "dark drill", these give ACE-Step
# a precise combined descriptor that nails the emotional + genre character.
_GENRE_MOOD_COMBO = {
    ("trap", "sad"):       "sad emotional trap, minor key piano melody over heavy trap drums and 808, tearful introspective trap energy",
    ("trap", "dark"):      "dark menacing trap, ominous minor atmosphere, cold threatening 808 bass, dark brooding trap production",
    ("trap", "melodic"):   "melodic emotional trap, beautiful piano hook over heavy 808 kicks and rolling hi-hats, harmonic dark trap",
    ("trap", "emotional"): "emotional melodic trap, raw emotional piano loop, heavy 808 sub kicks, heartfelt sad trap energy",
    ("trap", "hard"):      "hard aggressive trap, heavy pounding 808 kicks, rapid hi-hat triplets, aggressive menacing atmosphere",
    ("trap", "aggressive"):"aggressive hard trap, explosive 808, relentless hi-hat rolls, dark threatening hard trap production",
    ("trap", "chill"):     "chill melodic trap, relaxed rolling hi-hats, smooth 808 groove, ambient dark atmosphere",
    ("drill", "dark"):     "dark cold drill, sinister ominous atmosphere, deep sliding 808, cold minimal dark drill energy",
    ("drill", "sad"):      "sad melodic drill, emotional minor melody over cold 808 slides and sparse hi-hats, melancholic drill",
    ("drill", "melodic"):  "melodic drill, emotional melody over dark sliding 808 and cold hi-hats, harmonic cold drill",
    ("drill", "aggressive"):"aggressive hard drill, cold brutal 808, sharp staccato hi-hats, menacing dark aggressive drill",
    ("hip hop", "dark"):   "dark underground hip hop, cinematic dark samples, heavy knocking kick and snare, dark brooding atmosphere",
    ("hip hop", "sad"):    "sad emotional hip hop, melancholic soul sample, swinging drum groove, introspective heartbreak",
    ("hip hop", "hard"):   "hard-hitting hip hop, heavy punchy kick and snare, dark samples, aggressive street energy",
    ("boom bap", "dark"):  "dark boom bap, ominous sample, heavy kick and rimshot snare, dusty dark underground feel",
    ("boom bap", "sad"):   "sad boom bap, melancholic soul chop, swinging drum break, emotional introspective hip hop",
    ("phonk", "dark"):     "dark aggressive phonk, distorted 808, cowbell percussion, sinister dark atmosphere",
    ("phonk", "hard"):     "hard aggressive phonk, heavy distorted kick, rapid hi-hat rolls, brutal dark phonk energy",
    ("phonk", "melodic"):  "melodic phonk, haunting emotional sample over distorted 808 and cowbell, melodic dark phonk",
    ("r&b", "dark"):       "dark R&B, minor key emotional chord progressions, moody bass groove, dark intimate atmosphere",
    ("r&b", "sad"):        "sad R&B, heartbreaking emotional melody, slow groove, minor key vulnerability, soulful sadness",
    ("r&b", "melodic"):    "melodic R&B, beautiful chord progressions, emotional lead melody, rich harmonic R&B texture",
    ("dubstep", "dark"):   "dark dubstep, ominous cinematic tension, industrial bass growl, dystopian half-time aggression",
    ("dubstep", "melodic"):"melodic dubstep, soaring emotional synth lead, cathartic massive drop, beautiful dubstep energy",
    ("house", "dark"):     "dark house, minor key atmospheric pads, driving four-on-the-floor kick, dark underground club energy",
    ("techno", "dark"):    "dark Berlin techno, cold industrial atmosphere, crushing kick, hypnotic mechanical dark energy",
    ("edm", "emotional"):  "emotional EDM, euphoric uplifting chord progression, soaring synth lead, cathartic massive drop",
    ("synthwave", "dark"): "dark synthwave, cold dystopian synth, Blade Runner atmosphere, driving dark electronic rhythm",
    ("synthwave", "sad"):  "sad synthwave, melancholic minor key synth, 80s nostalgic sadness, emotional retro atmosphere",
    ("lo-fi", "sad"):      "sad lo-fi, melancholic piano, dusty vinyl crackle, lonely late-night sadness, soft brushed drums",
    ("cinematic", "dark"): "dark cinematic score, ominous orchestral tension, dissonant brass, tense horror atmosphere",
    ("cinematic", "epic"): "epic cinematic score, heroic orchestral power, sweeping strings, massive percussion, triumphant",
    ("cinematic", "sad"):  "sad emotional cinematic, weeping strings, solo piano, heartbreaking orchestral melody",
    ("cloud rap", "sad"):  "sad cloud rap, lonely atmospheric trap, reverb-drenched emotional melody, hollow distant feel",
    ("pluggnb", "sad"):    "sad pluggnb, slow emotional trap drums, minor key piano, lonely dark atmosphere, heartbreak",
    ("jazz", "dark"):      "dark jazz, dissonant noir atmosphere, minor key chords, sparse brooding jazz percussion",
    ("rock", "dark"):      "dark rock, heavy distorted guitar, minor key brooding atmosphere, powerful live drums",
}

# ── Default BPM hint per genre (injected when user doesn't specify BPM) ──────
# These steer ACE-Step toward the right tempo range for the genre.
_GENRE_DEFAULT_BPM = {
    "trap":          "130-145 BPM",
    "drill":         "140-150 BPM",
    "hip hop":       "85-100 BPM",
    "hip-hop":       "85-100 BPM",
    "boom bap":      "85-95 BPM",
    "rap":           "90-110 BPM",
    "phonk":         "130-145 BPM",
    "cloud rap":     "120-140 BPM",
    "pluggnb":       "120-140 BPM",
    "trap soul":     "60-80 BPM",
    "r&b":           "70-95 BPM",
    "soul":          "70-95 BPM",
    "funk":          "95-115 BPM",
    "gospel":        "75-100 BPM",
    "house":         "124-130 BPM",
    "deep house":    "120-126 BPM",
    "tech house":    "126-132 BPM",
    "techno":        "135-150 BPM",
    "edm":           "126-132 BPM",
    "dubstep":       "138-142 BPM",
    "drum and bass": "168-175 BPM",
    "dnb":           "168-175 BPM",
    "jungle":        "155-170 BPM",
    "electronic":    "120-140 BPM",
    "grime":         "138-142 BPM",
    "uk garage":     "130-135 BPM",
    "jersey club":   "130-140 BPM",
    "hyperpop":      "140-160 BPM",
    "afrobeats":     "98-110 BPM",
    "amapiano":      "108-114 BPM",
    "afro house":    "120-126 BPM",
    "dancehall":     "90-110 BPM",
    "reggaeton":     "90-100 BPM",
    "reggae":        "75-90 BPM",
    "latin":         "95-130 BPM",
    "synthwave":     "110-130 BPM",
    "vaporwave":     "75-85 BPM",
    "lo-fi":         "70-90 BPM",
    "jazz":          "120-180 BPM",  # jazz swing tempo
    "blues":         "70-90 BPM",
    "pop":           "100-130 BPM",
    "rock":          "110-145 BPM",
    "metal":         "140-200 BPM",
    "cinematic":     "60-120 BPM",
    "ambient":       "60-80 BPM",
    "future bass":   "140-150 BPM",
}

# Per-genre energy descriptors — one picked randomly and injected into the prompt
_GENRE_ENERGY_TAGS = {
    "dubstep":       ["massive sub-bass pressure, crushing reese wobble, half-time fury",
                      "face-melting bass modulation, filthy growl bass, devastating drop energy",
                      "reese bass destruction, crushing kick, overwhelming sub-bass weight",
                      "filthy wobble bass, relentless half-time groove, bone-crushing drop"],
    "drum and bass": ["170 BPM relentless energy, frantic rolling breaks, sub-bass speed",
                      "amen break intensity, fast driving momentum, deep sub pressure",
                      "rapid breakbeat drive, powerful sub bass, relentless forward energy"],
    "dnb":           ["170 BPM relentless energy, frantic rolling breaks, sub-bass speed",
                      "amen break intensity, fast driving momentum, deep sub pressure",
                      "rapid breakbeat drive, powerful sub bass, relentless forward energy"],
    "jungle":        ["frantic chopped amen, reggae bass wobble, raw rhythmic intensity",
                      "complex breakbeat patterns, deep bass pressure, underground raw power"],
    "techno":        ["relentless driving force, crushing kick pressure, hypnotic dark intensity",
                      "industrial mechanical energy, peak-hour power, cold relentless groove",
                      "dystopian kick drive, pounding rhythm, dark hypnotic forward momentum"],
    "tech house":    ["hypnotic bass-driven intensity, relentless groove pressure, peak-hour energy",
                      "driving mechanical force, thick bass stab, pounding underground groove"],
    "edm":           ["euphoric crowd energy, massive drop impact, stadium-filling power",
                      "epic build tension release, soaring festival energy, huge crowd moment",
                      "explosive drop energy, massive synth chord stabs, overwhelming crowd power"],
    "house":         ["four-on-the-floor driving power, peak-hour dancefloor energy, full layers",
                      "crowd-lifting groove force, stomping kick drive, soulful energetic arrangement"],
    "deep house":    ["deep pulsing bass groove, warm driving energy, immersive dancefloor power"],
    "electronic":    ["full-spectrum energy, layered modular intensity, driving electronic force",
                      "complex high-energy production, powerful synth layers, relentless drive"],
    "hyperpop":      ["chaotic maximum energy, distorted glitch intensity, maximalist overwhelming sound",
                      "explosive glitch energy, heavy distorted 808, hyper-saturated production"],
    "trap":          ["rolling hi-hat intensity, heavy sub-bass pressure, dark atmospheric energy",
                      "punchy 808 weight, relentless trap groove, dense layered atmosphere"],
    "drill":         ["cold menacing energy, dark aggressive intensity, heavy slide 808 weight",
                      "relentless hard drive, ominous bass presence, cold brutal groove"],
    "hip hop":       ["head-nodding groove power, punchy drum impact, deep bass pocket",
                      "soulful harmonic richness, knocking beat momentum, full groove"],
    "boom bap":      ["knocking drum power, punchy sample chop, deep groove momentum",
                      "heavy drum knock, dusty soul energy, full boom bap groove"],
    "phonk":         ["dark distorted energy, heavy 808 aggression, Memphis raw power",
                      "chopped soul intensity, distorted bass, dark phonk groove force"],
    "grime":         ["staccato aggressive energy, cold UK intensity, hard-hitting dark power",
                      "relentless grime groove, cold stabs, aggressive forward drive"],
    "rock":          ["driving guitar power, explosive live energy, wall of sound intensity",
                      "full band arrangement, powerful drums, distorted guitar layers"],
    "metal":         ["crushing heavy intensity, thunderous drum power, aggressive riff energy",
                      "wall of distorted guitars, blastbeat power, overwhelming heavy force"],
    "afrobeats":     ["infectious afro groove, layered percussion energy, vibrant full arrangement",
                      "driving afrobeats rhythm, melodic hook energy, full percussive power"],
    "reggaeton":     ["punchy dembow drive, thick bass presence, energetic Latin groove",
                      "heavy dembow rhythm, urban bass energy, full reggaeton power"],
    "latin":         ["vibrant layered percussion, brass and rhythm energy, full Latin arrangement"],
    "funk":          ["deep funky groove, slapped bass power, full band funk energy",
                      "tight syncopated rhythm, layered funk arrangement, groove-driven force"],
    "soul":          ["deep soulful groove, full band arrangement, warm powerful energy",
                      "layered soul production, gospel energy, rich harmonic power"],
    "r&b":           ["smooth powerful groove, layered R&B arrangement, warm bass energy",
                      "rich harmonic depth, full R&B production, punchy groove force"],
    "synthwave":     ["pulsing retro energy, driving analog synth power, neon-lit momentum",
                      "relentless synthwave drive, layered synth stacks, 80s peak energy"],
    "pop":           ["hooky powerful energy, full polished arrangement, driving pop momentum",
                      "layered pop production, punchy mix, catchy energetic arrangement"],
    "cinematic":     ["sweeping dramatic power, full orchestral energy, cinematic climax force",
                      "building tension and release, massive cinematic drop, overwhelming score energy"],
}

# Guidance scale range (min, max) per genre.
# A random value is sampled each generation so beats stay varied.
# Higher values = ACE-Step sticks tighter to the genre description / more energy.
# All ranges are intentionally high — beats should be full and energetic by default.
_GENRE_GUIDANCE_RANGE = {
    # ── Electronic / club ──────────────────────────────────────────────────
    "dubstep":       (9.5, 10.0),
    "drum and bass": (9.5, 10.0),
    "dnb":           (9.5, 10.0),
    "jungle":        (9.0, 10.0),
    "techno":        (9.5, 10.0),
    "tech house":    (9.0, 10.0),
    "edm":           (9.5, 10.0),
    "house":         (9.0, 10.0),
    "deep house":    (8.5,  9.5),
    "electronic":    (9.0, 10.0),
    "hyperpop":      (9.5, 10.0),
    "jersey club":   (9.0, 10.0),
    "uk garage":     (8.5,  9.5),
    "grime":         (9.0, 10.0),
    # ── Synthwave / retro ──────────────────────────────────────────────────
    "synthwave":     (8.5,  9.5),
    "vaporwave":     (7.5,  8.5),
    # ── Hip-hop / rap ──────────────────────────────────────────────────────
    "trap":          (8.5,  9.5),
    "drill":         (8.5,  9.5),
    "hip hop":       (8.0,  9.0),
    "hip-hop":       (8.0,  9.0),
    "rap":           (8.0,  9.0),
    "boom bap":      (8.0,  9.0),
    "phonk":         (8.5,  9.5),
    "cloud rap":     (7.5,  8.5),
    "pluggnb":       (7.5,  8.5),
    # ── R&B / Soul / Funk ──────────────────────────────────────────────────
    "r&b":           (8.0,  9.0),
    "soul":          (8.0,  9.0),
    "funk":          (8.5,  9.5),
    "gospel":        (8.5,  9.5),
    # ── Afro / Caribbean / Latin ───────────────────────────────────────────
    "afrobeats":     (8.5,  9.5),
    "dancehall":     (8.5,  9.5),
    "reggaeton":     (8.5,  9.5),
    "reggae":        (8.0,  9.0),
    "latin":         (8.5,  9.5),
    # ── Live instruments ───────────────────────────────────────────────────
    "rock":          (9.0, 10.0),
    "metal":         (9.5, 10.0),
    "blues":         (8.0,  9.0),
    "jazz":          (7.5,  8.5),
    # ── Atmospheric / chill ────────────────────────────────────────────────
    "lo-fi":         (7.0,  8.0),
    "ambient":       (6.5,  7.5),
    "cinematic":     (8.0,  9.0),
    # ── Pop ────────────────────────────────────────────────────────────────
    "pop":           (8.0,  9.0),
}

# Tags injected when user specifies "energic" / "energetic" in their prompt
_ENERGIC_BOOST_TAGS = [
    "maximum energy, relentless intensity, explosive powerful production",
    "full power, aggressive high-energy, overwhelming sonic force",
    "peak energy, driving relentless momentum, intense high-impact sound",
    "maximum intensity, hard-hitting explosive production, full-throttle energy",
]

# Explicit instrument descriptors — injected when user mentions a specific instrument.
# Each keyword maps to a LIST of variants so the same input sounds different each time.
# These go FIRST in the tag string so ACE-Step can't miss them.
_INSTRUMENT_FORCE_TAGS = {
    "acoustic guitar": [
        "fingerpicked acoustic guitar with warm room reverb, intimate string resonance",
        "open-tuned steel-string acoustic guitar, rich warm sustain",
        "strummed acoustic guitar with natural body resonance, organic texture",
        "fingerstyle acoustic guitar picking, delicate string tone, close-mic warmth",
        "percussive acoustic guitar strumming, rhythmic body taps, raw organic feel",
    ],
    "guitar": [
        "electric guitar lead melody, prominent guitar riff, guitar-driven sound",
        "distorted electric guitar with chorus and reverb, powerful guitar tone",
        "clean Fender Stratocaster guitar lead, glassy bright tone",
        "warm Les Paul electric guitar, thick sustained melody",
        "wah-pedal funk guitar, expressive tone, guitar-forward mix",
        "slide guitar with heavy reverb, bluesy guitar lead",
        "tremolo electric guitar, shimmering tone, guitar-driven melody",
    ],
    "dark piano": [
        "dark minor-key piano, heavy resonant bass notes, ominous piano melody",
        "detuned dark piano with reverb tail, haunting piano atmosphere",
        "deep dark piano chords, dissonant minor harmony, brooding piano lead",
        "dark prepared piano with eerie overtones, unsettling piano texture",
        "slow dark piano melody, heavy sustain pedal, cinematic dark piano",
    ],
    "smooth piano": [
        "smooth jazz piano comping, warm Rhodes-like tone, silky piano lead",
        "gentle smooth piano melody, soft touch, warm harmonic piano texture",
        "smooth laid-back piano groove, sophisticated chord voicings, velvet tone",
        "soft smooth piano with light reverb, mellow melodic piano line",
        "lush smooth piano arpeggios, warm sustain, elegant piano phrasing",
    ],
    "piano": [
        "prominent acoustic piano lead, piano-driven melody, rich piano tone",
        "expressive concert grand piano, dynamic piano melody, resonant strings",
        "intimate upright piano, warm woody tone, piano-forward mix",
        "bright studio piano, crisp attack, melodic piano hook",
        "emotional piano ballad style, sensitive touch, singing piano line",
        "dramatic piano with strong left hand, full piano arrangement",
    ],
    "synth": [
        "prominent synthesizer lead melody, synth-driven hook, bright analog synth",
        "fat detuned supersaw synth lead, wide stereo synth melody",
        "warm Moog-style synth, smooth resonant lead, synth-forward mix",
        "plucky arpeggio synth with filter envelope, rhythmic synth line",
        "glassy FM synthesis lead, bell-like synth melody, bright digital tone",
        "lush pad synth with slow attack, atmospheric synth lead, evolving tone",
    ],
    "bass": [
        "prominent deep bass line, bass-forward mix, punchy sub groove",
        "slapped Fender bass with funk groove, prominent bass melody",
        "warm upright double bass, prominent bass, acoustic bass texture",
        "filtered acid bass with resonance, prominent bass modulation",
        "smooth fretless bass slide, prominent bass lead, melodic bass line",
    ],
    "violin": [
        "solo violin lead melody, expressive vibrato, prominent violin",
        "emotional violin with lush reverb, singing violin tone",
        "pizzicato violin accents over sustained melody, prominent violin",
        "raw folk violin, rich overtones, violin-driven melody",
    ],
    "strings": [
        "lush orchestral string section with vibrato, full strings arrangement",
        "sweeping string ensemble, rich harmonic strings, prominent string layer",
        "tremolo string texture, building tension, prominent string arrangement",
        "intimate string quartet, warm close-mic strings, melodic string lead",
    ],
    "brass": [
        "prominent brass section, punchy horn stabs, brass-forward mix",
        "smooth jazz trumpet lead, warm brass melody, prominent horn",
        "bold trombone melody, rich brass tone, brass-driven arrangement",
        "staccato brass stabs, energetic horn accents, prominent brass layer",
    ],
    "flute": [
        "breathy solo flute melody, prominent flute, woodwind-driven sound",
        "expressive flute with vibrato, warm flute tone, flute lead melody",
        "alto flute, dark warm tone, prominent low flute melody",
    ],
    "saxophone": [
        "tenor saxophone melody, prominent sax lead, jazz saxophone tone",
        "alto saxophone with breathy tone, expressive sax melody, sax-forward",
        "smooth saxophone solo, warm vibrato, prominent sax in the mix",
    ],
    "organ": [
        "Hammond B3 organ with rotary speaker, prominent organ groove",
        "gospel organ with full chords, organ-driven arrangement, warm organ",
        "jazz organ comping, swirling rotary effect, prominent organ melody",
    ],
    "drums": [
        "prominent drum kit, drum-forward mix, powerful live drum groove",
        "heavy live drums with room reverb, dominant drum sound, driving rhythm",
        "tight studio drum kit, punchy snare, prominent drum mix",
    ],
}

def _get_guidance_scale(genre: str, prompt: str = "") -> float:
    """Return a randomly sampled guidance scale for the given genre.
    If the user explicitly asked for 'energic'/'energetic', pin to 10.0.
    """
    p = prompt.lower() if prompt else ""
    if "energic" in p or "energetic" in p:
        return 10.0
    lo, hi = _GENRE_GUIDANCE_RANGE.get(genre, (7.5, 8.5))
    return round(random.uniform(lo, hi), 2)

# Melodic stem to extract per genre (ACE-Step extract instruction stem name)
GENRE_MELODY_STEM = {
    "trap":          "synth",
    "drill":         "synth",
    "hip hop":       "piano",
    "hip-hop":       "piano",
    "rap":           "piano",
    "boom bap":      "piano",
    "r&b":           "piano",
    "afrobeats":     "keyboard",
    "dancehall":     "keyboard",
    "lo-fi":         "piano",
    "electronic":    "synth",
    "house":         "synth",
    "deep house":    "synth",
    "tech house":    "synth",
    "techno":        "synth",
    "edm":           "synth",
    "dubstep":       "synth",
    "drum and bass": "synth",
    "dnb":           "synth",
    "jungle":        "synth",
    "reggaeton":     "keyboard",
    "latin":         "keyboard",
    "jazz":          "piano",
    "soul":          "piano",
    "funk":          "guitar",
    "pop":           "piano",
    "cinematic":     "strings",
    "ambient":       "synth",
    "phonk":         "synth",
    "cloud rap":     "piano",
    "grime":         "synth",
    "uk garage":     "keyboard",
    "synthwave":     "synth",
    "vaporwave":     "synth",
    "hyperpop":      "synth",
    "pluggnb":       "piano",
    "jersey club":   "synth",
    "reggae":        "guitar",
    "blues":         "guitar",
    "rock":          "guitar",
    "gospel":        "piano",
}

# Keywords in user prompt → ACE-Step stem name to generate
# Piano and keyboard are separate stems so piano prompts get acoustic piano sound.
INSTRUMENT_STEM_MAP = [
    ("piano",           "piano"),       # acoustic piano
    ("melodic piano",   "piano"),
    ("sad piano",       "piano"),
    ("type beat",       "piano"),       # "type beat" usually implies piano/melody
    ("epiano",          "keyboard"),
    ("e-piano",         "keyboard"),
    ("rhodes",          "keyboard"),
    ("organ",           "keyboard"),
    ("keys",            "keyboard"),
    ("strings",         "strings"),
    ("violin",          "strings"),
    ("cello",           "strings"),
    ("viola",           "strings"),
    ("orchestra",       "strings"),
    ("orchestral",      "strings"),
    ("brass",           "brass"),
    ("trumpet",         "brass"),
    ("trombone",        "brass"),
    ("horn",            "brass"),
    ("guitar",          "guitar"),
    ("acoustic guitar", "guitar"),
    ("synth",           "synth"),
    ("synthesizer",     "synth"),
    ("lead synth",      "synth"),
    ("pad",             "synth"),
    ("choir",           "vocals"),
    ("woodwind",        "woodwinds"),
    ("flute",           "woodwinds"),
    ("clarinet",        "woodwinds"),
    ("sax",             "woodwinds"),
    ("saxophone",       "woodwinds"),
]

# Map ACE-Step stem name → studio instrument type
ACE_STEM_TO_STUDIO = {
    "drums":     "drums",
    "percs":     "drums",
    "bass":      "bass",
    "piano":     "piano",
    "keyboard":  "piano",
    "guitar":    "synth",
    "strings":   "pad",
    "synth":     "synth",
    "brass":     "synth",
    "woodwinds": "piano",
    "vocals":    "pad",
    "percussion":"drums",
}

def _drum_hint(style: str, user_prompt: str, genre: str) -> str:
    """Returns extra drum detail when user explicitly requests specific drum characteristics.
    Note: _GENRE_DRUM_TAGS is ALWAYS injected in build_tags() for the genre foundation.
    This function adds ADDITIONAL detail on top when user specifies drum keywords."""
    combined = (style + " " + user_prompt).lower()
    parts = []
    if any(x in combined for x in ["hard drum", "heavy drum", "aggressive drum", "banging drum"]):
        parts.append("hard-hitting aggressive drums, heavy kick and snare impact")
    elif any(x in combined for x in ["soft drum", "light drum", "minimal drum"]):
        parts.append("soft minimal drums, subtle percussion arrangement")
    if any(x in combined for x in ["hard kick", "heavy kick", "punchy kick", "808 kick"]):
        parts.append("heavy punchy kick drum with deep sub impact")
    if any(x in combined for x in ["rolling hi-hat", "fast hi-hat", "triplet hi-hat", "rapid hi-hat"]):
        parts.append("rolling triplet hi-hats at high speed")
    elif any(x in combined for x in ["soft hi-hat", "light hi-hat", "no hi-hat"]):
        parts.append("soft minimal hi-hats")
    if any(x in combined for x in ["snare roll", "snare build", "snare fill"]):
        parts.append("dramatic snare roll build-up")
    return ", ".join(parts) if parts else ""


# Harmonic color tags added randomly for chord variety
_HARMONIC_COLORS = [
    "minor 9th chord progression", "suspended 2nd and 4th chords",
    "major 7th chord voicings", "diminished passing chords",
    "dominant 7th chord resolution", "ii-V-I jazz movement",
    "chromatic bass line motion", "borrowed chords from parallel minor",
    "lydian mode bright sound", "phrygian dark modal feel",
    "dorian mode soulful groove", "pentatonic scale melody",
    "tritone substitution", "quartal harmony voicings",
]

# Specific instrument flavor descriptors — replaces generic "piano"/"synth" in prompt enrichment
_INSTRUMENT_FLAVORS = {
    "piano": [
        "melancholic Fender Rhodes with tremolo", "dusty upright acoustic piano",
        "warm vintage grand piano with room reverb", "soft intimate piano with pedal sustain",
        "cold digital piano with crystal clarity", "prepared piano with muffled hammer sound",
    ],
    "synth": [
        "warm analog Moog synth lead", "cold digital wavetable synthesizer",
        "bright JP-8 supersaw synth pad", "dark resonant filter sweep synth",
        "vintage Prophet-5 polysynth chords", "lush Juno chorus synth pad",
        "aggressive modular synth sequence", "glassy FM synthesis tones",
    ],
    "guitar": [
        "clean Stratocaster guitar with reverb", "warm hollow-body jazz guitar",
        "distorted electric guitar with chorus", "fingerpicked acoustic guitar",
        "wah-pedal funk guitar lick", "slide guitar with heavy reverb",
    ],
    "strings": [
        "lush orchestral string section with vibrato", "solo cello with intimate room reverb",
        "tremolo string ensemble", "pizzicato string accents",
        "string quartet with close microphone warmth",
    ],
    "bass": [
        "deep 808 sub bass with slow attack", "slapped Fender bass with punch",
        "warm upright double bass", "filtered acid bass with resonance",
        "distorted bass with overdrive grit", "smooth fretless bass slide",
    ],
    "keyboard": [
        "warm Wurlitzer electric piano", "vintage Hammond organ with rotary speaker",
        "clean Rhodes electric piano with chorus", "bright clavinet with wah filter",
    ],
}

# Production technique tags for realism and depth
_PRODUCTION_TECHNIQUES = [
    "sidechain compression pumping pads",
    "lush plate reverb tail on the snare",
    "tape saturation warmth on the mix",
    "lo-fi vinyl grain and tape hiss texture",
    "heavy room reverb on the drums",
    "stereo widening on the pads",
    "subtle pitch variation for human feel",
    "layered harmonic overtones",
    "punchy parallel compression on kick",
    "soft limiting for analog warmth",
    "delay throws on the melody",
    "filtered low-pass sweep on the pad",
    "transient shaping on the snare",
]


def build_tags(prompt: str, genre: str, mood: str, bpm: int, key: str) -> str:
    g = genre.lower().strip() if genre else ""
    m = mood.lower().strip() if mood else ""
    p_lower = prompt.lower() if prompt else ""

    is_energic = "energic" in p_lower or "energetic" in p_lower

    # ── Detect mood keywords embedded in the free-text prompt ─────────────────
    # e.g. user types "sad melodic dark piano trap energic" — extract sad/dark/melodic
    _PROMPT_MOOD_KEYWORDS = [
        "sad", "dark", "hard", "aggressive", "melodic", "emotional", "chill",
        "romantic", "angry", "epic", "dreamy", "nostalgic", "happy", "uplifting",
        "mysterious", "raw", "heavy", "intense", "lonely", "pain", "hyped", "bouncy",
    ]
    detected_moods = []
    for kw in _PROMPT_MOOD_KEYWORDS:
        if kw in p_lower and kw not in m:
            detected_moods.append(kw)

    parts = []

    # ── Explicit instrument override (FIRST so ACE-Step can't ignore it) ─────
    # Sorted longest-first: "acoustic guitar" before "guitar", "dark piano" before "piano"
    for kw in sorted(_INSTRUMENT_FORCE_TAGS.keys(), key=len, reverse=True):
        if kw in p_lower:
            parts.append(random.choice(_INSTRUMENT_FORCE_TAGS[kw]))
            break

    # ── Genre+mood combo tag (high priority — very specific combined descriptor) ─
    # Try primary mood first, then detected moods from prompt
    combo_inserted = False
    for mood_kw in ([m] + detected_moods):
        combo = _GENRE_MOOD_COMBO.get((g, mood_kw))
        if combo:
            parts.append(combo)
            combo_inserted = True
            break

    # ── Genre variant ─────────────────────────────────────────────────────────
    genre_variants = GENRE_VARIANTS.get(g)
    if genre_variants:
        parts.append(random.choice(genre_variants))
    elif g:
        parts.append(g)

    # ── Genre drum tags — ALWAYS injected so genre rhythm is never lost ───────
    drum_tags = _GENRE_DRUM_TAGS.get(g)
    if drum_tags:
        parts.append(random.choice(drum_tags))

    # ── Extra drum hint for explicit drum keywords in prompt ──────────────────
    extra_drums = _drum_hint("", prompt or "", g)
    if extra_drums:
        parts.append(extra_drums)

    # ── Genre-specific energy tag ─────────────────────────────────────────────
    energy_tags = _GENRE_ENERGY_TAGS.get(g)
    if energy_tags:
        parts.append(random.choice(energy_tags))

    # ── "Energic" boost ───────────────────────────────────────────────────────
    if is_energic:
        parts.append(random.choice(_ENERGIC_BOOST_TAGS))
        parts.append("massive wall of sound, relentless high-energy arrangement, powerful full-mix impact")

    # ── Mood (explicit field) ─────────────────────────────────────────────────
    mood_variants = MOOD_VARIANTS.get(m)
    if mood_variants:
        parts.append(random.choice(mood_variants))
    elif m:
        parts.append(m)

    # ── Detected moods from user prompt text (pick up to 2) ───────────────────
    moods_added = 0
    for dm in detected_moods:
        if moods_added >= 2:
            break
        variants = MOOD_VARIANTS.get(dm)
        if variants:
            parts.append(random.choice(variants))
            moods_added += 1

    # ── User prompt (raw) ─────────────────────────────────────────────────────
    if prompt:
        parts.append(prompt)

    # ── Atmospheric feel tag (60% chance) ─────────────────────────────────────
    if random.random() < 0.60:
        parts.append(random.choice(_FEEL_TAGS))

    # ── Mix fullness ──────────────────────────────────────────────────────────
    parts.append(random.choice(_FULLNESS_TAGS))

    # ── Production texture ────────────────────────────────────────────────────
    textures = _PRODUCTION_TEXTURES.get(g, _PRODUCTION_TEXTURES["_default"])
    parts.append(random.choice(textures))

    # ── Production technique ──────────────────────────────────────────────────
    parts.append(random.choice(_PRODUCTION_TECHNIQUES))

    # ── Harmonic color (40% chance) ───────────────────────────────────────────
    if random.random() < 0.40:
        parts.append(random.choice(_HARMONIC_COLORS))

    # ── Quality / BPM / key footer ────────────────────────────────────────────
    parts.append("instrumental, no vocals, no singing, no spoken words")
    if bpm:
        parts.append(f"{bpm} BPM")
    elif g in _GENRE_DEFAULT_BPM:
        # Inject a genre-appropriate BPM hint to steer ACE-Step's tempo
        parts.append(_GENRE_DEFAULT_BPM[g])
    if key:
        parts.append(f"key of {key}")
    if is_energic:
        parts.append("explosive high-energy production, full powerful mix, maximum impact and intensity")
    elif g in _ELECTRONIC_GENRES:
        parts.append("full energetic production, powerful mix, rich layered arrangement, high impact")
    else:
        parts.append("full energetic studio production, rich mix depth, powerful arrangement, emotionally expressive")

    # ── Deduplicate preserving order ──────────────────────────────────────────
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return ", ".join(out)


def _build_lyrics_structure(duration: float, user_prompt: str = "", genre: str = "") -> str:
    """
    Build a section structure for ACE-Step's arrangement dynamics.

    If the user mentions specific sections in the prompt (e.g. "2 verses 2 hooks
    with a bridge"), parse those and use them verbatim.  Otherwise pick a random
    structure that fits the duration so beats don't all follow the same pattern.

    For electronic genres (dubstep, techno, EDM, DnB, etc.) we use genre-specific
    section labels: [drop], [build], [breakdown] for more accurate energy dynamics.
    """
    g = genre.lower().strip() if genre else ""
    is_electronic = g in _ELECTRONIC_GENRES

    # ── Parse user-specified structure ──────────────────────────────────────
    p = user_prompt.lower()

    # Detect explicit section counts: "2 verses", "3 hooks", "1 bridge", etc.
    import re as _re
    def _count(pattern):
        m = _re.search(r'(\d+)\s*' + pattern, p)
        return int(m.group(1)) if m else None

    n_verse  = _count(r'verse') or _count(r'couplet')
    n_chorus = _count(r'(?:chorus|hook)')
    n_pre    = _count(r'pre.?(?:hook|chorus)')
    n_bridge = _count(r'bridge')
    n_outro  = _count(r'outro')
    n_intro  = _count(r'intro')
    # Electronic-specific keywords
    n_drop   = _count(r'drop')
    n_build  = _count(r'build')

    has_explicit = any(x is not None for x in [n_verse, n_chorus, n_pre, n_bridge, n_drop, n_build])

    if has_explicit:
        secs = []
        if n_intro is None or n_intro > 0:
            secs.append("[intro]")
        for i in range(n_verse or 1):
            if n_pre and n_pre > 0:
                secs.append("[pre-chorus]")
            secs.append("[verse]")
            if i < (n_chorus or 1):
                secs.append("[chorus]")
        for _ in range(max(0, (n_chorus or 1) - (n_verse or 1))):
            secs.append("[chorus]")
        for _ in range(n_bridge or 0):
            secs.append("[bridge]")
        if n_outro is None or n_outro > 0:
            secs.append("[outro]")
        return "\n[inst]\n".join(secs) + "\n[inst]"

    # ── Electronic genre section pools (drop/build/breakdown) ────────────────
    # These section labels tell ACE-Step to create proper energy dynamics
    if is_electronic:
        if duration <= 90:
            pool = [
                ["[intro]", "[build]", "[drop]", "[outro]"],
                ["[intro]", "[drop]", "[breakdown]", "[drop]"],
                ["[build]", "[drop]", "[breakdown]", "[drop]"],
            ]
        elif duration <= 130:
            pool = [
                ["[intro]", "[build]", "[drop]", "[breakdown]", "[drop]", "[outro]"],
                ["[intro]", "[drop]", "[breakdown]", "[build]", "[drop]", "[outro]"],
                ["[intro]", "[build]", "[drop]", "[build]", "[drop]", "[outro]"],
                ["[build]", "[drop]", "[breakdown]", "[drop]", "[breakdown]", "[outro]"],
                ["[intro]", "[breakdown]", "[build]", "[drop]", "[drop]", "[outro]"],
            ]
        elif duration <= 180:
            pool = [
                ["[intro]", "[build]", "[drop]", "[breakdown]", "[build]", "[drop]", "[outro]"],
                ["[intro]", "[drop]", "[breakdown]", "[build]", "[drop]", "[breakdown]", "[drop]", "[outro]"],
                ["[intro]", "[build]", "[drop]", "[build]", "[drop]", "[breakdown]", "[outro]"],
                ["[intro]", "[breakdown]", "[build]", "[drop]", "[drop]", "[breakdown]", "[drop]", "[outro]"],
                ["[build]", "[drop]", "[breakdown]", "[build]", "[drop]", "[breakdown]", "[drop]", "[outro]"],
                ["[intro]", "[build]", "[drop]", "[breakdown]", "[drop]", "[build]", "[drop]", "[outro]"],
            ]
        else:
            pool = [
                ["[intro]", "[build]", "[drop]", "[breakdown]", "[build]", "[drop]", "[breakdown]", "[drop]", "[outro]"],
                ["[intro]", "[drop]", "[breakdown]", "[build]", "[drop]", "[breakdown]", "[build]", "[drop]", "[outro]"],
                ["[intro]", "[build]", "[drop]", "[drop]", "[breakdown]", "[build]", "[drop]", "[breakdown]", "[outro]"],
                ["[build]", "[drop]", "[breakdown]", "[build]", "[drop]", "[drop]", "[breakdown]", "[drop]", "[outro]"],
                ["[intro]", "[build]", "[drop]", "[breakdown]", "[drop]", "[build]", "[drop]", "[outro]"],
            ]
        secs = random.choice(pool)
        return "\n[inst]\n".join(secs) + "\n[inst]"

    # ── Random structure pool by duration (non-electronic) ───────────────────
    # Each entry is a list of section tags; pick one randomly.
    if duration <= 90:
        pool = [
            ["[intro]", "[verse]", "[chorus]", "[outro]"],
            ["[verse]", "[chorus]", "[verse]", "[outro]"],
            ["[intro]", "[chorus]", "[verse]", "[chorus]"],
        ]
    elif duration <= 130:
        pool = [
            ["[intro]", "[verse]", "[chorus]", "[verse]", "[outro]"],
            ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[outro]"],
            ["[intro]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
            ["[verse]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
            ["[intro]", "[verse]", "[verse]", "[chorus]", "[chorus]", "[outro]"],
        ]
    elif duration <= 180:
        pool = [
            ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
            ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
            ["[intro]", "[chorus]", "[verse]", "[pre-chorus]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
            ["[intro]", "[verse]", "[chorus]", "[verse]", "[pre-chorus]", "[chorus]", "[outro]"],
            ["[intro]", "[verse]", "[verse]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
            ["[intro]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
        ]
    else:
        pool = [
            ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
            ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[verse]", "[bridge]", "[outro]"],
            ["[intro]", "[chorus]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
            ["[intro]", "[verse]", "[verse]", "[chorus]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"],
            ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"],
        ]

    secs = random.choice(pool)
    return "\n[inst]\n".join(secs) + "\n[inst]"


# Genres that include auxiliary percussion (congas, shakers, tambourine, etc.)
_PERC_GENRES = {
    "afrobeats", "reggaeton", "latin", "dancehall", "funk", "jazz", "boom bap",
    "trap", "hip hop", "hip-hop", "r&b", "soul", "phonk", "lo-fi", "grime",
}

# Keywords to REMOVE from caption when generating a specific stem
# Words to strip from the full-beat caption when building a stem caption.
# Prevents the stem prompt from mentioning rival instruments.
_STEM_EXCLUDE = {
    "drums":    ["bass", "guitar", "piano", "keyboard", "synth", "strings", "violin",
                 "cello", "flute", "trumpet", "saxophone", "sax", "melody", "chord",
                 "pad", "lead", "conga", "bongo", "shaker", "tambourine"],
    "percs":    ["bass", "guitar", "piano", "keyboard", "synth", "strings", "melody",
                 "chord", "pad", "lead", "kick", "snare", "hi-hat", "hihat"],
    "bass":     ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "guitar", "piano", "keyboard", "synth", "strings", "melody", "pad", "lead"],
    "piano":    ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "guitar", "synth", "pad", "keyboard", "strings"],
    "guitar":   ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "piano", "keyboard", "strings", "melody"],
    "keyboard": ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "guitar", "piano"],
    "synth":    ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "guitar", "piano", "keyboard", "strings"],
    "strings":  ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "guitar", "piano", "synth"],
    "brass":    ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "guitar", "piano", "synth", "strings"],
    "woodwinds":["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "clap", "conga",
                 "bass", "guitar", "synth", "strings"],
    "vocals":   ["drums", "kick", "snare", "hi-hat", "hihat", "cymbal", "bass",
                 "guitar", "piano", "synth", "strings"],
}

# Stem caption prefix: extremely explicit isolation language.
# High guidance_scale (used below) keeps the model anchored to these words.
_STEM_FOCUS = {
    "drums":    (
        "isolated drum track only, pure rhythm section, kick drum snare hi-hat cymbal, "
        "absolutely no bass no guitar no piano no synth no melody no harmony no chords, "
        "only percussion beats, silent everything except drums"
    ),
    "percs":    (
        "isolated auxiliary percussion only, congas bongos shakers tambourine cowbell clap, "
        "absolutely no kick drum no snare no hi-hat no bass no melody no synth no piano no guitar, "
        "only secondary hand percussion, total silence of all other instruments"
    ),
    "bass":     (
        "isolated bass line only, deep bass guitar sub bass 808 bassline, "
        "absolutely no drums no kick no snare no hi-hat no guitar no piano no synth no melody, "
        "only the bass frequency, silence everything except bass"
    ),
    "piano":    (
        "isolated acoustic piano only, solo piano melody, piano keys acoustic grand piano, "
        "absolutely no drums no bass no guitar no synth no strings no hi-hat no percussion, "
        "only piano notes, complete silence of all other instruments"
    ),
    "guitar":   (
        "isolated guitar only, solo guitar riff guitar melody clean guitar, "
        "absolutely no drums no bass no piano no synth no strings no percussion, "
        "only guitar, silence everything except guitar"
    ),
    "keyboard": (
        "isolated keyboard only, electric piano rhodes organ keys, "
        "absolutely no drums no bass no guitar no acoustic piano no percussion, "
        "only keyboard, silence everything except keyboard"
    ),
    "synth":    (
        "isolated synth lead only, synthesizer melody synth arp electronic lead, "
        "absolutely no drums no bass no piano no guitar no strings no percussion, "
        "only synthesizer, silence everything except synth"
    ),
    "strings":  (
        "isolated string section only, orchestral strings violin cello viola, "
        "absolutely no drums no bass no guitar no piano no synth no percussion, "
        "only strings, silence everything except strings"
    ),
    "brass":    (
        "isolated brass section only, trumpet trombone brass stabs horns, "
        "absolutely no drums no bass no guitar no piano no synth no strings no percussion, "
        "only brass instruments, silence everything except brass"
    ),
    "woodwinds":(
        "isolated woodwinds only, flute saxophone clarinet oboe, "
        "absolutely no drums no bass no guitar no piano no synth no strings no percussion, "
        "only woodwind instruments, silence everything except woodwinds"
    ),
    "vocals":   (
        "isolated vocal chops only, vocal stabs ad-libs vocal samples, "
        "absolutely no instruments no drums no bass no guitar no piano no synth, "
        "only vocals, silence everything except voice"
    ),
}


def _stem_tags(stem_name: str, full_tags: str) -> str:
    """Build an instrument-focused caption for a single stem generation."""
    focus   = _STEM_FOCUS.get(stem_name, f"{stem_name} only, isolated instrument")
    exclude = _STEM_EXCLUDE.get(stem_name, [])

    # Remove excluded instrument words from the full tags
    filtered_words = []
    for word in full_tags.split(","):
        w = word.strip().lower()
        if not any(ex in w for ex in exclude):
            filtered_words.append(word.strip())

    filtered = ", ".join(filtered_words)
    return f"{focus}, {filtered}" if filtered else focus


def _stems_to_generate(genre: str, style: str, user_prompt: str) -> list:
    """
    Return ordered list of (stem_name, display_name) for this beat.
    Drums and bass are always included. Additional stems depend on genre/user input.
    """
    combined = (style + " " + user_prompt).lower()
    g = genre.lower() if genre else ""

    stems = [("drums", "Drums"), ("bass", "Bass")]

    # Percussion for rhythm-heavy genres
    if g in _PERC_GENRES or any(x in combined for x in ["conga", "bongo", "shaker", "tambourine", "perc", "cowbell"]):
        stems.append(("percs", "Percs"))

    # User-requested instruments
    user_stems_added = set()
    for kw, ace_stem in INSTRUMENT_STEM_MAP:
        if kw in combined and ace_stem not in user_stems_added and ace_stem not in ("drums", "bass", "percs"):
            stems.append((ace_stem, ace_stem.capitalize()))
            user_stems_added.add(ace_stem)

    # Genre-based melody stem
    genre_stem = GENRE_MELODY_STEM.get(g, "keyboard")
    if genre_stem not in user_stems_added and genre_stem not in ("drums", "bass", "percs"):
        stems.append((genre_stem, genre_stem.capitalize()))

    # Add strings for cinematic/ambient/orchestral genres
    if g in {"cinematic", "ambient", "classical", "orchestral"} and "strings" not in user_stems_added:
        stems.append(("strings", "Strings"))

    # Add brass for jazz/soul/funk if not already present
    if g in {"jazz", "funk", "soul", "afrobeats"} and "brass" not in user_stems_added:
        stems.append(("brass", "Brass"))

    # Deduplicate keeping first occurrence
    seen, out = set(), []
    for s in stems:
        if s[0] not in seen:
            seen.add(s[0]); out.append(s)
    return out


# Keep old name as alias for any remaining references
_stems_to_extract = _stems_to_generate


# ═══════════════════════════════════════════════════════════════════════════════
# ── AUDIO MODE — ACE-Step 1.5 generation + stem extraction ────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _ace_generate(dit, llm, params: GenerationParams, save_path: str) -> str:
    """Run generate_music and return the path to the output WAV file."""
    config = GenerationConfig(batch_size=1, audio_format="wav")
    save_dir = os.path.dirname(save_path) or "/tmp"
    os.makedirs(save_dir, exist_ok=True)

    result = generate_music(dit, llm, params, config, save_dir=save_dir)

    # Inspect result
    if hasattr(result, "audios"):
        if result.audios:
            return result.audios[0]["path"]
        # audios is empty — log result for debugging and scan save_dir for output files
        print(f"[acestep] WARNING: result.audios is empty. result={result!r}", flush=True)
        wavs = sorted([
            os.path.join(save_dir, f) for f in os.listdir(save_dir)
            if f.endswith(".wav") or f.endswith(".mp3")
        ], key=os.path.getmtime, reverse=True)
        if wavs:
            print(f"[acestep] Found output file in save_dir: {wavs[0]}", flush=True)
            return wavs[0]
        raise RuntimeError(f"generate_music returned empty audios. result={result!r}")

    # Fallback: result may be a list of paths
    if isinstance(result, (list, tuple)) and result:
        return str(result[0])
    if isinstance(result, str) and os.path.exists(result):
        return result
    raise RuntimeError(f"generate_music returned unexpected result type: {type(result)!r}: {result!r}")


def _read_audio(path: str) -> np.ndarray:
    audio, _ = sf.read(path, dtype="float32")
    return audio


def transcribe_drums_from_stem(drum_stem_path: str, bpm: float) -> list:
    """
    Transcribe a Demucs drum stem to MIDI notes via onset detection +
    frequency-band classification (kick / snare / hi-hat / open hi-hat).

    Uses librosa onset detection on the full drum stem, then analyses a short
    window around each onset to decide which drum voice was hit.

    Returns a list of note dicts in BeatHole format.
    GM drum map: 36=Kick, 38=Snare, 42=Closed HH, 46=Open HH
    """
    try:
        import librosa
    except ImportError:
        print("[drum-transcribe] librosa not available — falling back to hardcoded grid", flush=True)
        return []

    try:
        print(f"[drum-transcribe] Loading drum stem...", flush=True)
        audio, sr = librosa.load(drum_stem_path, sr=22050, mono=True)

        # ── Onset detection ─────────────────────────────────────────────────
        # Use the drum-optimised onset envelope (RMS energy changes)
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr,
            units="time",
            backtrack=True,
            pre_max=2, post_max=2,
            pre_avg=30, post_avg=30,
            delta=0.07,
            wait=5,
        )
        print(f"[drum-transcribe] {len(onset_frames)} onsets detected", flush=True)

        beats_per_second = bpm / 60.0
        WIN_SAMPLES      = int(0.05 * sr)   # 50 ms analysis window
        notes            = []

        for onset_time in onset_frames:
            i0  = int(onset_time * sr)
            i1  = min(i0 + WIN_SAMPLES, len(audio))
            if i1 <= i0:
                continue

            window = audio[i0:i1]
            peak   = float(np.max(np.abs(window)))
            if peak < 0.01:        # skip very quiet hits (noise floor)
                continue

            # ── Frequency-band energy classification ────────────────────────
            fft   = np.abs(np.fft.rfft(window, n=2048))
            freqs = np.fft.rfftfreq(2048, d=1.0 / sr)

            low  = float(np.sum(fft[(freqs >= 20)   & (freqs < 250)]))   # kick sub
            mid  = float(np.sum(fft[(freqs >= 250)  & (freqs < 3000)]))  # snare body
            high = float(np.sum(fft[(freqs >= 5000) & (freqs < 20000)])) # hihat
            total = low + mid + high
            if total < 1e-9:
                continue

            low_r, mid_r, high_r = low / total, mid / total, high / total

            if low_r > 0.45:
                midi_note = 36   # Kick
            elif high_r > 0.45:
                # Distinguish open vs closed hi-hat by energy duration
                tail_end    = min(i0 + int(0.12 * sr), len(audio))
                tail_rms    = float(np.sqrt(np.mean(audio[i1:tail_end] ** 2))) if tail_end > i1 else 0.0
                midi_note   = 46 if tail_rms > 0.015 else 42
            else:
                midi_note = 38   # Snare

            velocity   = max(40, min(127, int(peak * 127 * 2.5)))
            start_beat = round(onset_time * beats_per_second, 4)
            notes.append(make_note(midi_note, start_beat, 0.125, velocity))

        print(f"[drum-transcribe] {len(notes)} drum MIDI notes produced", flush=True)
        return notes

    except Exception as e:
        print(f"[drum-transcribe] Failed: {e}", flush=True)
        return []


def transcribe_to_midi(stem_path: str, bpm: float, stem_name: str) -> list:
    """
    Transcribe a pitched audio stem to MIDI notes using Basic Pitch.
    Returns a list of note dicts in BeatHole format, or [] if unavailable/failed.
    """
    if not _BASIC_PITCH_OK:
        return []
    try:
        print(f"[midi-transcribe] Transcribing '{stem_name}'...", flush=True)
        _, _, note_events = _bp_predict(stem_path)
        notes = []
        beats_per_second = bpm / 60.0
        for ev in note_events:
            start_s, end_s, pitch, amplitude = ev[0], ev[1], ev[2], ev[3]
            start_beat = round(start_s * beats_per_second, 4)
            dur_beats  = round(max(0.0625, (end_s - start_s) * beats_per_second), 4)
            velocity   = max(1, min(127, int(amplitude * 127)))
            notes.append(make_note(int(pitch), start_beat, dur_beats, velocity))
        print(f"[midi-transcribe] '{stem_name}': {len(notes)} notes", flush=True)
        return notes
    except Exception as e:
        print(f"[midi-transcribe] '{stem_name}' failed: {e}", flush=True)
        return []


# Maps Demucs source names → (beathole_stem_key, display_name)
_DEMUCS_STEM_MAP = {
    "drums":  ("drums",  "Drums"),
    "bass":   ("bass",   "Bass"),
    "guitar": ("guitar", "Guitar"),
    "piano":  ("piano",  "Piano"),
    "other":  ("synth",  "Synth"),    # synth / pads / arps / misc melody
    "vocals": ("vocals", "Vocals"),
}

# Demucs stem key → studio instrument type
_DEMUCS_INST_TYPE = {
    "drums":  "drums",
    "bass":   "bass",
    "guitar": "synth",
    "piano":  "piano",
    "synth":  "synth",
    "vocals": "pad",
}


def _get_demucs():
    """Lazy-load and return the Demucs htdemucs_6s model."""
    global _demucs_model
    if _demucs_model is None:
        print(f"[demucs] Loading {_DEMUCS_MODEL_NAME}...", flush=True)
        _demucs_model = _demucs_get_model(_DEMUCS_MODEL_NAME)
        _demucs_model.eval()
        if torch.cuda.is_available():
            _demucs_model.cuda()
        print(f"[demucs] Loaded — stems: {_demucs_model.sources}", flush=True)
    return _demucs_model


def separate_stems_demucs(audio_path: str) -> dict:
    """
    Separate a mixed audio file into stems using Demucs htdemucs_6s.

    Returns a dict:
      { beathole_key: (mono_np_array_float32, display_name, sample_rate), ... }

    Stems returned: drums, bass, guitar, piano, synth (=other), and vocals
    if loud enough. Nearly-silent stems are skipped.
    """
    if not _DEMUCS_OK:
        return {}

    model   = _get_demucs()
    d_sr    = model.samplerate   # 44100

    # ── Load + resample to Demucs SR ─────────────────────────────────────────
    # Use soundfile (already installed) to avoid torchaudio codec backend issues
    # (torchaudio 2.10 defaults to torchcodec which may not be available)
    audio_np, src_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # soundfile returns [T, C] → convert to [C, T] tensor
    wav = torch.from_numpy(audio_np.T.copy())   # [C, T]
    if src_sr != d_sr:
        wav = torch.nn.functional.interpolate(
            wav.unsqueeze(0),                  # [1, C, T]
            size=int(wav.shape[-1] * d_sr / src_sr),
            mode="linear",
            align_corners=False,
        ).squeeze(0)                            # [C, T]

    # Ensure stereo [2, T]
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    wav = wav.unsqueeze(0)   # [1, 2, T]
    if torch.cuda.is_available():
        wav = wav.cuda()

    dur_s = wav.shape[-1] / d_sr
    print(f"[demucs] Separating {dur_s:.1f}s audio into stems...", flush=True)

    with torch.no_grad():
        sources = _demucs_apply(model, wav, progress=False)  # [1, num_stems, 2, T]

    sources = sources.squeeze(0).cpu()   # [num_stems, 2, T]

    result = {}
    for i, src_name in enumerate(model.sources):
        stem_mono = sources[i].mean(0).numpy().astype(np.float32)   # mono [T]
        rms       = float(np.sqrt(np.mean(stem_mono ** 2)))
        print(f"[demucs] '{src_name}' RMS={rms:.5f}", flush=True)

        # Skip nearly-silent stems
        if rms < 0.002:
            print(f"[demucs] Skipping silent '{src_name}'", flush=True)
            continue

        bh_key, display = _DEMUCS_STEM_MAP.get(src_name, (src_name, src_name.capitalize()))
        result[bh_key]  = (stem_mono, display, d_sr)
        print(f"[demucs] '{src_name}' → '{bh_key}' OK", flush=True)

    print(f"[demucs] Done — {len(result)} stems: {list(result.keys())}", flush=True)
    return result


def generate_audio_with_stems(job_input: dict) -> dict:
    """
    1. Generate the full beat with ACE-Step 1.5 text2music (coherent, properly structured).
    2. Extract stems (drums, bass, melody/instrument) from the generated audio via
       ACE-Step's extract task — stems always match the beat 100%.
    3. Upload stems to backend. Return combined beat WAV + stem URLs.
    """
    dit, dit_base, llm = get_handlers()

    user_prompt = job_input.get("prompt", "").strip()
    _dur_raw    = job_input.get("duration")
    duration    = float(_dur_raw) if _dur_raw else random.uniform(120, 240)
    duration    = max(90.0, min(240.0, duration))

    genre      = job_input.get("genre",  "").strip().lower()
    bpm_raw    = job_input.get("bpm")
    bpm        = int(bpm_raw) if bpm_raw else random.randint(85, 145)
    key        = (job_input.get("key") or "").strip()
    mood       = job_input.get("mood",  "").strip().lower()
    style      = job_input.get("style", "").strip()
    beat_id    = job_input.get("beatId", "")
    infer_step = int(job_input.get("inferSteps") or 40)

    ref_audio_b64  = job_input.get("referenceAudio")
    ref_strength   = float(job_input.get("referenceStrength") or 0.5)

    tags       = build_tags(user_prompt or style, genre, mood, bpm, key)
    lyrics_str = _build_lyrics_structure(duration, user_prompt, genre)
    seed       = random.randint(0, 2**31 - 1)
    g_scale    = _get_guidance_scale(genre, user_prompt)

    print(f"[gen] Tags: {tags}", flush=True)
    print(f"[gen] Duration: {duration:.0f}s | BPM: {bpm} | Key: {key} | Seed: {seed} | guidance={g_scale}", flush=True)

    # Handle reference audio → temp file
    ref_path = None
    if ref_audio_b64:
        ref_path = "/tmp/bh_reference.wav"
        with open(ref_path, "wb") as f:
            f.write(base64.b64decode(ref_audio_b64))
        print(f"[gen] Reference audio saved", flush=True)

    # ── 1. Generate the full beat (text2music) ────────────────────────────────
    print(f"[gen] Generating full beat ({duration:.0f}s, {bpm} BPM, seed={seed})...", flush=True)
    full_params = GenerationParams(
        task_type       = "text2music",
        caption         = tags,
        lyrics          = lyrics_str,
        duration        = int(duration),
        bpm             = bpm,
        keyscale        = key if key else "N/A",
        timesignature   = "4/4",
        inference_steps = infer_step,
        guidance_scale  = g_scale,
        seed            = seed,
    )
    if ref_path:
        full_params.audio2audio_enable = True
        full_params.ref_audio_input    = ref_path
        full_params.ref_audio_strength = ref_strength

    full_beat_path = _ace_generate(dit, llm, full_params, "/tmp/bh_main.wav")
    main_audio     = _read_audio(full_beat_path)
    actual_dur     = (main_audio.shape[0] if main_audio.ndim == 1 else main_audio.shape[0]) / SAMPLE_RATE
    print(f"[gen] Full beat done — shape: {main_audio.shape}, dur: {actual_dur:.1f}s", flush=True)

    # ── 2. Separate stems with Demucs htdemucs_6s ────────────────────────────
    # Replaces ACE-Step extract — Demucs is a proper source separator with no
    # inter-stem bleed (no drums in synth, no piano masquerading as synth, etc.)
    print("[stems] Separating stems with Demucs htdemucs_6s...", flush=True)
    stems_audio = {}   # { bh_key: (mono_array, display_name, sample_rate) }
    if _DEMUCS_OK:
        try:
            demucs_result = separate_stems_demucs(full_beat_path)
            stems_audio   = demucs_result
        except Exception as e:
            print(f"[stems] Demucs separation failed: {e}", flush=True)
    else:
        print("[stems] Demucs unavailable — no stems will be uploaded", flush=True)

    # ── 3. Upload stems to backend ────────────────────────────────────────────
    backend_url  = os.environ.get("BACKEND_URL", "").rstrip("/")
    internal_key = os.environ.get("INTERNAL_API_KEY", "")
    stem_urls    = {}

    if backend_url and internal_key and beat_id:
        for bh_stem, (stem_audio, _display, d_sr) in stems_audio.items():
            try:
                audio_down = resample_mono(stem_audio, d_sr, STEM_SR)
                b64  = np_to_wav_b64(audio_down, sr=STEM_SR)
                url  = upload_stem(backend_url, internal_key, beat_id, bh_stem, b64)
                stem_urls[bh_stem] = url
                print(f"[stems] Uploaded '{bh_stem}' → {url}", flush=True)
            except Exception as e:
                print(f"[stems] Upload failed for '{bh_stem}': {e}", flush=True)
    else:
        print("[stems] Skipping upload — env vars not set", flush=True)

    # ── 4. Upload main beat ───────────────────────────────────────────────────
    wav_url = None
    if backend_url and internal_key and beat_id:
        try:
            main_b64 = np_to_wav_b64(main_audio)
            wav_url  = upload_main_audio(backend_url, internal_key, beat_id, main_b64)
            print(f"[gen] Uploaded main beat → {wav_url}", flush=True)
        except Exception as e:
            print(f"[gen] Main beat upload failed: {e}", flush=True)

    # ── 5. Return ─────────────────────────────────────────────────────────────
    return {
        "wav_url":          wav_url,
        "stem_urls":        stem_urls,
        "sample_rate":      SAMPLE_RATE,
        "duration_seconds": round(actual_dur, 2),
        "bpm":              bpm,
        "key":              key,
        "tags":             tags,
    }


def generate_preview_audio(job_input: dict) -> tuple:
    """Short 60s preview for MIDI mode beat page. Returns (wav_base64, duration_sec)."""
    dit, dit_base, llm = get_handlers()

    genre  = job_input.get("genre",  "").strip().lower()
    mood   = job_input.get("mood",   "").strip().lower()
    bpm    = int(job_input.get("bpm") or 120)
    key    = (job_input.get("key") or "").strip()
    prompt = (job_input.get("prompt") or job_input.get("style") or "").strip()

    tags = build_tags(prompt, genre, mood, bpm, key)
    print(f"[midi-preview] Generating 60s preview...", flush=True)

    params = GenerationParams(
        task_type      = "text2music",
        caption        = tags,
        lyrics         = "[intro]\n[inst]\n[verse]\n[inst]\n[chorus]\n[inst]",
        duration       = 60,
        bpm            = bpm,
        keyscale       = key if key else "N/A",
        timesignature  = "4/4",
        inference_steps= 20,
        guidance_scale = 7.5,
        seed           = random.randint(0, 2**31 - 1),
    )

    path  = _ace_generate(dit, llm, params, "/tmp/bh_preview.wav")
    audio = _read_audio(path)
    print("[midi-preview] Preview done", flush=True)
    dur = len(audio) / SAMPLE_RATE if audio.ndim == 1 else audio.shape[-1] / SAMPLE_RATE
    return np_to_wav_b64(audio), round(dur, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# ── MIDI MODE — programmatic structured MIDI generation ───────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

NOTE_ROOT    = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
_NOTE_ALIAS  = {'Db':1,'Eb':3,'Gb':6,'Ab':8,'Bb':10}

def parse_key_input(raw: str):
    s  = (raw or 'C').strip()
    sl = s.lower()
    mode_override = None
    if any(x in sl for x in ['minor', ' min', 'moll']):
        mode_override = 'minor'
    elif any(x in sl for x in ['major', ' maj', 'dur']):
        mode_override = 'major'
    all_notes = {**NOTE_ROOT, **_NOTE_ALIAS}
    root, matched = 0, 0
    for length in (2, 1):
        chunk = s[:length]
        if chunk in all_notes:
            root, matched = all_notes[chunk], length
            break
    if mode_override is None and matched > 0:
        rest = s[matched:].strip().lower()
        if rest.startswith('m') and not rest.startswith('maj'):
            mode_override = 'minor'
    return root, mode_override

# ── Per-genre 16-step drum probability grids ─────────────────────────────────
DRUM_GRIDS = {
    'trap': {
        'kick':  [1.0,0.00,0.00,0.05, 0.00,0.00,0.85,0.10, 0.00,0.00,0.90,0.05, 0.00,0.05,0.75,0.15],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.80,0.90,0.75, 0.90,0.80,0.90,0.75, 0.90,0.80,0.90,0.75, 0.90,0.80,0.90,0.70],
        'open':  [0.0,0.00,0.00,0.65, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.65, 0.00,0.00,0.00,0.00],
    },
    'drill': {
        'kick':  [1.0,0.00,0.00,0.10, 0.00,0.00,0.00,0.15, 0.00,0.00,0.85,0.05, 0.00,0.00,0.90,0.20],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00],
    },
    'hip hop': {
        'kick':  [1.0,0.00,0.00,0.10, 0.00,0.20,0.00,0.10, 1.00,0.00,0.00,0.15, 0.00,0.00,0.00,0.20],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.10, 0.00,0.00,0.00,0.00, 1.00,0.00,0.10,0.00],
        'hihat': [0.9,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00],
        'open':  [0.0,0.00,0.00,0.35, 0.00,0.00,0.00,0.35, 0.00,0.00,0.00,0.35, 0.00,0.00,0.00,0.35],
    },
    'boom bap': {
        'kick':  [1.0,0.00,0.00,0.20, 0.00,0.10,0.00,0.30, 1.00,0.00,0.00,0.25, 0.00,0.00,0.30,0.10],
        'snare': [0.0,0.00,0.00,0.10, 1.00,0.00,0.15,0.00, 0.00,0.10,0.00,0.00, 1.00,0.00,0.15,0.00],
        'hihat': [0.9,0.65,0.75,0.55, 0.90,0.65,0.75,0.55, 0.90,0.65,0.75,0.55, 0.90,0.65,0.75,0.55],
        'open':  [0.0,0.00,0.00,0.45, 0.00,0.00,0.00,0.45, 0.00,0.00,0.00,0.45, 0.00,0.00,0.00,0.45],
    },
    'r&b': {
        'kick':  [1.0,0.00,0.00,0.10, 0.00,0.00,0.00,0.35, 1.00,0.00,0.00,0.10, 0.00,0.30,0.00,0.15],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.20,0.00, 0.00,0.10,0.00,0.20, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.75,0.85,0.65, 0.90,0.75,0.85,0.65, 0.90,0.75,0.85,0.65, 0.90,0.75,0.85,0.65],
        'open':  [0.0,0.00,0.00,0.50, 0.00,0.00,0.00,0.50, 0.00,0.00,0.00,0.50, 0.00,0.00,0.00,0.50],
    },
    'pop': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.15, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.15],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.85,0.90,0.80, 0.90,0.85,0.90,0.80, 0.90,0.85,0.90,0.80, 0.90,0.85,0.90,0.80],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.45, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.45],
    },
    'electronic': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.10],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.15, 0.00,0.00,0.00,0.00, 1.00,0.00,0.15,0.00],
        'hihat': [1.0,0.90,1.00,0.90, 1.00,0.90,1.00,0.90, 1.00,0.90,1.00,0.90, 1.00,0.90,1.00,0.90],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.55, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.55],
    },
    'lo-fi': {
        'kick':  [1.0,0.00,0.00,0.25, 0.00,0.00,0.15,0.00, 0.90,0.00,0.00,0.25, 0.00,0.15,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 0.90,0.00,0.15,0.00, 0.00,0.15,0.00,0.00, 0.90,0.00,0.15,0.00],
        'hihat': [0.8,0.00,0.00,0.70, 0.80,0.00,0.00,0.70, 0.80,0.00,0.00,0.70, 0.80,0.00,0.00,0.70],
        'open':  [0.0,0.00,0.00,0.55, 0.00,0.00,0.00,0.55, 0.00,0.00,0.00,0.55, 0.00,0.00,0.00,0.55],
    },
    'phonk': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.85,0.00, 0.00,0.00,0.90,0.25],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.80,0.90,0.80, 0.90,0.80,0.90,0.80, 0.90,0.80,0.90,0.80, 0.90,0.80,0.90,0.80],
        'open':  [0.0,0.00,0.00,0.60, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.60, 0.00,0.00,0.00,0.00],
    },
    'cloud rap': {
        'kick':  [1.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.35, 0.00,0.00,0.85,0.00, 0.00,0.00,0.00,0.25],
        'snare': [0.0,0.00,0.00,0.00, 0.90,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.90,0.00,0.00,0.00],
        'hihat': [0.7,0.70,0.70,0.70, 0.70,0.70,0.70,0.70, 0.70,0.70,0.70,0.70, 0.70,0.70,0.70,0.70],
        'open':  [0.0,0.00,0.00,0.45, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.45, 0.00,0.00,0.00,0.00],
    },
    'afrobeats': {
        'kick':  [1.0,0.00,0.00,0.00, 0.00,0.85,0.00,0.15, 0.90,0.00,0.00,0.00, 0.00,0.85,0.00,0.25],
        'snare': [0.0,0.00,0.00,0.00, 0.00,0.00,0.85,0.00, 0.00,0.00,0.00,0.00, 0.85,0.00,0.00,0.00],
        'hihat': [0.9,0.70,0.85,0.80, 0.90,0.70,0.85,0.80, 0.90,0.70,0.85,0.80, 0.90,0.70,0.85,0.80],
        'open':  [0.0,0.00,0.50,0.00, 0.00,0.00,0.50,0.00, 0.00,0.00,0.50,0.00, 0.00,0.00,0.50,0.00],
    },
    'dancehall': {
        'kick':  [0.0,0.00,0.00,0.85, 0.00,0.00,0.00,0.15, 0.00,0.00,0.90,0.00, 0.00,0.00,0.85,0.10],
        'snare': [0.0,0.00,0.00,0.00, 0.90,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.85,0.00,0.00],
        'hihat': [0.9,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00],
        'open':  [0.0,0.00,0.50,0.00, 0.00,0.00,0.50,0.00, 0.00,0.00,0.50,0.00, 0.00,0.00,0.50,0.00],
    },
    'house': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.85,0.90,0.85, 0.90,0.85,0.90,0.85, 0.90,0.85,0.90,0.85, 0.90,0.85,0.90,0.85],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.75,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.75,0.00,0.00],
    },
    'deep house': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 0.85,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.85,0.00,0.00,0.00],
        'hihat': [0.8,0.00,0.75,0.00, 0.80,0.00,0.75,0.00, 0.80,0.00,0.75,0.00, 0.80,0.00,0.75,0.00],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.65,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.65,0.00,0.00],
    },
    'tech house': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.10],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.15,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.15,0.00],
        'hihat': [1.0,0.90,1.00,0.90, 1.00,0.90,1.00,0.90, 1.00,0.90,1.00,0.90, 1.00,0.90,1.00,0.90],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.50,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.50,0.00,0.00],
    },
    'techno': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [1.0,1.00,1.00,1.00, 1.00,1.00,1.00,1.00, 1.00,1.00,1.00,1.00, 1.00,1.00,1.00,1.00],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.40, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.40],
    },
    'edm': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.90,0.90,0.90, 0.90,0.90,0.90,0.90, 0.90,0.90,0.90,0.90, 0.90,0.90,0.90,0.90],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.65, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.65],
    },
    'dubstep': {
        'kick':  [1.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.75,0.00,0.00,0.25, 0.00,0.00,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00],
        'hihat': [0.8,0.00,0.70,0.00, 0.00,0.70,0.00,0.00, 0.80,0.00,0.70,0.00, 0.00,0.70,0.00,0.00],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00],
    },
    'drum and bass': {
        'kick':  [1.0,0.00,0.00,0.00, 0.00,0.00,0.70,0.00, 0.00,0.00,0.80,0.00, 0.00,0.60,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.30],
        'hihat': [0.9,0.90,0.90,0.90, 0.90,0.90,0.90,0.90, 0.90,0.90,0.90,0.90, 0.90,0.90,0.90,0.90],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.45, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.45],
    },
    'synthwave': {
        'kick':  [1.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.85,0.90,0.85, 0.90,0.85,0.90,0.85, 0.90,0.85,0.90,0.85, 0.90,0.85,0.90,0.85],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.55, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.55],
    },
    'funk': {
        'kick':  [1.0,0.00,0.15,0.00, 0.00,0.80,0.00,0.20, 0.90,0.00,0.00,0.70, 0.00,0.00,0.85,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.30,0.00, 0.00,0.25,0.00,0.00, 1.00,0.00,0.30,0.00],
        'hihat': [0.9,0.80,0.85,0.75, 0.90,0.80,0.85,0.75, 0.90,0.80,0.85,0.75, 0.90,0.80,0.85,0.75],
        'open':  [0.0,0.00,0.00,0.60, 0.00,0.00,0.00,0.60, 0.00,0.00,0.00,0.60, 0.00,0.00,0.00,0.60],
    },
    'jazz': {
        'kick':  [0.9,0.00,0.00,0.25, 0.00,0.00,0.20,0.00, 0.85,0.00,0.00,0.25, 0.00,0.00,0.15,0.20],
        'snare': [0.0,0.00,0.00,0.00, 0.80,0.00,0.15,0.00, 0.00,0.10,0.00,0.00, 0.80,0.00,0.15,0.00],
        'hihat': [0.9,0.30,0.80,0.30, 0.90,0.30,0.80,0.30, 0.90,0.30,0.80,0.30, 0.90,0.30,0.80,0.30],
        'open':  [0.0,0.00,0.00,0.55, 0.00,0.00,0.00,0.55, 0.00,0.00,0.00,0.55, 0.00,0.00,0.00,0.55],
    },
    'reggaeton': {
        'kick':  [1.0,0.00,0.00,0.85, 0.00,0.00,0.00,0.00, 0.00,0.85,0.00,0.00, 0.00,0.00,0.00,0.85],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.70,0.85,0.70, 0.90,0.70,0.85,0.70, 0.90,0.70,0.85,0.70, 0.90,0.70,0.85,0.70],
        'open':  [0.0,0.00,0.00,0.40, 0.00,0.00,0.00,0.40, 0.00,0.00,0.00,0.40, 0.00,0.00,0.00,0.40],
    },
    'grime': {
        'kick':  [1.0,0.00,0.00,0.00, 0.00,0.00,0.80,0.00, 1.00,0.00,0.00,0.00, 0.00,0.75,0.00,0.00],
        'snare': [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00],
        'hihat': [0.9,0.85,0.90,0.80, 0.90,0.85,0.90,0.80, 0.90,0.85,0.90,0.80, 0.90,0.85,0.90,0.80],
        'open':  [0.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.45, 0.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.45],
    },
    '_default': {
        'kick':  [1.0,0.00,0.00,0.00, 0.00,0.00,0.00,0.20, 0.90,0.00,0.00,0.00, 0.00,0.00,0.25,0.00],
        'snare': [0.0,0.00,0.00,0.00, 1.00,0.00,0.00,0.00, 0.00,0.00,0.00,0.00, 1.00,0.00,0.00,0.00],
        'hihat': [0.9,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00, 0.90,0.00,0.85,0.00],
        'open':  [0.0,0.00,0.00,0.35, 0.00,0.00,0.00,0.35, 0.00,0.00,0.00,0.35, 0.00,0.00,0.00,0.35],
    },
}
DRUM_GRIDS['hip-hop']   = DRUM_GRIDS['hip hop']
DRUM_GRIDS['dnb']       = DRUM_GRIDS['drum and bass']
DRUM_GRIDS['jungle']    = DRUM_GRIDS['drum and bass']
DRUM_GRIDS['vaporwave'] = DRUM_GRIDS['synthwave']
DRUM_GRIDS['hyperpop']  = DRUM_GRIDS['edm']
DRUM_GRIDS['uk garage'] = DRUM_GRIDS['house']
DRUM_GRIDS['latin']     = DRUM_GRIDS['reggaeton']
DRUM_GRIDS['soul']      = DRUM_GRIDS['r&b']
DRUM_GRIDS['cinematic'] = DRUM_GRIDS['_default']
DRUM_GRIDS['ambient']   = DRUM_GRIDS['_default']

CHORD_PROG_SETS = {
    'trap':          [[0,10,8,3],  [0,8,5,3],   [0,10,5,8],  [0,3,10,8]],
    'drill':         [[0,8,10,5],  [0,3,8,10],  [0,5,8,3],   [0,8,3,10]],
    'hip hop':       [[0,10,8,5],  [0,5,3,8],   [0,7,5,10],  [0,3,5,10]],
    'hip-hop':       [[0,10,8,5],  [0,5,3,8],   [0,7,5,10],  [0,3,5,10]],
    'boom bap':      [[0,5,3,8],   [0,10,5,8],  [0,7,3,5],   [0,5,10,3]],
    'r&b':           [[0,9,5,7],   [0,4,9,7],   [0,9,7,5],   [0,5,9,4]],
    'pop':           [[0,7,9,5],   [0,5,7,9],   [0,9,5,7],   [0,4,7,5]],
    'electronic':    [[0,5,7,3],   [0,3,7,5],   [0,7,3,10],  [0,5,10,7]],
    'lo-fi':         [[0,7,9,5],   [0,5,3,7],   [0,9,5,3],   [0,7,5,9]],
    'phonk':         [[0,8,5,10],  [0,3,8,5],   [0,10,3,8],  [0,5,8,10]],
    'cloud rap':     [[0,10,8,10], [0,8,10,8],  [0,3,10,8],  [0,10,5,8]],
    'afrobeats':     [[0,5,7,5],   [0,7,5,2],   [0,5,2,7],   [0,2,5,9]],
    'dancehall':     [[0,5,7,0],   [0,7,5,7],   [0,2,5,7],   [0,5,0,7]],
    'house':         [[0,5,7,9],   [0,7,9,5],   [0,4,7,9],   [0,5,9,7]],
    'deep house':    [[0,5,7,9],   [0,9,5,7],   [0,7,5,2],   [0,5,2,9]],
    'tech house':    [[0,5,7,3],   [0,3,7,5],   [0,7,3,10],  [0,5,10,3]],
    'techno':        [[0,3,7,10],  [0,10,7,3],  [0,7,3,5],   [0,5,3,10]],
    'edm':           [[0,7,9,5],   [0,5,9,7],   [0,9,7,5],   [0,4,7,9]],
    'dubstep':       [[0,8,5,10],  [0,5,8,3],   [0,3,8,5],   [0,10,5,8]],
    'drum and bass': [[0,8,5,3],   [0,3,5,8],   [0,5,3,8],   [0,10,8,5]],
    'synthwave':     [[0,9,7,5],   [0,5,9,7],   [0,7,9,5],   [0,4,9,7]],
    'vaporwave':     [[0,9,7,5],   [0,5,7,9],   [0,7,5,3],   [0,9,5,7]],
    'hyperpop':      [[0,7,9,5],   [0,5,7,9],   [0,9,7,4],   [0,4,9,5]],
    'reggaeton':     [[0,5,7,0],   [0,7,5,2],   [0,2,5,7],   [0,5,2,0]],
    'latin':         [[0,5,7,0],   [0,7,5,2],   [0,2,7,5],   [0,5,0,7]],
    'funk':          [[0,5,7,9],   [0,7,9,5],   [0,9,5,7],   [0,4,7,9]],
    'jazz':          [[0,4,7,9],   [0,9,4,7],   [0,7,4,9],   [0,5,9,4]],
    'soul':          [[0,5,9,7],   [0,9,5,7],   [0,7,5,9],   [0,4,9,5]],
    'cinematic':     [[0,7,5,9],   [0,5,8,3],   [0,8,5,7],   [0,3,7,10]],
    'ambient':       [[0,7,9,5],   [0,5,7,9],   [0,9,5,3],   [0,7,3,9]],
    'grime':         [[0,3,8,5],   [0,8,3,10],  [0,5,3,8],   [0,10,5,3]],
    'uk garage':     [[0,5,9,7],   [0,7,9,5],   [0,4,9,5],   [0,9,5,7]],
    'pluggnb':       [[0,8,10,5],  [0,10,8,3],  [0,3,10,8],  [0,5,8,10]],
    'jersey club':   [[0,7,5,9],   [0,5,9,7],   [0,9,5,4],   [0,4,7,9]],
    '_default':      [[0,7,5,9],   [0,5,7,9],   [0,9,5,7],   [0,7,9,5]],
}

MINOR_GENRES  = {'trap','drill','hip hop','hip-hop','phonk','cloud rap','lo-fi',
                 'grime','dubstep','drum and bass','dnb','jungle','techno','cinematic'}
NATURAL_MINOR = [0,2,3,5,7,8,10]
MAJOR_SCALE   = [0,2,4,5,7,9,11]
PENT_MINOR    = [0,3,5,7,10]
PENT_MAJOR    = [0,2,4,7,9]

def uid(): return str(uuid.uuid4())[:8]

def make_note(note: int, start_beat: float, dur_beats: float, velocity: int) -> dict:
    return {"id": uid(), "note": note, "startBeat": round(start_beat, 4),
            "durationBeats": round(dur_beats, 4), "velocity": velocity}

def humanize(base: int, spread: int) -> int:
    return max(1, min(127, base + random.randint(-spread, spread)))

def get_drum_grid(genre: str) -> dict:
    return DRUM_GRIDS.get(genre, DRUM_GRIDS['_default'])

def get_chord_prog(genre: str) -> list:
    sets = CHORD_PROG_SETS.get(genre, CHORD_PROG_SETS['_default'])
    return random.choice(sets)

def generate_midi_tracks(job_input: dict) -> dict:
    genre    = job_input.get("genre", "").strip().lower()
    style    = job_input.get("style", "").strip().lower()
    raw_key  = job_input.get("key", "C").strip()
    mood     = job_input.get("mood", "").strip().lower()
    bpm      = float(job_input.get("bpm", 140) or 140)
    user_p   = job_input.get("prompt", "").strip().lower()
    _dur_raw = job_input.get("duration")
    duration = float(_dur_raw) if _dur_raw else random.uniform(120, 240)
    duration = max(90.0, min(240.0, duration))
    no_drums = any(x in user_p for x in ["no drums","without drums","drumless","no drum","geen drums","no percussion"])

    total_beats = math.ceil((bpm * duration) / 60)
    total_bars  = min(math.ceil(total_beats / 4), 128)

    key_root, mode_override = parse_key_input(raw_key)
    if mode_override == 'minor':
        is_minor = True
    elif mode_override == 'major':
        is_minor = False
    else:
        is_minor = (genre in MINOR_GENRES or
                    any(m in mood + user_p for m in ["sad","dark","minor","melancholic","ominous"]))

    scale_full = NATURAL_MINOR if is_minor else MAJOR_SCALE
    pent       = PENT_MINOR    if is_minor else PENT_MAJOR

    chord_prog = get_chord_prog(genre)

    if bpm < 85:
        swing = 0.07
    elif bpm < 100:
        swing = 0.05
    elif bpm < 120:
        swing = 0.03
    else:
        swing = 0.0

    tracks = []

    # ── Drums ──────────────────────────────────────────────────────────────────
    # GM notes: 36=Kick, 38=Snare, 39=Clap, 42=Closed HH, 46=Open HH,
    #           49=Crash, 51=Ride, 54=Tambourine, 56=Cowbell, 63=HiConga,
    #           64=LoConga, 69=Cabasa/Shaker
    if not no_drums:
        grid = get_drum_grid(genre)
        kick_notes, snare_notes, hihat_notes, clap_notes, perc_notes = [], [], [], [], []

        # Genre-based clap/perc probability
        clap_prob  = 0.85 if genre in {"trap","drill","phonk","hip hop","hip-hop","pop","edm","dancehall"} else 0.40
        has_percs  = genre in _PERC_GENRES or any(x in user_p for x in ["perc","conga","shaker","cowbell"])
        perc_note_choices = {
            "afrobeats":  [(63, 0.7), (64, 0.5), (69, 0.6)],
            "reggaeton":  [(63, 0.6), (64, 0.4), (54, 0.5)],
            "latin":      [(63, 0.7), (64, 0.6), (54, 0.4)],
            "funk":       [(54, 0.6), (56, 0.3), (69, 0.5)],
            "jazz":       [(56, 0.3), (69, 0.4), (51, 0.5)],
            "trap":       [(54, 0.4), (69, 0.5)],
            "hip hop":    [(54, 0.4), (69, 0.4)],
            "hip-hop":    [(54, 0.4), (69, 0.4)],
            "boom bap":   [(54, 0.35), (69, 0.35)],
            "lo-fi":      [(54, 0.35), (69, 0.3)],
            "_default":   [(54, 0.25), (69, 0.3)],
        }
        perc_choices = perc_note_choices.get(genre, perc_note_choices["_default"])

        for bar in range(total_bars):
            base    = bar * 4.0
            is_fill = (bar % 8 == 7)

            for step in range(16):
                swing_off = swing if (step % 4 == 2) else 0.0
                t      = base + step * 0.25 + swing_off
                fill_b = 0.45 if (is_fill and step >= 12) else 0.0

                p = min(1.0, grid['kick'][step] + fill_b * 0.30)
                if p > 0 and random.random() < p:
                    base_vel = 108 if step == 0 else 98 if step == 8 else 84
                    kick_notes.append(make_note(36, t, 0.22, humanize(base_vel, 10)))

                p = min(1.0, grid['snare'][step] + fill_b * 0.40)
                if p > 0 and random.random() < p:
                    is_ghost = grid['snare'][step] < 0.3
                    base_vel = 36 if is_ghost else 94
                    snare_notes.append(make_note(38, t, 0.18, humanize(base_vel, 12)))
                    # Clap layered on snare beats (not ghost notes)
                    if not is_ghost and random.random() < clap_prob:
                        clap_notes.append(make_note(39, t + random.uniform(0, 0.01), 0.12, humanize(82, 10)))

                p = min(1.0, grid['hihat'][step] + fill_b * 0.50)
                if p > 0 and random.random() < p:
                    is_open  = grid['open'][step] > 0 and random.random() < grid['open'][step]
                    note_num = 46 if is_open else 42
                    dur      = 0.22 if is_open else 0.08
                    base_vel = 78 if step % 4 == 0 else 58
                    hihat_notes.append(make_note(note_num, t, dur, humanize(base_vel, 15)))

                # Auxiliary percussion (congas, shaker, tambourine, cowbell)
                if has_percs:
                    for perc_note, perc_p in perc_choices:
                        adjusted = min(1.0, perc_p + (fill_b * 0.2))
                        if random.random() < adjusted:
                            perc_notes.append(make_note(perc_note, t + random.uniform(0, 0.02),
                                                        0.12, humanize(68, 15)))

        tracks.append({"name": "Kick",   "instrument": "drums", "notes": kick_notes,  "total_beats": total_bars * 4})
        tracks.append({"name": "Snare",  "instrument": "drums", "notes": snare_notes, "total_beats": total_bars * 4})
        tracks.append({"name": "Hi-Hat", "instrument": "drums", "notes": hihat_notes, "total_beats": total_bars * 4})
        if clap_notes:
            tracks.append({"name": "Clap",   "instrument": "drums", "notes": clap_notes,  "total_beats": total_bars * 4})
        if perc_notes:
            tracks.append({"name": "Percs",  "instrument": "drums", "notes": perc_notes,  "total_beats": total_bars * 4})

    # ── Bass line ─────────────────────────────────────────────────────────────
    bass_notes  = []
    bass_octave = 36

    BASS_RHYTHMS = [
        [(0.0, 1.50, 95)],
        [(0.0, 1.00, 95), (2.0,  0.75, 80)],
        [(0.0, 0.75, 95), (1.5,  0.50, 75), (2.5, 0.50, 70)],
        [(0.0, 0.50, 95), (0.75, 0.50, 78), (2.0, 1.00, 85)],
        [(0.0, 1.00, 95), (2.75, 0.50, 72)],
        [(0.0, 2.00, 95)],
    ]

    for bar in range(total_bars):
        chord_idx  = (bar // 2) % len(chord_prog)
        chord_root = (key_root + chord_prog[chord_idx]) % 12
        bass_note  = bass_octave + chord_root
        fifth_note = bass_octave + (chord_root + 7) % 12
        pattern = BASS_RHYTHMS[(bar // 2) % len(BASS_RHYTHMS)]
        for i, (beat_off, dur, base_vel) in enumerate(pattern):
            note = fifth_note if (i > 0 and random.random() < 0.25) else bass_note
            bass_notes.append(make_note(note, bar * 4.0 + beat_off, dur, humanize(base_vel, 8)))

    bass_inst = ("synth" if genre in {"trap","drill","phonk","cloud rap","techno","edm",
                                     "dubstep","drum and bass","dnb","grime","hyperpop",
                                     "house","deep house","tech house","synthwave","electronic"}
                 else "bass")
    tracks.append({"name": "808 Bass" if bass_inst == "synth" else "Bass Line",
                   "instrument": bass_inst, "notes": bass_notes, "total_beats": total_bars * 4})

    # ── Chord pads ────────────────────────────────────────────────────────────
    pad_notes    = []
    chord_octave = 48

    def build_chord(root_off, minor, add7_prob=0.25):
        third  = root_off + (3 if minor else 4)
        fifth  = root_off + 7
        result = [root_off, third, fifth]
        if random.random() < add7_prob:
            result.append(root_off + (10 if minor else 11))
        return result

    for bar in range(total_bars):
        chord_idx      = (bar // 2) % len(chord_prog)
        chord_root     = (key_root + chord_prog[chord_idx]) % 12
        intervals      = build_chord(chord_root, is_minor)
        stagger        = random.random() < 0.35
        octave_shift   = random.choice([0, 0, 0, 12])
        for j, interval in enumerate(intervals):
            note_num  = chord_octave + octave_shift + interval
            start_off = j * 0.04 if stagger else 0.0
            vel       = humanize(68 if j == 0 else 62, 8)
            pad_notes.append(make_note(note_num, bar * 4.0 + start_off, 3.5, vel))

    pad_inst = ("pad"    if genre in {"cloud rap","r&b","lo-fi","electronic","pop","house",
                                     "deep house","synthwave","vaporwave","ambient","edm"}
                else "epiano")
    tracks.append({"name": "Chords", "instrument": pad_inst,
                   "notes": pad_notes, "total_beats": total_bars * 4})

    # ── Melody ────────────────────────────────────────────────────────────────
    mel_notes  = []
    mel_octave = 60
    prev_note  = mel_octave + key_root

    if any(x in style + user_p for x in ["piano"]):
        mel_inst = "piano"
    elif any(x in style + user_p for x in ["synth","lead synth","electronic lead"]):
        mel_inst = "synth"
    elif any(x in style + user_p for x in ["rhodes","epiano","e-piano"]):
        mel_inst = "epiano"
    elif genre in {"r&b","boom bap","lo-fi","soul","jazz"}:
        mel_inst = "epiano"
    elif genre in {"electronic","phonk","techno","edm","dubstep","drum and bass","dnb",
                   "house","deep house","tech house","synthwave","vaporwave","grime","hyperpop"}:
        mel_inst = "synth"
    elif genre in {"cinematic","ambient"}:
        mel_inst = "pad"
    else:
        mel_inst = "piano"

    scale_notes = [mel_octave + (key_root + iv) % 12 for iv in pent]
    scale_notes += [n + 12 for n in scale_notes]

    def nearest_scale_note(target, notes):
        return min(notes, key=lambda n: abs(n - target))

    for bar in range(total_bars):
        chord_idx  = (bar // 2) % len(chord_prog)
        chord_root = (key_root + chord_prog[chord_idx]) % 12
        in_chorus  = (bar % 8) >= 4
        n_notes    = random.randint(3, 5) if in_chorus else random.randint(1, 3)
        positions  = sorted(random.sample(range(8), min(n_notes, 8)))

        for pos in positions:
            if random.random() < 0.55:
                target = mel_octave + (chord_root + random.choice([0, 3 if is_minor else 4, 7])) % 12
            else:
                step = random.choice([-2, -1, 0, 1, 2])
                try:
                    idx = scale_notes.index(nearest_scale_note(prev_note, scale_notes))
                except ValueError:
                    idx = 0
                idx    = max(0, min(len(scale_notes) - 1, idx + step))
                target = scale_notes[idx]

            snote     = nearest_scale_note(target, scale_notes)
            prev_note = snote
            dur       = random.choice([0.25, 0.5, 0.5, 0.75])
            mel_notes.append(make_note(snote, bar * 4.0 + pos * 0.5, dur, humanize(82, 12)))

    tracks.append({"name": mel_inst.capitalize(), "instrument": mel_inst,
                   "notes": mel_notes, "total_beats": total_bars * 4})

    # ── Extra instruments the user explicitly requested ────────────────────────
    style_combined = (style + " " + user_p).lower()
    EXTRA_INSTS = [
        ("strings",          "pad",    "Strings"),
        ("violin",           "pad",    "Violin"),
        ("cello",            "pad",    "Cello"),
        ("choir",            "pad",    "Choir"),
        ("pad",              "pad",    "Pad"),
        ("synth",            "synth",  "Synth Lead"),
        ("organ",            "epiano", "Organ"),
        ("epiano",           "epiano", "E-Piano"),
        ("rhodes",           "epiano", "Rhodes"),
        ("flute",            "piano",  "Flute"),
        ("guitar",           "synth",  "Guitar"),
        ("acoustic guitar",  "piano",  "Acoustic Guitar"),
        ("trumpet",          "synth",  "Trumpet"),
        ("brass",            "synth",  "Brass"),
        ("horn",             "synth",  "Horn"),
        ("sax",              "epiano", "Saxophone"),
        ("saxophone",        "epiano", "Saxophone"),
        ("harp",             "piano",  "Harp"),
        ("bells",            "piano",  "Bells"),
        ("marimba",          "piano",  "Marimba"),
        ("vibraphone",       "epiano", "Vibraphone"),
        ("xylophone",        "piano",  "Xylophone"),
        ("clarinet",         "piano",  "Clarinet"),
        ("banjo",            "piano",  "Banjo"),
        ("ukulele",          "piano",  "Ukulele"),
    ]
    for kw, inst_type, inst_label in EXTRA_INSTS:
        if kw not in style_combined:
            continue
        if inst_type == mel_inst and kw in mel_inst:
            continue

        extra_notes  = []
        pad_oct      = 48 + (12 if random.random() < 0.5 else 0)
        for bar in range(total_bars):
            chord_idx  = (bar // 2) % len(chord_prog)
            chord_root = (key_root + chord_prog[chord_idx]) % 12
            intervals  = build_chord(chord_root, is_minor, add7_prob=0.2)
            for j, interval in enumerate(intervals):
                midi_note = pad_oct + interval
                t_on = bar * 4.0 + j * 0.06
                extra_notes.append(make_note(midi_note, t_on, 3.6, humanize(72, 8)))

        tracks.append({"name": inst_label, "instrument": inst_type,
                       "notes": extra_notes, "total_beats": total_bars * 4})
        print(f"[midi] Added extra track: {inst_label}", flush=True)

    return {
        "midi_tracks": tracks,
        "tempo_bpm":   bpm,
        "key":         raw_key,
        "scale":       "minor" if is_minor else "major",
        "total_bars":  total_bars,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── MIDI mode — ACE-Step generation + Basic Pitch transcription ────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Stems eligible for MIDI transcription (pitched instruments only — no drums/percs)
_MIDI_TRANSCRIBE_STEMS = {"bass", "piano", "keyboard", "guitar", "synth", "strings",
                           "brass", "woodwinds"}

# ACE-Step extract track name for each stem
_MIDI_EXTRACT_TRACK = {
    "bass":      "bass",
    "piano":     "keyboard",
    "keyboard":  "keyboard",
    "guitar":    "guitar",
    "synth":     "synth",
    "strings":   "strings",
    "brass":     "brass",
    "woodwinds": "woodwinds",
}


def generate_midi_from_audio(job_input: dict) -> dict:
    """
    MIDI mode with real transcription via Basic Pitch:
    1. Generate a full beat with ACE-Step (text2music, shorter duration for speed)
    2. Extract each pitched stem with task_type="extract"
    3. Transcribe each stem to MIDI via Basic Pitch (real notes from the audio)
    4. Add hardcoded drum MIDI grid (Basic Pitch is a pitch tracker, not a drum detector)
    5. Return MIDI tracks + main beat audio for the preview player
    """
    dit, dit_base, llm = get_handlers()

    genre  = job_input.get("genre",  "").strip().lower()
    mood   = job_input.get("mood",   "").strip().lower()
    bpm    = int(job_input.get("bpm") or 120)
    key    = (job_input.get("key") or "").strip()
    prompt = (job_input.get("prompt") or job_input.get("style") or "").strip()
    style  = job_input.get("style", "").strip()
    user_p = prompt.lower()

    # Use shorter duration in MIDI mode (extraction + transcription is slower)
    _dur_raw = job_input.get("duration")
    duration = float(_dur_raw) if _dur_raw else 90.0
    duration = max(60.0, min(120.0, duration))

    tags       = build_tags(prompt, genre, mood, bpm, key)
    lyrics_str = _build_lyrics_structure(duration, prompt, genre)
    seed       = random.randint(0, 2**31 - 1)
    g_scale    = _get_guidance_scale(genre, prompt)

    # ── 1. Generate full beat ─────────────────────────────────────────────────
    print(f"[midi-gen] Generating {duration:.0f}s beat for transcription...", flush=True)
    full_params = GenerationParams(
        task_type       = "text2music",
        caption         = tags,
        lyrics          = lyrics_str,
        duration        = int(duration),
        bpm             = bpm,
        keyscale        = key if key else "N/A",
        timesignature   = "4/4",
        inference_steps = 20,
        guidance_scale  = g_scale,
        seed            = seed,
    )
    full_beat_path = _ace_generate(dit, llm, full_params, "/tmp/bh_midi_main.wav")
    main_audio     = _read_audio(full_beat_path)
    actual_dur     = (len(main_audio) if main_audio.ndim == 1 else main_audio.shape[0]) / SAMPLE_RATE
    total_beats    = round(actual_dur * bpm / 60)
    print(f"[midi-gen] Beat done — {actual_dur:.1f}s", flush=True)

    # ── 2. Separate stems with Demucs + transcribe all stems to MIDI ──────────
    print("[midi-gen] Separating stems with Demucs for MIDI transcription...", flush=True)
    transcribed_tracks = []
    demucs_stems       = {}   # kept in outer scope so step 3 can access drum stem
    if _DEMUCS_OK:
        try:
            demucs_stems = separate_stems_demucs(full_beat_path)
            # Transcribe every pitched stem with Basic Pitch
            _MIDI_PITCH_STEMS = {"bass", "guitar", "piano", "synth"}
            for bh_key, (stem_mono, display_name, d_sr) in demucs_stems.items():
                if bh_key not in _MIDI_PITCH_STEMS:
                    continue
                stem_tmp = f"/tmp/bh_midi_demucs_{bh_key}.wav"
                sf.write(stem_tmp, stem_mono, d_sr)
                notes = transcribe_to_midi(stem_tmp, bpm, bh_key)
                if notes:
                    inst_type = _DEMUCS_INST_TYPE.get(bh_key, "synth")
                    transcribed_tracks.append({
                        "name":        display_name,
                        "instrument":  inst_type,
                        "notes":       notes,
                        "total_beats": total_beats,
                    })
                    print(f"[midi-gen] '{bh_key}' → {len(notes)} MIDI notes", flush=True)
        except Exception as e:
            print(f"[midi-gen] Demucs separation failed: {e}", flush=True)
    else:
        print("[midi-gen] Demucs unavailable — no MIDI transcription", flush=True)

    # ── 3. Drum MIDI from Demucs drum stem (onset detection + freq analysis) ──
    no_drums = any(x in user_p for x in ["no drums", "without drums", "drumless", "geen drums"])
    drum_tracks = []
    if not no_drums:
        if "drums" in demucs_stems:
            try:
                drum_mono, _, d_sr = demucs_stems["drums"]
                drum_tmp = "/tmp/bh_midi_demucs_drums.wav"
                sf.write(drum_tmp, drum_mono, d_sr)
                drum_notes = transcribe_drums_from_stem(drum_tmp, bpm)
                if drum_notes:
                    drum_tracks = [{
                        "name":        "Drums",
                        "instrument":  "drums",
                        "notes":       drum_notes,
                        "total_beats": total_beats,
                    }]
                    print(f"[midi-gen] drums → {len(drum_notes)} MIDI hits", flush=True)
            except Exception as e:
                print(f"[midi-gen] Drum transcription failed: {e}", flush=True)
        if not drum_tracks:
            # Fallback to genre-based grid if Demucs drum stem unavailable
            print("[midi-gen] Falling back to hardcoded drum grid", flush=True)
            drum_job    = {**job_input, "duration": actual_dur}
            drum_result = generate_midi_tracks(drum_job)
            drum_tracks = [t for t in drum_result["midi_tracks"] if t["instrument"] == "drums"]

    # Drums first, then transcribed melodic tracks
    all_tracks = drum_tracks + transcribed_tracks

    return {
        "midi_tracks":      all_tracks,
        "wav_base64":       np_to_wav_b64(main_audio),
        "duration_seconds": round(actual_dur, 2),
        "tempo_bpm":        bpm,
        "key":              key,
        "scale":            "minor" if genre in MINOR_GENRES else "major",
        "total_bars":       total_beats // 4,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# ── RunPod handler ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def handler(job: dict) -> dict:
    job_input   = job.get("input", {})
    output_mode = job_input.get("output_mode", "audio")

    print(f"[job] mode={output_mode}", flush=True)

    try:
        if output_mode == "midi":
            if _BASIC_PITCH_OK:
                # Real MIDI: generate beat → extract stems → Basic Pitch transcription
                return generate_midi_from_audio(job_input)
            else:
                # Fallback: hardcoded MIDI patterns + short preview audio
                print("[job] Basic Pitch unavailable — falling back to hardcoded MIDI", flush=True)
                midi_result = generate_midi_tracks(job_input)
                wav_b64, dur = generate_preview_audio(job_input)
                midi_result["wav_base64"]       = wav_b64
                midi_result["duration_seconds"] = dur
                return midi_result
        else:
            return generate_audio_with_stems(job_input)
    except Exception as e:
        import traceback
        print(f"[job] ERROR: {e}\n{traceback.format_exc()}", flush=True)
        return {"error": str(e)}


print("[startup] Handler ready — waiting for jobs", flush=True)
runpod.serverless.start({"handler": handler})
