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
    try:
        from acestep.inference import (
            create_sample as _ace_create_sample,
            format_sample as _ace_format_sample,
        )
    except ImportError:
        _ace_create_sample = None
        _ace_format_sample = None
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
        "hard trap banger, thunderous 808 sub kick with deep portamento pitch slide on beat 1, rapid-fire 32nd-note triplet hi-hat rolls stuttering from closed to open, velocity-modulated hi-hat attacks accelerating into snare, hard snare CRACK on beat 3 with heavy room reverb, syncopated kick ghost hits, menacing dark Atlanta atmosphere, aggressive and relentless",
        "dark Atlanta trap, deep distorted 808 bass with long sustain and saturation, 32nd-note triplet hi-hat rolls with velocity dynamics and hi-hat flams, thunderous sub kick on beat 1 and syncopated additional kicks, hard reverb-drenched snare crack on beat 3, chopped soul sample, ominous minor melody, cold threatening energy",
        "melodic trap, emotional piano loop over thunderous 808 sub kick with portamento slides, rolling 32nd-note triplet hi-hats building from closed to open, hard cracking snare beat 3 layered with clap, heavy saturation on 808, introspective dark energy, sad atmospheric trap with relentless hi-hat aggression",
        "cinematic trap, orchestral strings over massive sub-frequency 808 kick with pitch slides, rapid-fire hi-hat triplet rolls stuttering open, hard snare crack beat 3 with room reverb, epic dark production, cold dramatic atmosphere, aggressive dark energy",
        "aggressive trap banger, explosive overdriven 808 bass drum with deep sub rumble and portamento, relentless 32nd-note triplet hi-hat patterns with velocity-modulated stuttering flams, very sparse kick on beat 1 plus off-beat syncopation, hard reverb-soaked snare crack beat 3, dark threatening Atlanta sound, menacing and powerful",
        "drill-influenced dark trap, deep sliding 808 kick with heavy distortion and long sustain, rapid-fire hi-hat triplet rolls with hi-hat flams and stuttering dynamics, extra syncopated kick hits, hard snare crack beat 3 with layered clap, cold ominous atmosphere, aggressive street energy",
        "late-night trap, massive detuned 808 sub kick with portamento pitch slides, haunting piano sample, 32nd-note hi-hat rolls building tension with velocity dynamics, punchy dry snare crack beat 3 with reverb, deep dark brooding groove, heavy 808 saturation, cold introspective aggression",
        "club trap banger, heavy saturation-crushed 808 kick with sub-frequency punch and pitch slides, relentless rapid-fire hi-hat triplet rolls stuttering from closed to open with flams, very sparse kick pattern on beat 1, explosive snare crack on beat 3 with heavy reverb and layered clap, peak-hour dark energy, aggressive menacing club atmosphere",
    ],
    "drill": [
        "UK drill, deep sliding 808 kick drum with dramatic double-time glide, SPARSE cold staccato 16th-note hi-hats with long silences between, sharp crisp snare with short tight decay, irregular off-beat kick placement, ominous minor string stabs, dark gritty cold London street sound, 140-150 BPM, space and silence are essential",
        "Brooklyn NY drill, heavy booming 808 bass with long dramatic pitch glide, minimal cold sparse hi-hat pattern with staccato hits and empty space, tight hard dry snare, eerie piano loop, dark aggressive production, irregular kick pattern not on 1 and 3, cold menacing street energy",
        "Chicago drill, menacing minor chords, booming 808 kick with dramatic slow pitch slides, cold staccato sparse hi-hats with deliberate silence, hard punchy snare short decay, irregular thunderous off-beat percussion, dark threatening Chicago street atmosphere, 140 BPM",
        "melodic UK drill, haunting flute or violin loop over deep sliding 808 kicks with dramatic portamento glides, minimal cold staccato hi-hats with long pauses, crisp hard snare short decay, cold emotional energy, irregular kick placement, dark drill groove, 145 BPM sparse feel",
        "aggressive drill, cold brutal sliding 808 with extra-long dramatic glides, sharp staccato minimal hi-hats with maximum space between hits, hard punchy snare crack, irregular kick pattern creating unpredictable rhythm, menacing dark aggressive drill, 140-150 BPM",
        "dark drill, ominous atmosphere with deep sliding 808 sub kicks and long dramatic portamento, extremely sparse cold hi-hat pattern, hard dry snare, irregular kick placement far from standard beat 1 and 3, dark cinematic strings, cold threatening energy, 145 BPM",
        "cold drill, ice-cold atmosphere, very sparse staccato hi-hats separated by long silence, massive 808 slides with dramatic glide, minimal cold snare, irregular off-beat 808 sub hits, no melodic warmth, cold mechanical drill production, 140 BPM",
    ],
    "hip hop": [
        "soulful hip hop, punchy sampled kick on beats 1 and 3, crisp snare on 2 and 4, swinging 16th-note hi-hats with groove, warm vinyl-sampled piano, deep punchy bass pocket, 90 BPM",
        "dusty boom bap-influenced hip hop, hard knocking kick and snare, loose swinging hi-hats with shuffle feel, muffled soul chop, warm analog texture, vinyl crackle, head-nodding groove, 92 BPM",
        "dark hip hop, boom bap kick and snare pattern, cinematic orchestral sample, swinging hi-hats with ghost notes, underground raw feel, ominous bass line",
        "smooth hip hop, laid-back kick groove on 1 and 3, crisp snare 2 and 4, shuffled hi-hats, lush Rhodes melody, relaxed head-nodding vibe, warm deep bass",
        "raw hip hop, heavy punchy kick, cracking snare, gritty sampled horn stab, swinging hi-hat pattern, street authenticity, deep bass pocket, urban energy",
    ],
    "hip-hop": [
        "soulful hip hop, punchy sampled kick on beats 1 and 3, crisp snare on 2 and 4, swinging 16th-note hi-hats, warm vinyl-sampled piano, deep groove, 90 BPM",
        "dark hip hop, hard knocking kick and snare, loose swinging hi-hats, cinematic orchestral sample, boom bap rhythm, underground feel",
        "smooth hip hop, laid-back kick and snare pocket, shuffled hi-hats, lush Rhodes melody, relaxed head-nodding vibe",
        "raw gritty hip hop, heavy punchy kick, cracking sampled snare, gritty horn stab, swinging hi-hats, street authenticity",
    ],
    "boom bap": [
        "classic boom bap, deep sampled kick drum from vinyl on beats 1 and 3 with natural organic feel, hard rimshot snare on 2 and 4, loose swinging hi-hats with ghost notes not perfectly quantized, dusty vinyl soul sample, vinyl crackle texture, 90 BPM loose swing",
        "golden era boom bap, heavy sampled drum break from vinyl record with natural imperfection, swinging hi-hat groove with choke and ghost notes, soulful horn sample, jazz-infused dusty groove, warm analog compression, not quantized naturally loose feel",
        "gritty boom bap, punchy compressed sampled kick with vinyl snap, cracking rimshot snare from sample, loose triplet hi-hat swing not quantized, chopped vocal sample, crackly vinyl texture, underground rawness, sampled bass not 808",
        "cinematic boom bap, deep sampled drum break from vinyl, swinging hi-hats with fills and ghost notes, orchestral brass sample, booming sampled kick, timeless old-school feel, warm analog texture, naturally swinging rhythm",
        "hard boom bap, heavy knocking sampled kick on 1 and 3, loud cracking rimshot snare 2 and 4, tight hi-hat pattern with swing, aggressive underground hip hop energy, vinyl crackle, sampled bass groove",
    ],
    "r&b": [
        "silky smooth R&B, smooth kick and snare pocket groove, shuffled hi-hat with ghost notes, warm Fender Rhodes chords, lush reverb, intimate late-night feel, 80 BPM",
        "neo-soul R&B, laid-back kick on 1 and 3, snare ghost notes on 2 and 4, swinging hi-hats, live electric guitar licks, sensual atmosphere, complex chord voicings",
        "dark R&B, deep kick groove, snare on 2 and 4 with fills, shuffled hi-hats, minor key piano, deep bass pulse, moody emotional tension, brooding atmosphere",
        "contemporary R&B, punchy modern kick pattern, sharp snare, 16th-note hi-hat drive, lush pad harmonies, polished feel, warm melodic hooks",
        "soulful R&B, warm pocket kick and snare, swinging hi-hats, gospel-influenced chord progressions, warm vintage keys, heartfelt authentic expression",
    ],
    "afrobeats": [
        "vibrant afrobeats, syncopated kick pattern, talking drum accents, layered conga and shaker, off-beat hi-hat groove, bright guitar melody, joyful West African rhythm, 105 BPM",
        "afrobeats banger, complex afro polyrhythm, layered hand percussion, syncopated kick, catchy synth hook, infectious high-energy arrangement, vibrant percussion layers",
        "amapiano-influenced afrobeats, log drum bass hits, hypnotic layered percussion groove, talking drum, shaker, South African sound, deep bass",
        "afro fusion, vibrant syncopated percussion layers, kora-inspired melody, rich polyrhythm, conga and bongo groove, warm organic feel, West African roots",
    ],
    "dancehall": [
        "riddim dancehall, punchy kick on beat 1 and and-beat-3, snare on 2 and 4, offbeat hi-hat chop, deep digital bass, Caribbean energy, 100 BPM",
        "modern dancehall, trap-influenced drum pattern, melodic synth hook, punchy kick, snare, danceable Caribbean groove, vibrant rhythm",
        "lovers rock dancehall, smooth reggae-inspired kick and snare, offbeat hi-hat, warm bass, romantic Caribbean vibe, soulful feel",
    ],
    "lo-fi": [
        "lo-fi hip hop, muffled kick with soft attack, brushed snare with vinyl grain not quantized, loose dusty hi-hats, mellow Rhodes melody, late night studying atmosphere, 80 BPM vinyl crackle",
        "rainy day lo-fi, dusty muffled kick drum, soft brushed snare decay with natural imperfection, gentle hi-hats, warm tape saturation, sleepy piano chords, nostalgic haze, vinyl texture",
        "jazzy lo-fi, laid-back kick and brushed snare groove with swing, dusty vinyl hi-hats, muted guitar chords, vinyl grain texture, cozy introspective atmosphere, 75 BPM",
        "lo-fi soul, gentle kick drum, soft brushed snare, warm upright bass sampled from vinyl, dusty hi-hats, Rhodes chords, slow meditative groove, tape warmth",
        "lo-fi chill, relaxed kick and snare pattern, soft hi-hats, vocal chop sample, mellow analog warmth, drifting daydream feel, cassette tape texture",
    ],
    "electronic": [
        "cutting-edge electronic, driving kick on every beat, aggressive 16th-note hi-hats, layered synthesizer arpeggios, explosive driving pulse, massive synth stabs, full arrangement",
        "experimental electronic, complex mechanical percussion pattern, textured modular synth, evolving complex pads, intense futuristic atmosphere, relentless forward motion",
        "dark electronic, cold industrial kick pattern, hypnotic relentless percussion, heavy bass punch, industrial synth, underground club power, dark atmosphere",
        "melodic electronic, four-on-the-floor kick drive, crisp clap on backbeats, soaring emotional synth lead, lush atmospheric depth, euphoric energy",
        "hard electronic, brutal crushing percussion hits, distorted bass pressure, relentless drum pattern, intense aggressive energy, raw club force",
    ],
    "house": [
        "classic Chicago house, stomping four-on-the-floor kick hitting EVERY beat without exception, open hi-hat on every 8th-note offbeat, crisp clap on 2 and 4, punchy soulful piano chords, warm analog bassline separate from kick, peak-hour dancefloor energy, 126 BPM",
        "funky house, driving four-on-the-floor kick on every beat, syncopated open hi-hat on offbeats, punchy clap on backbeats 2 and 4, chopped vocal stab, thick deep melodic bass line, lively dancefloor energy, 128 BPM",
        "melodic house, powerful four-on-the-floor kick on every beat, open hi-hat offbeat groove, euphoric emotional piano chord progression, soaring lush pads, crowd-lifting energy, 124 BPM",
        "afro house, driving four-on-the-floor kick every beat, hypnotic tribal conga and shaker percussion layers, open hi-hat offbeats, heavy groove, driving deep bassline, spiritual dancefloor power, 124 BPM",
        "peak-hour house, aggressive four-on-the-floor kick on every beat, syncopated open hi-hat and clap on 2 and 4, relentless bass stab separate melody, intense peak club energy, full powerful production, 130 BPM",
        "deep-influenced house, warm four-on-the-floor kick every beat, gentle open hi-hat offbeats, clap on 2 and 4, soulful jazzy chord progression, warm melodic bassline, late-night intimate house groove, 122 BPM",
    ],
    "deep house": [
        "deep house, soft four-on-the-floor kick on every beat with warm attack, subtle open hi-hat on offbeats, soft clap on backbeats, warm sub bass melodic bassline, rich atmospheric pads, late-night groove, 122 BPM",
        "soulful deep house, rolling four-on-the-floor kick every beat, gentle swinging hi-hat groove on offbeats, dusty organ chords, rolling punchy deep melodic bass, emotional intimate dancefloor energy, 120 BPM",
        "minimal deep house, soft kick on every beat, subtle hi-hat offbeat, hypnotic groove, deep resonant bass thump melodic line, dark underground tension, 120 BPM",
        "vocal deep house, rolling kick every beat, gentle hi-hat swing on offbeats, emotional chord progression, lush reverb depth, late night warmth, 124 BPM",
    ],
    "tech house": [
        "driving tech house, hard punchy kick on every beat, percussive hi-hat groove with rim shot accents, menacing hypnotic bassline, dark relentless underground groove, peak-hour intensity, 130 BPM",
        "funky tech house, heavy mechanical four-on-the-floor kick, aggressive syncopated hi-hat, thick filtered bass stab, hard rim percussion, high-energy arrangement, 128 BPM",
        "industrial tech house, crushing distorted kick every beat, industrial hi-hat pattern, heavy percussion hits, raw club energy, intense mechanical force, 132 BPM",
        "hard tech house, aggressive bass pressure, relentless four-on-the-floor groove, dystopian underground energy, full powerful production, 130 BPM",
    ],
    "techno": [
        "dark Berlin techno, HARD industrial kick drum on EVERY single beat at 140 BPM, distorted not smooth kick, aggressive 16th-note hi-hats relentless mechanical throughout, heavy metallic percussion, very minimal cold melody, cold mechanical dystopian energy",
        "hard techno, massive hard distorted kick every single beat at 145 BPM, rapid mechanical 16th-note hi-hats throughout, industrial percussion texture, aggressive distortion, brutal driving intensity, no melody cold and mechanical",
        "melodic techno, relentless industrial hard kick on every beat at 138 BPM, 16th-note hi-hat mechanical drive throughout, soaring emotional synth lead over cold industrial rhythm, building cathartic peak-hour energy",
        "minimal techno, hard pounding industrial kick drum on every beat at 132 BPM, hypnotic 16th-note hi-hat groove throughout, deep resonant bass, rising tension, dark underground drive, very minimal production",
        "industrial techno, crushing distorted kick drum every beat at 148 BPM, 16th-note metallic hi-hats unrelenting, industrial noise texture, cold robotic dystopian atmosphere, no warmth, mechanical and brutal",
    ],
    "edm": [
        "festival EDM, four-on-the-floor kick at 128 BPM, punchy clap on 2 and 4, build-up snare roll into massive drop, massive soaring synth lead, explosive epic stadium production",
        "progressive EDM, driving kick on every beat, clap on backbeats, pre-drop snare fill, emotional chord progression, huge soaring synth lead, euphoric breakdown into massive drop",
        "electro EDM, punchy kick every beat, sharp clap, retro-futuristic synth stab, high-energy peak-hour intensity, full energetic production",
        "big room EDM, crushing four-on-the-floor kick drop, punchy clap, huge chord stabs, massive build tension release, overwhelming crowd energy, 130 BPM",
        "electro house EDM, driving four-on-the-floor kick, punchy distorted bass, relentless clap and hi-hat, explosive drop energy, high-intensity production",
    ],
    "dubstep": [
        "heavy riddim dubstep, HALF-TIME feel at 140 BPM — kick on beat 1 only, MASSIVE reverb-soaked snare on beat 3 ONLY, SPARSE hi-hats few and far between, crushing Reese bass wobble with LFO filter, devastating sub-bass drop, filthy modulated bass growl, half-time groove is defining feature",
        "brostep dubstep, half-time drum pattern at 140 BPM, kick beat 1, enormous snare crack on beat 3 only with massive reverb, sparse hi-hats, massive distorted growl bass with LFO wobble, face-melting drop intensity, full-frequency destruction",
        "melodic dubstep, half-time kick beat 1 and massive snare beat 3 at 140 BPM, sparse hi-hats, soaring emotional synth lead, cinematic orchestral build, explosive huge cathartic drop with wobble bass",
        "dark dubstep, half-time kick pattern beat 1 at 140 BPM, bone-crushing reverb snare beat 3 only, very sparse hi-hats, ominous cinematic tension build, industrial bass growl with LFO, dystopian overwhelming energy",
        "neuro dubstep, half-time drum pattern at 140 BPM, kick beat 1 massive snare beat 3, sparse hi-hats, complex technical bass sound design with LFO modulation, intense mechanical energy, deep sub pressure",
    ],
    "drum and bass": [
        "liquid drum and bass, rapid 170 BPM syncopated amen break-style breakbeat, rolling syncopated kick and sharp snare fills not on 1 and 3, fast 16th-note hi-hats, Reese sub bass, soulful Rhodes melody, smooth atmospheric feel",
        "dark drum and bass, frantic amen break at 170 BPM, syncopated kick and snare with breakbeat complexity, rapid hi-hat drive, ominous Reese bass wobble, intense underground energy",
        "neurofunk drum and bass, precise breakbeat at 170 BPM, technical syncopated percussion with complex amen break pattern, complex Reese bass modulation, futuristic intense mechanical energy",
        "jump up drum and bass, energetic amen-style break at 170 BPM, fast syncopated kick and snare, heavyweight punchy bass stab, crowd-moving energy, rolling breakbeat",
        "hard drum and bass, relentless fast breakbeat at 170 BPM, rapid syncopated percussion, crushing Reese sub bass pressure, intense raw underground power, amen break pattern",
    ],
    "dnb": [
        "liquid DnB, rolling 170 BPM amen break syncopated pattern, syncopated kick and snare, rapid hi-hats, Reese sub bass, warm atmospheric pads, soulful melodic forward momentum",
        "dark DnB, frantic breakbeat at 170 BPM, aggressive syncopated percussion, rapid amen break pattern, Reese bass, relentless underground raw energy",
        "neurofunk DnB, precise 170 BPM breakbeat, technical syncopated percussion, Reese bass clinical design, cold futuristic atmosphere",
        "jump up DnB, energetic fast amen break at 170 BPM, syncopated kick pattern, sharp snare hits, rapid hi-hat rolls, heavy punchy bass, high-intensity dancefloor power",
    ],
    "jungle": [
        "classic jungle, frantic chopped amen break with syncopated complexity, rapid syncopated kick and snare layering, reggae bass wobble, raw 90s underground energy, rhythmic intensity, 160 BPM",
        "dark jungle, ominous frantic amen break pattern, rapid syncopated percussion complexity, intense rhythmic drive, powerful underground energy, reggae bass",
    ],
    "reggaeton": [
        "Latin reggaeton, dembow kick on beat 1 and the-and-of-beat-3, snare on beats 2 and 4, shuffled hi-hat, bright synth melody, urban Caribbean groove, 95 BPM",
        "dark reggaeton, aggressive dembow kick pattern, snare on backbeats, shuffled hi-hat, minor key melody, street urban Latin energy",
        "romantic reggaeton, punchy dembow kick and snare, smooth hi-hat groove, smooth melody, warm production, sensual Latin groove",
    ],
    "latin": [
        "vibrant Latin beat, layered conga and bongo patterns, timbales, shaker and claves, complex polyrhythmic Latin percussion, bright brass stabs, festive rhythmic energy",
        "Latin jazz fusion, complex polyrhythm with congas, timbales, and piano, complex chord voicings, rich harmonic groove, 110 BPM",
        "dark Latin, minor key guitar over layered Latin percussion, conga and shaker, moody atmosphere, intense rhythmic tension",
    ],
    "jazz": [
        "late-night jazz, swinging jazz ride cymbal, brushed snare with ghost notes, kick on downbeats, hi-hat on 2 and 4, smoky walking bass, complex chord voicings, intimate atmosphere",
        "jazz fusion, live jazz drum kit with swing feel, ride cymbal pattern, ghost note snare, electric piano, intricate rhythmic interplay, improvisational energy",
        "dark jazz, loose swinging brushed drums, dissonant chords, sparse minor melody, noir cinematic atmosphere",
        "upbeat jazz, driving bebop kick and snare, swinging ride cymbal, brass section, lively groove, classic bebop energy",
    ],
    "soul": [
        "deep soul, tight groove kick on 1 and 3, snare on 2 and 4 with pocket, shuffled hi-hat, warm organ chords, gospel-influenced melody, heartfelt emotional expression, 80 BPM",
        "classic soul, punchy Motown kick and snare groove, swinging hi-hats, lush string arrangement, timeless emotional depth, warm production",
        "neo-soul, laid-back kick and snare pocket, swinging hi-hats, complex jazz chords, live bass groove, introspective warm atmosphere",
        "southern soul, driving kick and snare, syncopated hi-hat, raw emotional guitar licks, deep groove, authentic feel",
    ],
    "funk": [
        "classic funk, syncopated kick on the one, tight ghost-note snare, open hi-hat choke pattern, slapped bass groove, wah guitar, deep pocket, James Brown influence, 105 BPM",
        "jazz funk, driving funky kick pattern, crisp snare with ghost notes, open hi-hat rhythm, complex chord progression, electric piano, sophisticated groove",
        "dark funk, syncopated minor key kick groove, tight snare, heavy bass line, tense rhythmic interplay",
        "P-funk inspired, deep funky kick and snare groove, layered percussion, spacey synth over heavy groove, cosmic feel",
    ],
    "pop": [
        "polished pop, punchy four-on-the-floor kick, sharp snare on 2 and 4, 16th-note hi-hat drive, catchy piano hook, modern production, hook-driven radio sound, 120 BPM",
        "dark pop, minor key drum groove, emotional kick and snare pattern, hi-hat drive, minor key emotional melody, lush production",
        "indie pop, live drum feel with organic kick and snare, natural hi-hat groove, warm melody, bittersweet emotional atmosphere",
        "electropop, driving electronic kick every beat, punchy clap, bright synth melody, contemporary high-energy production",
    ],
    "cinematic": [
        "epic cinematic score, orchestral percussion with timpani and taiko drums, dramatic snare rolls, sweeping orchestral strings, powerful tension and release, full orchestra",
        "dark cinematic, dramatic percussion build, heavy timpani hits, dissonant brass stabs, ominous cello, tense suspenseful atmosphere",
        "emotional cinematic, soft orchestral percussion, delicate strings, solo piano over lush strings, heartbreaking melodic theme",
        "action cinematic, driving taiko and snare rolls, aggressive orchestral percussion, heroic brass, relentless forward momentum",
        "ambient cinematic, soft atmospheric percussion, evolving pads, sparse piano motif, vast spacious sound",
    ],
    "ambient": [
        "deep ambient, minimal soft percussion texture, slowly evolving pad layers, vast spacious dreamscape, no defined rhythm",
        "dark ambient, ominous drone, sparse percussion texture, unsettling atmospheric tension, cinematic darkness",
        "ambient electronic, soft rhythmic pulse, crystalline synth arpeggios, peaceful floating atmosphere",
        "nature-inspired ambient, organic textural rhythm, gentle melodic motif, serene evolving soundscape",
    ],
    "phonk": [
        "dark Memphis phonk, heavy distorted overdriven 808 kick with crunchy saturation, cowbell on every upbeat and offbeat — cowbell is essential, aggressive rolling hi-hats with Memphis character, chopped soul vocal, heavily distorted 808 bass, vintage cassette tape texture, 140 BPM",
        "aggressive phonk, crunchy overdriven distorted 808 kick with heavy saturation, rapid rolling hi-hat triplets with cowbell accents on upbeats, hard snare slam, dark chord sample, cowbell is defining element, raw underground Memphis energy",
        "drift phonk, hard overdriven distorted 808 bass drum with cassette saturation, rolling hi-hats and prominent cowbell on offbeats, dark atmospheric synth, high-energy intense momentum, Memphis rap roots, distorted crunchy 808",
        "melodic phonk, heavy overdriven distorted 808 kick, cowbell pattern on offbeats with hi-hats, haunting vocal chop, distorted saturated bass, eerie dark production, cassette tape texture, cowbell essential",
        "Memphis phonk, raw distorted 808 kick with tape saturation and crunch, cowbell hitting on every upbeat, rolling hi-hats, chopped rap vocal sample, dark Memphis atmosphere, crunchy overdriven bass",
        "hard phonk, crushing overdriven 808 kick with extreme distortion and saturation, fast rolling hi-hats with cowbell on offbeats, slamming snare, dark threatening atmosphere, cowbell and distorted 808 define the sound",
    ],
    "cloud rap": [
        "hazy cloud rap, slow minimal trap drums, sparse reverb-soaked kick and distant reverb snare, soft hi-hats, washed-out ambient trap, atmospheric reverb-drenched melody, ethereal feel, 130 BPM",
        "dark cloud rap, slow muffled kick, distant reverb snare, ominous pad texture, soft hi-hats, cold atmospheric trap, heavy reverb",
        "melodic cloud rap, slow hazy drum pattern, muffled kick and soft reverb snare, emotional piano under heavy reverb, dreamy introspective atmosphere",
    ],
    "grime": [
        "raw UK grime, staccato 140 BPM kick pattern, sharp metallic hi-hats, hard snare hit, staccato synth stabs, rolling 8-bar beat, aggressive dark energy, cold London sound",
        "dark grime, aggressive 140 BPM drum pattern, cold hi-hat rhythm, hard snare, minor key synth riff, sparse industrial rhythm, cold dark atmosphere",
        "melodic grime, driving 140 BPM percussion, rapid hi-hat, hard snare, emotional keyboard lead, intense urban energy",
    ],
    "uk garage": [
        "UK garage, 2-step shuffle kick on beat 1 and and-beat-3, snare on 2 and 4, syncopated swinging hi-hat groove, pitched vocal chop, warm bass, soulful house influence, 132 BPM",
        "speed garage, shuffled 2-step kick, swinging hi-hat, deep sub bass, skippy hi-hat rhythm, late-night underground groove",
        "dark UK garage, shuffled 2-step drum pattern, minor key synth, swinging hi-hats, moody introspective atmosphere",
    ],
    "synthwave": [
        "retro synthwave, driving electronic drum machine pattern, four-on-the-floor kick, massive gated reverb snare on 2 and 4, 16th-note closed hi-hats, pulsing analog bass, neon-lit arpeggios, 80s cinematic atmosphere, 120 BPM",
        "dark synthwave, cold electronic drum pattern, hard gated reverb snare, driving kick, cold minor key synth, dystopian Blade Runner atmosphere",
        "melodic synthwave, electronic drum machine groove, gated snare reverb, driving kick, emotional lead synth, lush reverb, romantic 80s nostalgia",
        "outrun synthwave, powerful electronic drum pattern, four-on-the-floor kick, gated reverb snare, powerful bass pulse, heroic lead melody",
    ],
    "vaporwave": [
        "vaporwave, slow hazy drum machine pattern, slowed pitched-down melody, lush reverb wash, nostalgic 80s corporate dream, 80 BPM",
        "dark vaporwave, unsettling slow drum pattern, slowed sample, dreamlike distortion, lonely atmosphere",
        "mallsoft vaporwave, ambient minimal drum texture, surreal nostalgic sound, deconstructed elevator music",
    ],
    "hyperpop": [
        "hyperpop, distorted glitchy drum pattern, heavy distorted 808 kick, glitched snare effects, chaotic hi-hat fills, maximalist layered production, chaotic hyper-saturated energy",
        "dark hyperpop, aggressive glitched-out drums, heavy distorted kick, stutter snare, pitched vocal glitch, intense hyper-saturated contrast",
        "melodic hyperpop, distorted drum pattern, heavy glitch kick, emotional lead over glitchy production, bittersweet energy",
    ],
    "pluggnb": [
        "dark pluggnb, slow melodic trap drums, heavy 808 sub kick, soft snare on beat 3, gentle hi-hat pattern, smooth minor key melody, emotional introspective feel, 130 BPM",
        "pluggnb, laid-back slow trap groove, deep 808 kick, minimal hi-hats, lush pad chords, slow dark 808 bass, hazy romantic atmosphere",
        "melodic pluggnb, slow trap drum groove, heavy 808 kick, minimal percussion, warm piano over slow trap rhythm, intimate emotional depth",
    ],
    "jersey club": [
        "jersey club, frantic 4x4 kick pattern at 130+ BPM, rapid hi-hat chopping, syncopated snare hits, chopped vocal sample, high-energy dancefloor groove",
        "dark jersey club, fast four-on-the-floor kick, chopped percussion pattern, rapid hi-hat, ominous bass stab, intense underground club energy",
    ],
    "reggae": [
        "classic reggae, one-drop kick on beat 3 only, rim shot on beats 2 and 4, open hi-hat on offbeats, offbeat guitar skank, warm bass, roots vibration, 80 BPM",
        "dub reggae, sparse one-drop kick pattern, echo-soaked snare, heavy bass, delay effects, hypnotic trippy feel",
        "dancehall reggae, punchy digital riddim kick and snare, offbeat hi-hat chop, modern Caribbean production energy",
    ],
    "gospel": [
        "powerful gospel, driving kick and snare groove, full choir harmony, driving rhythm, uplifting spiritual energy, 90 BPM",
        "contemporary gospel, modern punchy drum production, emotional piano, soulful choir texture",
        "dark gospel, driving minor key drum groove, kick and snare pattern, raw emotional power, building drama",
    ],
    "blues": [
        "delta blues, shuffle rhythm kick and snare, driving 12-bar groove, raw electric guitar riff, gritty soulful expression, 80 BPM",
        "modern blues, steady kick and snare groove, swinging hi-hats, emotional lead guitar, deep heartfelt feel",
        "blues jazz fusion, swinging jazz-influenced drum groove, complex chord progression, expressive guitar, sophisticated feel",
    ],
    "rock": [
        "hard rock, powerful live drum kit, hard kick on beats 1 and 3, crashing snare on 2 and 4, driving 8th-note hi-hats, driving electric guitar riff, raw energy, 130 BPM",
        "indie rock, live drum kit feel, punchy kick and snare groove, natural hi-hat rhythm, warm guitar tone, melodic hook",
        "dark rock, heavy live drums, hard kick and crashing snare, aggressive hi-hat, heavy distorted guitar, minor key tension, brooding atmosphere",
    ],
    "rap": [
        "hard rap beat, heavy punchy kick on 1 and 3, cracking snare on 2 and 4, hi-hat drive, dark sample, street energy, raw production, 95 BPM",
        "melodic rap, deep kick groove, crisp snare pattern, swinging hi-hats, emotional piano loop, introspective atmosphere",
    ],
    "metal": [
        "heavy metal, thunderous double-kick drum blast, crashing snare on 2 and 4, rapid 16th-note hi-hats, crushing distorted guitar riffs, massive wall of sound, 160 BPM",
        "death metal, blastbeat double-kick pattern, alternating kick and snare, rapid percussion, crushing heavy distorted guitars, overwhelming power",
        "melodic metal, driving kick pattern, heavy snare, fast hi-hat drive, melodic guitar leads over crushing rhythm guitars, epic power",
    ],
    "amapiano": [
        "amapiano, deep log drum bass hits on every beat, layered piano chords, shuffled hi-hat groove, syncopated percussion, South African house energy, 112 BPM",
        "amapiano banger, punchy log drum kick pattern, piano stab chords, rolling percussion, deep sub bass, vibrant South African groove",
    ],
    "afro house": [
        "afro house, driving four-on-the-floor kick, layered tribal percussion, congas and shakers, deep bassline, spiritual dancefloor energy, 124 BPM",
        "afro house banger, stomping kick every beat, complex afro percussion layers, hypnotic groove, driving bassline, peak-hour energy",
    ],
    "future bass": [
        "future bass, driving electronic kick, powerful clap on backbeats, lush chord swells, bright supersaw synth, emotional atmospheric build, festival energy, 150 BPM",
        "melodic future bass, punchy electronic drums, emotive chord progressions, soaring synth leads, massive euphoric drop, beautiful emotional energy",
    ],
    "trap soul": [
        "trap soul, slow trap drums with heavy 808 sub kick, minimal hi-hats, soft snare on beat 3, warm RnB-influenced chords, emotional dark melody, intimate atmosphere, 70 BPM",
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
_FEEL_TAGS_HIGH_ENERGY = [
    "peak-hour crowd energy, packed dancefloor momentum",
    "explosive concert energy, live performance power",
    "adrenaline-pumping high-speed intensity, non-stop drive",
    "underground club banger, dark room bass pressure",
    "full-throttle production, relentless forward momentum",
    "hard-hitting street energy, raw urban power",
    "stadium anthem energy, massive crowd impact",
    "club ready drop energy, maximum dancefloor force",
    "driving hard rhythm pushing the listener forward",
    "high-impact aggressive production, hard and heavy sound",
    "relentless beat, powerful driving rhythm, full intensity",
    "punchy hard production, every hit lands with force",
]

_FEEL_TAGS_CHILL = [
    "late night 3am energy", "rain on the window",
    "introspective inner thoughts", "driving alone at night",
    "neon lights and fog", "headphone listening music",
    "bedroom studio intimacy", "rooftop sunset session",
    "cinematic storytelling", "raw emotional honesty",
    "luxury late night", "spiritual transcendence",
]

_CHILL_GENRES = {"lo-fi", "ambient", "jazz", "vaporwave", "cloud rap", "deep house"}

# Moods that imply dark/negative emotional tone — energy must NOT override this to happy
_DARK_SAD_MOODS = {
    "sad", "dark", "emotional", "lonely", "pain", "melancholic", "mysterious",
    "angry", "nostalgic", "heavy", "depressed", "heartbreak", "grief", "gloomy",
    "somber", "haunting", "eerie", "bitter", "hurt", "tragic",
}

# Atmospheric feel tags for dark + energetic beats — intense but emotionally heavy
_FEEL_TAGS_DARK_ENERGY = [
    "dark underground rage energy, heavy emotional weight with full power",
    "menacing aggressive intensity, emotional pain channeled into pure force",
    "cinematic dark energy, overwhelming powerful emotion, heavy hard sound",
    "brooding relentless drive, dark minor-key intensity pushing forward hard",
    "cold aggressive power, ominous hard-hitting momentum, dark relentless force",
    "intense somber rage, pain-driven energy, dark powerful production",
    "gritty dark street intensity, heavy emotional power, raw aggression",
    "haunting powerful drive, dark orchestral energy, emotional high-impact sound",
]

def _feel_tags_for_genre(genre: str, is_dark_sad: bool = False, is_energic: bool = False):
    if genre in _CHILL_GENRES:
        return _FEEL_TAGS_CHILL
    if is_dark_sad and is_energic:
        return _FEEL_TAGS_DARK_ENERGY
    return _FEEL_TAGS_HIGH_ENERGY

# High-energy electronic genres — use electronic section labels + higher guidance
_ELECTRONIC_GENRES = {
    "dubstep", "drum and bass", "dnb", "jungle", "techno", "tech house",
    "edm", "electronic", "house", "deep house", "hyperpop", "jersey club",
    "grime", "uk garage",
}

# Mix fullness tags — always injected to push ACE-Step toward richer, harder arrangements
_FULLNESS_TAGS = [
    "multiple hard-hitting layered instruments, full powerful arrangement, every frequency filled with energy",
    "dense aggressive layered production, punchy bass, loud mid-range elements, bright highs, no empty space",
    "full hard-hitting arrangement with loud drums, deep bass, prominent melody, rich atmosphere all pushing forward",
    "richly layered loud mix, thunderous sub bass, punchy mid bass, cutting synth leads, full powerful spectrum",
    "full-bodied hard production, punchy powerful kick, deep distorted bass, energetic melodic layers, full impact",
    "packed high-energy arrangement, multiple simultaneous heavy instrument layers, maximum full frequency impact",
    "dense driving energetic mix, powerful driving rhythm section with loud melodic and harmonic layers on top",
    "full powerful production with prominent hard drums, deep punchy bass groove, cutting lead melody, full impact",
]

# ── Genre-specific drum tags — ALWAYS injected so ACE-Step never forgets the rhythm ──
# These describe the EXACT percussion character of each genre in enough detail
# that the model knows the kick pattern, hi-hat style, and snare placement.
_GENRE_DRUM_TAGS = {
    "trap": [
        "rapid-fire 32nd-note triplet hi-hat rolls with velocity dynamics, stuttering from closed to open hi-hat, hi-hat accelerating and flaming into snare hit, thunderous 808 sub kick with portamento pitch slide on beat 1, very sparse syncopated additional kick hits, hard snare crack on beat 3 with heavy room reverb",
        "relentless 32nd-note triplet hi-hat rolls velocity-modulated from quiet to loud, hi-hat flams and stuttering closed-to-open dynamics, massive 808 sub kick with deep portamento pitch slide beat 1, extra syncopated kick ghost hit, explosive snare crack beat 3 with reverb tail, layered clap on snare",
        "velocity-dynamic 32nd-note hi-hat triplet rolls building from closed to open with flams and stutters, thunderous deeply distorted 808 kick with long portamento slide beat 1, additional off-beat kick syncopation very sparse, hard dry snare crack beat 3 with huge reverb, heavy 808 saturation and sub rumble",
        "aggressive rapid-fire hi-hat triplet rolls at 32nd-note density with velocity swells and hi-hat chokes, massive sub-frequency 808 kick with portamento slide on beat 1 and additional syncopated kick hits, hard snare CRACK beat 3 with room reverb and layered clap, 808 sustain and distortion prominent",
    ],
    "drill": [
        "deep 808 kick drum with long dramatic double-time portamento pitch glide, SPARSE cold staccato hi-hats with deliberate long silence between each hit, sharp crisp dry snare short tight decay, irregular off-beat kick placement not on standard beats, cold minimal UK drill percussion 140 BPM",
        "booming 808 bass kick with extra-long dramatic pitch glide, minimal cold hi-hat hits separated by extended silence, tight hard dry snare beat 3 short decay, irregular kick placement far from standard 1-and-3 pattern, ominous cold drill rhythm 145 BPM",
        "massive 808 sub kick with dramatic slow portamento glide, staccato sparse hi-hat hits with maximum space and silence between, hard punchy snare very short decay, off-beat 808 sub hits irregular and unpredictable, cold menacing minimal drill percussion 140-150 BPM",
        "deep sliding 808 kick with long pitch glide, extremely sparse staccato cold hi-hats far apart, hard snare crack with tight short decay, irregular syncopated kick placement creating unexpected rhythm, cold dark drill drum pattern very minimal 145 BPM",
    ],
    "hip hop": [
        "punchy sampled kick on beats 1 and 3 with natural organic attack, crisp sampled snare on beats 2 and 4, swinging 16th-note hi-hats with groove and slight shuffle, deep hip hop drum pocket 90 BPM",
        "hard knocking boom bap-influenced sampled kick and snare, loose swinging hi-hats with shuffle and slight imperfection, organic natural drum feel not quantized, deep pocket groove with vinyl texture",
        "heavy sampled kick from vinyl with snap, cracking sampled snare with pop, swinging syncopated hi-hats with ghost notes, boom bap hip hop rhythm section warm analog feel",
        "punchy compressed sampled kick on 1 and 3, sharp rimshot snare on 2 and 4, swinging hi-hat groove with choke and ghost notes, dusty hip hop drum pocket warm and analog",
    ],
    "hip-hop": [
        "punchy sampled kick on beats 1 and 3, crisp snare on beats 2 and 4, swinging 16th-note hi-hats, hip hop drum groove with natural organic feel",
        "hard knocking kick and snare with organic feel, loose swinging hi-hats, deep pocket hip hop rhythm section",
        "sampled kick on 1 and 3, cracking snare 2 and 4, swinging hi-hat groove, warm hip hop drum arrangement",
        "punchy kick and crisp snare swing groove, shuffled hi-hats with ghost notes, boom bap influenced hip hop percussion",
    ],
    "boom bap": [
        "deep sampled kick drum from vinyl on beats 1 and 3 with natural organic transient, hard rimshot snare on beats 2 and 4 from vinyl sample, loose swinging hi-hats with ghost notes not perfectly quantized, dusty boom bap drum break, vinyl crackle texture",
        "heavy sampled drum break from vinyl record with natural imperfection and swing, hard punchy kick and cracking rimshot snare, swinging hi-hat choke pattern with ghost notes, classic golden era boom bap groove naturally loose feel",
        "punchy compressed sampled kick with vinyl snap on 1 and 3, cracking rimshot snare from vinyl sample on 2 and 4, loose triplet hi-hat swing not quantized, dusty boom bap drum pocket, sampled bass not 808",
        "deep sampled drum break with loose natural swing feel, hard knocking kick and rimshot snare from vinyl, swinging hi-hats with fills and chokes, golden era boom bap groove warm analog compression vinyl crackle",
    ],
    "phonk": [
        "heavy overdriven distorted 808 kick with crunchy tape saturation, rapid rolling hi-hat triplets with cowbell accents hitting on every upbeat — cowbell is essential, hard snare slam, Memphis phonk percussion with cassette texture",
        "crunchy saturated distorted 808 bass drum, rolling hi-hats with prominent cowbell on offbeats — cowbell defines phonk, punchy snare hit, raw Memphis phonk percussion with vintage cassette degradation",
        "heavily distorted overdriven 808 kick with tape crunch and saturation, rolling hi-hat triplets and cowbell on every upbeat mandatory, slamming snare, dark Memphis phonk percussion arrangement with crunchy distorted bass",
        "hard overdriven 808 kick extreme distortion and saturation, fast rolling hi-hats with cowbell prominent on offbeats every time — cowbell essential, hard snare crack, dark Memphis rap percussion cassette tape texture",
    ],
    "house": [
        "four-on-the-floor kick hitting EVERY single beat without exception, open hi-hat on every 8th-note offbeat between kicks, crisp clap on beats 2 and 4, driving house drum pattern 126 BPM",
        "stomping kick drum on every quarter note without fail, open hi-hat on offbeats between every kick, punchy clap on backbeats 2 and 4, classic house music percussion arrangement",
        "four-on-the-floor kick every beat always, syncopated open hi-hat accents on offbeats, crisp snare or clap on 2 and 4, energetic house drum pattern 128 BPM",
        "relentless four-on-the-floor kick on every beat, open hi-hat on every off-beat sixteenth between kicks, punchy clap on backbeats, driving peak-hour house percussion 130 BPM",
    ],
    "deep house": [
        "soft four-on-the-floor kick on every beat with warm rounded attack, subtle open hi-hat on offbeats, soft clap on beats 2 and 4, hypnotic deep house groove 122 BPM",
        "rolling kick drum with warm soft attack on every beat, gentle swinging hi-hat groove on offbeats, subtle clap on backbeats, hypnotic deep house drum pattern 120 BPM",
        "warm four-on-the-floor kick every beat soft attack, gentle open hi-hat offbeats, soft clap 2 and 4, late-night deep house drum arrangement 122 BPM",
        "soft kick every quarter note rounded attack, subtle hi-hat offbeat groove, warm clap on backbeats, smooth deep house percussion 120 BPM",
    ],
    "tech house": [
        "driving punchy kick every single beat, percussive hi-hat groove with rim shot accents, industrial mechanical rhythm, relentless tech house percussion 130 BPM",
        "heavy four-on-the-floor kick every beat, aggressive syncopated hi-hat and rim percussion, relentless driving tech house drum arrangement 128 BPM",
        "hard mechanical kick on every beat, percussive syncopated hi-hat rhythm with rim shots, heavy industrial percussion, punishing tech house groove 132 BPM",
        "punchy distorted kick every beat four-on-the-floor, aggressive hi-hat and rim patterns, mechanical industrial percussion, relentless tech house drum power 130 BPM",
    ],
    "techno": [
        "HARD industrial distorted kick drum on EVERY single beat 140 BPM, aggressive 16th-note hi-hats relentless and mechanical throughout entire track, heavy metallic cold percussion, very minimal melody cold dystopian",
        "hard pounding distorted industrial kick every beat 145 BPM, rapid mechanical 16th-note hi-hats throughout unrelenting, industrial metallic percussion texture, brutal cold techno drum arrangement",
        "crushing hard distorted kick drum on every beat 142 BPM, aggressive relentless 16th-note hi-hat mechanical drive, cold metallic percussion, dark industrial techno rhythm",
        "massive industrial hard kick every beat 148 BPM, relentless 16th-note mechanical hi-hats continuous, heavy metallic percussion, brutal pounding techno drum machine pattern",
    ],
    "edm": [
        "four-on-the-floor kick at 128 BPM, sharp clap on beats 2 and 4, build-up snare roll into drop, driving festival EDM drum pattern explosive",
        "stomping kick every quarter note, punchy clap on backbeats, pre-drop snare fill building tension, powerful festival EDM percussion 128 BPM",
        "driving four-on-the-floor kick 128 BPM, crisp clap 2 and 4, snare roll crescendo into drop, massive stadium EDM drum pattern",
        "relentless kick on every beat 130 BPM, punchy clap backbeats, build snare roll before drop, explosive festival EDM percussion",
    ],
    "dubstep": [
        "HALF-TIME drum groove at 140 BPM — kick on beat 1 ONLY, MASSIVE cavernous reverb-soaked snare on beat 3 ONLY, very SPARSE hi-hats few and far between, half-time feel is the defining characteristic",
        "half-time kick beat 1 at 140 BPM, enormous bone-crushing reverb snare beat 3 only, minimal sparse hi-hats with long silence between, devastating dubstep half-time percussion",
        "half-time groove 140 BPM: kick hits beat 1 only, huge reverb-drenched snare crashes beat 3 only, sparse syncopated hi-hats, half-time slow feel with massive reverb",
        "140 BPM half-time pattern: kick on 1 only, massive cavernous reverb snare on 3 only, very sparse hi-hats, half-time groove is everything, enormous reverb tails on snare",
    ],
    "drum and bass": [
        "170 BPM syncopated amen break-style breakbeat, fast rolling syncopated kick and sharp snare not on 1 and 3, rapid 16th-note hi-hat drive, relentless DnB forward momentum",
        "frantic breakbeat at 170 BPM, syncopated kick on off-beats and sharp fast snare rolls in amen break pattern, rapid hi-hat drive, intense drum and bass percussion",
        "170 BPM amen-style syncopated breakbeat, rolling kick and snare in complex syncopated pattern, rapid 16th-note hi-hats, relentless drum and bass breakbeat groove",
        "fast 170 BPM syncopated amen break pattern, kick and snare in complex non-standard placement, rapid hi-hat rolls, frantic drum and bass percussion with rolling momentum",
    ],
    "dnb": [
        "170 BPM rolling amen-style syncopated breakbeat, syncopated kick and snare complex pattern, rapid hi-hats, relentless DnB drum groove",
        "fast amen break at 170 BPM, syncopated kick pattern complex placement, sharp snare hits off-beat, rapid hi-hat rolls, energetic DnB percussion",
        "170 BPM breakbeat syncopated kick and snare amen pattern, rapid 16th-note hi-hats, rolling momentum, intense DnB drum arrangement",
        "frantic 170 BPM amen break, syncopated kick off-beat and snare complex pattern, rapid hi-hat drive, relentless rolling DnB percussion",
    ],
    "jungle": [
        "frantic chopped amen break, complex syncopated kick and snare layering in breakbeat pattern, rapid hi-hat patterns, raw 90s jungle drum arrangement 160 BPM",
        "breakbeat complexity with chopped amen samples, fast syncopated percussion, raw frantic jungle rhythm, rapid kick and snare complex placement",
        "chopped amen break at 160 BPM, syncopated kick and snare complex pattern, rapid hi-hats, intense jungle percussion arrangement",
        "frantic jungle breakbeat, amen break chopping and layering, complex syncopated kick snare hi-hat, raw underground 90s drum pattern",
    ],
    "afrobeats": [
        "syncopated afrobeats kick pattern off the grid, talking drum accents, layered conga and shaker groove, off-beat hi-hat, vibrant West African percussion 105 BPM",
        "complex afrobeats polyrhythm, layered hand percussion and talking drum, syncopated kick pattern, tambourine and shaker, vibrant afro rhythm section",
        "vibrant afrobeats kick syncopation, talking drum and conga layers, shaker and tambourine offbeats, West African polyrhythm 105 BPM",
        "syncopated afro kick pattern, layered conga bongo and talking drum, off-beat hi-hat chops, complex afrobeats polyrhythm vibrant percussion",
    ],
    "reggaeton": [
        "dembow kick on beat 1 and the-and-of-beat-3, snare on beats 2 and 4, shuffled hi-hat groove, classic reggaeton dembow rhythm 95 BPM",
        "punchy dembow pattern, kick on 1 and and-3, hard snare on backbeats, shuffled hi-hat, aggressive Latin dembow percussion",
        "dembow kick beat 1 and and-beat-3, snare beats 2 and 4, shuffled hi-hat, driving reggaeton dembow rhythm section",
        "classic dembow: kick 1 and and-3, snare on 2 and 4, shuffled hi-hat groove, punchy reggaeton percussion 95 BPM",
    ],
    "r&b": [
        "smooth kick and snare pocket groove, shuffled hi-hat with ghost notes, polished soulful R&B drum arrangement 80 BPM",
        "laid-back kick on 1 and 3, snare on 2 and 4 with ghost note fills, swinging hi-hat groove, soulful R&B percussion warm pocket",
        "punchy kick groove on 1 and 3, crisp snare 2 and 4, shuffled hi-hat with ghost notes, deep soulful R&B drum pocket",
        "smooth R&B kick and snare pocket, swinging hi-hats with ghost notes and shuffle, warm soulful drum arrangement intimate groove",
    ],
    "lo-fi": [
        "dusty muffled kick drum with soft attack, soft brushed snare with vinyl grain not perfectly quantized, loose dusty hi-hats, laid-back lo-fi drum groove with natural human imperfection",
        "muted kick with soft attack, brushed snare decay with vinyl texture, dusty hi-hats relaxed, lo-fi drum pattern with loose human feel and tape warmth",
        "soft muffled kick not quantized, brushed snare with vinyl crackle, gentle loose hi-hats, cozy lo-fi drum groove cassette warmth",
        "gentle muffled kick soft attack, dusty brushed snare vinyl grain imperfect timing, loose hi-hat groove, relaxed lo-fi percussion warm analog feel",
    ],
    "funk": [
        "syncopated funk kick on the one, tight ghost-note snare hits, open hi-hat choke pattern, deep pocket groove, James Brown funk percussion 105 BPM",
        "driving funky kick drum with syncopation, crisp snare with ghost notes, open hi-hat rhythm, tight pocket funk arrangement",
        "syncopated kick on beat 1 and syncopations, ghost note snare popping, open hi-hat choke and release, deep funky pocket groove",
        "tight funky kick syncopation on the one, ghost note snare fills, open hi-hat chop pattern, deep groove James Brown influence",
    ],
    "reggae": [
        "one-drop kick on beat 3 ONLY nothing else, rim shot on beats 2 and 4, open hi-hat on offbeats, classic reggae one-drop drum pattern 80 BPM",
        "roots reggae one-drop kick beat 3 only, syncopated rim shot 2 and 4, offbeat hi-hat chop, deep roots reggae groove",
        "one-drop pattern: kick beat 3 only, rimshot 2 and 4, open hi-hat on offbeats, classic reggae rhythm",
        "reggae one-drop: kick on beat 3 only, rim shot backbeats, offbeat hi-hat, deep roots reggae percussion 80 BPM",
    ],
    "grime": [
        "staccato 140 BPM kick pattern, sharp metallic hi-hats, hard snare hit, cold rhythmic UK grime percussion aggressive",
        "aggressive 140 BPM drum arrangement, rapid hi-hat rhythm, punchy cold snare, dark grime beat percussion London sound",
        "cold staccato 140 BPM kick, sharp metallic hi-hat, hard snare crack, dark grime percussion arrangement relentless",
        "140 BPM grime kick pattern staccato, metallic hi-hats sharp, cold hard snare, dark urban London grime rhythm",
    ],
    "uk garage": [
        "2-step garage shuffle kick on beat 1 and and-beat-3, snare on 2 and 4 with shuffle, syncopated swinging hi-hat, UK garage 2-step drum pattern 132 BPM",
        "shuffled 2-step kick pattern beat 1 and and-3, swinging syncopated hi-hats, backbeat snare, bouncy UK garage percussion",
        "2-step shuffle: kick on 1 and and-3, snare 2 and 4, swinging hi-hat, classic UK garage shuffled rhythm 132 BPM",
        "UK garage 2-step shuffle kick, swinging hi-hat groove, snare on backbeats, bouncy syncopated garage percussion",
    ],
    "synthwave": [
        "driving electronic drum machine, four-on-the-floor kick, massive gated reverb snare on beats 2 and 4, 16th-note closed hi-hats, 80s drum machine pattern 120 BPM",
        "electronic drum machine pattern, punchy kick every beat, heavy gated reverb snare 2 and 4, 16th-note hi-hat drive, retro synthwave percussion",
        "four-on-the-floor kick every beat, massive gated reverb snare on 2 and 4, 16th-note hi-hats driving, classic 80s drum machine sound",
        "driving kick every beat, huge gated reverb snare on backbeats 2 and 4, closed 16th-note hi-hats, vintage 80s drum machine synthwave",
    ],
    "hyperpop": [
        "distorted glitchy drum pattern, heavy distorted 808 kick, glitched stutter snare effects, chaotic hi-hat fills, hyperpop percussion maximalist",
        "aggressive glitched-out drums, heavy distorted kick, stutter snare, chaotic energetic drum arrangement hyper-saturated",
        "distorted 808 kick heavy, glitched snare stutter effects, chaotic hi-hat fills, maximalist hyperpop drum chaos",
        "heavy glitch kick distorted, stutter snare glitch effects, rapid chaotic hi-hat, maximalist distorted hyperpop percussion",
    ],
    "pluggnb": [
        "slow melodic trap drums, heavy 808 sub kick beat 1, soft snare on beat 3, gentle minimal hi-hat pattern, slow pluggnb percussion 130 BPM",
        "laid-back slow trap groove, deep 808 kick beat 1, minimal hi-hats sparse, slow relaxed pluggnb drum arrangement",
        "slow trap drums heavy 808 kick, soft snare beat 3, sparse hi-hats gentle, slow emotional pluggnb percussion",
        "heavy slow 808 kick beat 1, soft snare beat 3, minimal gentle hi-hats, relaxed slow pluggnb drum groove",
    ],
    "jersey club": [
        "frantic 4x4 kick at 130+ BPM, rapid hi-hat chopping, syncopated snare hits, chaotic jersey club percussion high-energy",
        "fast four-on-the-floor kick, chopped percussion pattern, rapid hi-hat groove, aggressive jersey club drum arrangement",
        "frantic kick every beat 130+ BPM, rapid chopped hi-hats, syncopated snare hits, energetic jersey club percussion",
        "driving kick 130 BPM, rapid hi-hat chops syncopated, snare hits off-beat, chaotic high-energy jersey club drums",
    ],
    "cloud rap": [
        "slow minimal trap drums, sparse reverb-soaked kick and distant reverb snare, soft hi-hats, dreamy cloud rap percussion hazy",
        "minimal slow trap pattern, muffled kick with reverb, distant reverb snare, airy cloud rap drum groove soft",
        "slow muffled reverb kick, distant reverb snare soft, sparse gentle hi-hats, minimal hazy cloud rap drum pattern",
        "reverb-drenched slow kick, distant soft reverb snare, minimal sparse hi-hats, dreamy hazy cloud rap percussion",
    ],
    "jazz": [
        "swinging jazz ride cymbal pattern, brushed snare with ghost notes, kick on downbeats, hi-hat on 2 and 4, bebop swing drum groove",
        "live jazz drum kit with swing feel, ride cymbal rhythm, brushed snare ghost notes, loose swing hi-hat choke, jazz percussion",
        "swinging ride cymbal, brushed snare with ghost notes and fills, kick on downbeats hi-hat 2 and 4, natural jazz drum feel",
        "bebop jazz swing: ride cymbal pattern, ghost note snare brush, kick downbeats hi-hat on 2 and 4, natural swing feel",
    ],
    "soul": [
        "tight soul groove kick on 1 and 3, snare on 2 and 4 with pocket and snap, shuffled hi-hat with groove, gospel-influenced drum pocket",
        "soulful drum pocket, punchy kick and snare with swing, layered percussion, warm soul music drum arrangement 80 BPM",
        "tight kick 1 and 3, snare 2 and 4 snappy, shuffled hi-hat groove, deep soulful drum pocket warm",
        "punchy kick and snare pocket groove, shuffled hi-hat with ghost notes, warm soulful drum arrangement",
    ],
    "pop": [
        "punchy kick on 1 and 3, sharp snare on 2 and 4, driving 16th-note hi-hats, modern pop drum arrangement 120 BPM",
        "four-on-the-floor or 1-and-3 kick, crisp clap on backbeats, energetic pop percussion driving",
        "punchy kick groove, sharp snare backbeats, 16th-note hi-hat drive, polished modern pop drum pattern",
        "driving kick on 1 and 3, crisp snare 2 and 4, 16th-note hi-hats, energetic pop percussion arrangement",
    ],
    "cinematic": [
        "orchestral percussion with powerful timpani hits, taiko drums, dramatic snare rolls, cinematic tension percussion full orchestra",
        "epic timpani and taiko drum hits, dramatic orchestral percussion build, powerful cinematic rhythm section sweeping",
        "timpani and taiko orchestral percussion, dramatic snare rolls building tension, powerful cinematic drum arrangement",
        "powerful timpani hits, taiko drum accents, dramatic snare rolls, orchestral cinematic percussion building tension",
    ],
    "rock": [
        "powerful live drum kit, hard kick on beats 1 and 3, crashing snare on 2 and 4, driving 8th-note hi-hats, rock drum groove 130 BPM",
        "driving rock drums, punchy kick, cracking snare on 2 and 4, aggressive hi-hat pattern, live drum sound energetic",
        "hard kick 1 and 3, crashing snare 2 and 4, driving 8th-note hi-hats, powerful live rock drum kit",
        "powerful kick and snare rock groove, driving hi-hat 8th notes, live drum kit feel, hard-hitting rock percussion",
    ],
    "metal": [
        "thunderous double-kick drum blast beats, crashing snare on 2 and 4, rapid 16th-note hi-hats, metal drum arrangement crushing",
        "blastbeat double-kick pattern, alternating kick and snare rapid, 16th-note hi-hats, crushing metal drum groove",
        "double bass kick thunderous, crashing snare 2 and 4, rapid 16th-note hi-hats, massive metal drum arrangement",
        "double kick blast beats, snare crashes 2 and 4, rapid 16th-note hi-hats, overwhelming metal percussion",
    ],
    "dancehall": [
        "digital riddim kick on beat 1 and and-3, snare on backbeats, offbeat hi-hat chop, Caribbean dancehall percussion",
        "punchy dancehall kick and snare rhythm, offbeat hi-hat, energetic Caribbean drum arrangement digital riddim",
        "riddim kick 1 and and-3, snare backbeats 2 and 4, offbeat hi-hat chop, Caribbean dancehall drum pattern",
        "digital dancehall kick and snare, offbeat hi-hat chops, punchy Caribbean rhythm section",
    ],
    "amapiano": [
        "deep log drum bass kick on every beat, layered percussive piano chords, shuffled hi-hat, syncopated percussion, amapiano drum groove 112 BPM",
        "punchy log drum kick pattern, rolling hi-hat groove, syncopated percussion layers, South African amapiano rhythm",
        "log drum kick every beat, shuffled hi-hat groove, layered syncopated percussion, vibrant South African amapiano 112 BPM",
        "deep log drum kick on every beat, shuffled hi-hat syncopated, layered conga percussion, South African amapiano groove",
    ],
    "afro house": [
        "driving four-on-the-floor kick every beat, layered tribal conga and shaker percussion, open hi-hat offbeats, afro house drum pattern 124 BPM",
        "stomping kick every beat, complex afro percussion layers, tribal drum groove, afro house rhythm section",
        "four-on-the-floor kick every beat, tribal conga shaker layers, open hi-hat offbeats, spiritual afro house percussion 124 BPM",
        "driving kick every beat, layered tribal percussion congas and shakers, open hi-hat on offbeats, afro house groove 124 BPM",
    ],
    "future bass": [
        "driving electronic kick, powerful clap on backbeats, energetic hi-hat drive, festival future bass drum pattern 150 BPM",
        "punchy electronic drums with driving kick, emotive clap on backbeats, aggressive hi-hat, future bass percussion",
        "driving electronic kick every beat, powerful clap 2 and 4, energetic hi-hat drive, future bass drum arrangement 150 BPM",
        "punchy kick electronic, clap on backbeats, driving hi-hat pattern, energetic future bass drum groove",
    ],
    "trap soul": [
        "slow trap drums, heavy 808 sub kick beat 1, minimal gentle hi-hats, soft snare on beat 3, slow intimate trap soul percussion 70 BPM",
        "slow heavy 808 kicks, gentle rolling hi-hats, minimal soft snare, slow intimate trap soul drum groove",
        "slow 808 kick beat 1, soft snare beat 3, minimal sparse hi-hats, gentle slow trap soul drum pattern",
        "heavy slow 808 kick, soft snare beat 3 minimal, sparse hi-hats gentle, slow emotional trap soul percussion",
    ],
    "latin": [
        "layered conga and bongo patterns, timbales, shaker and claves, complex Latin polyrhythmic percussion arrangement",
        "vibrant Latin percussion section: congas, timbales, shakers, bass drum, energetic layered Latin rhythm",
        "conga bongo timbale and claves layered, shaker groove, complex polyrhythm, vibrant Latin percussion",
        "complex Latin percussion: congas, timbales, claves, shakers, bass drum, energetic polyrhythmic arrangement",
    ],
    "blues": [
        "shuffle rhythm kick and snare, driving 12-bar groove, swinging hi-hats, blues drum arrangement",
        "steady kick and snare groove, swinging hi-hats with shuffle feel, blues percussion 80 BPM",
        "blues shuffle kick and snare, swinging hi-hat groove, driving 12-bar rhythm section",
        "shuffled kick and snare 12-bar groove, swinging hi-hats, deep blues percussion feel",
    ],
    "gospel": [
        "driving kick and snare groove, powerful gospel percussion, uplifting spiritual rhythm section",
        "punchy gospel kick and snare, driving rhythm, powerful spiritual drum arrangement uplifting",
        "driving kick on 1 and 3, snare 2 and 4, powerful gospel percussion, uplifting driving rhythm",
        "gospel drum groove driving kick and snare, powerful rhythm section, spiritual uplifting percussion",
    ],
    "vaporwave": [
        "slow hazy drum machine pattern, soft muffled kick, gentle snare, minimal vaporwave percussion 80 BPM",
        "slow drum machine beat, soft kick and snare, minimal dreamy vaporwave drum texture",
        "slow soft drum machine, muffled gentle kick and snare, minimal atmospheric vaporwave percussion",
        "hazy slow drum machine pattern, soft kick and snare gentle, minimal nostalgic vaporwave drums",
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
    "dubstep":       ["massive sub-bass pressure, crushing Reese wobble, devastating half-time fury",
                      "face-melting LFO bass modulation, filthy growl bass, overwhelming drop energy",
                      "Reese bass destruction, crushing half-time kick, bone-crushing sub-bass weight",
                      "filthy wobble bass LFO, relentless half-time half-speed groove, devastating drop"],
    "drum and bass": ["170 BPM relentless breakbeat energy, frantic rolling amen breaks, sub-bass driving speed",
                      "amen break intensity, fast frantic momentum, deep Reese sub pressure relentless",
                      "rapid syncopated breakbeat drive, powerful Reese sub bass, relentless 170 BPM forward energy",
                      "frantic 170 BPM amen break power, rolling syncopated momentum, crushing sub bass drive"],
    "dnb":           ["170 BPM relentless amen breakbeat energy, frantic rolling breaks, Reese sub-bass speed",
                      "amen break intensity, fast driving 170 BPM momentum, deep sub pressure",
                      "rapid syncopated breakbeat drive, powerful sub bass Reese, relentless forward energy 170 BPM",
                      "frantic 170 BPM momentum, rolling amen break power, crushing sub bass drive"],
    "jungle":        ["frantic chopped amen breakbeat, reggae bass wobble, raw rhythmic intensity 160 BPM",
                      "complex breakbeat patterns, deep bass pressure, underground raw jungle power",
                      "rapid syncopated amen chops, reggae sub bass, frantic rhythmic drive",
                      "chopped amen frenzy, heavy bass wobble, raw underground jungle energy"],
    "techno":        ["relentless industrial driving force, crushing hard kick pressure on every beat, cold hypnotic dark intensity",
                      "mechanical industrial energy, peak-hour pounding power, cold relentless dystopian groove",
                      "dystopian hard kick drive on every beat, pounding mechanical rhythm, dark hypnotic forward momentum",
                      "cold industrial crushing force, relentless 16th-note hi-hat energy, hard kick pounding every beat"],
    "tech house":    ["hypnotic bass-driven intensity, relentless mechanical groove pressure, peak-hour underground energy",
                      "driving mechanical force, thick bass stab, pounding underground groove relentless",
                      "hard mechanical kick pressure, hypnotic bass stab drive, relentless tech house power",
                      "industrial bass groove, crushing kick every beat, dark relentless underground intensity"],
    "edm":           ["euphoric crowd energy, massive drop impact, stadium-filling power overwhelming",
                      "epic build tension release, soaring festival energy, huge crowd-moving moment",
                      "explosive drop energy, massive synth chord stabs, overwhelming stadium crowd power",
                      "massive festival euphoria, epic build and drop, soaring synth leads crowd-lifting energy"],
    "house":         ["four-on-the-floor driving power, peak-hour dancefloor energy, full soulful layers",
                      "crowd-lifting groove force, stomping kick on every beat, soulful energetic arrangement",
                      "driving stomp every beat, peak-hour dancefloor lift, full soulful house energy",
                      "relentless four-on-the-floor groove, crowd energy, stomping kick soulful power"],
    "deep house":    ["deep pulsing bass groove, warm driving energy, immersive late-night dancefloor power",
                      "warm bass pulse driving, hypnotic deep groove, late-night immersive energy",
                      "deep warm late-night energy, hypnotic bass groove, immersive emotional power",
                      "warm late-night deep groove, pulsing bass energy, intimate dancefloor immersion"],
    "electronic":    ["full-spectrum energy, layered modular intensity, driving electronic force",
                      "complex high-energy production, powerful synth layers, relentless electronic drive",
                      "full electronic spectrum power, layered synth intensity, relentless driving energy",
                      "complex modular synth energy, full arrangement power, relentless electronic force"],
    "hyperpop":      ["chaotic maximum energy, distorted glitch intensity, maximalist overwhelming sound",
                      "explosive glitch energy, heavy distorted 808, hyper-saturated production chaos",
                      "maximum hyper-saturated chaos, distorted maximalist energy, glitch overload",
                      "chaotic distorted hyper energy, maximalist glitch overload, overwhelming saturation"],
    "trap":          ["relentless hi-hat triplet aggression, thunderous 808 sub pressure, dark menacing atmospheric energy",
                      "rapid-fire 32nd-note hi-hat intensity, thunderous 808 portamento kick weight, dark threatening groove",
                      "velocity-dynamic hi-hat triplet relentlessness, massive sub-frequency 808 pressure, aggressive dark force",
                      "thunderous 808 sub-bass pressure, rapid-fire hi-hat roll aggression, hard snare crack power, dark menace"],
    "drill":         ["cold menacing energy, dark aggressive intensity, heavy dramatic 808 slide weight",
                      "relentless cold hard drive, ominous bass presence, cold brutal sparse groove",
                      "cold threatening energy, deep 808 slide pressure, sparse icy minimal groove intensity",
                      "cold ominous aggressive force, dramatic 808 portamento weight, sparse cold brutal energy"],
    "hip hop":       ["head-nodding groove power, punchy drum impact, deep bass pocket momentum",
                      "soulful harmonic richness, knocking beat momentum, full warm groove",
                      "hard-knocking drum power, deep bass pocket, soulful warm energy",
                      "deep groove momentum, punchy kick impact, warm soulful bass energy"],
    "boom bap":      ["knocking drum power, punchy vinyl sample chop, deep natural groove momentum",
                      "heavy drum knock, dusty soul energy, full loose boom bap groove",
                      "vinyl drum knock intensity, dusty soulful chop energy, organic loose groove",
                      "hard-knocking boom bap drums, dusty soul sample power, deep organic groove"],
    "phonk":         ["dark distorted Memphis energy, heavy crunchy 808 aggression, cowbell-driven raw power",
                      "chopped soul intensity, overdriven distorted bass, dark phonk groove force cowbell",
                      "cassette-saturated distorted energy, Memphis raw power, cowbell rhythmic force",
                      "heavy overdriven 808 distortion, Memphis dark phonk aggression, cowbell-driven energy"],
    "grime":         ["staccato aggressive energy, cold UK grime intensity, hard-hitting dark power",
                      "relentless staccato grime groove, cold synth stabs, aggressive dark forward drive",
                      "cold mechanical staccato force, UK grime aggression, hard dark energy",
                      "staccato cold UK intensity, grime beat aggression, hard dark forward momentum"],
    "rock":          ["driving guitar power, explosive live drum energy, wall of sound intensity",
                      "full live band arrangement, powerful drums, distorted guitar layers energy",
                      "explosive live rock energy, powerful drum drive, distorted guitar wall of sound",
                      "driving guitar riff power, live drum intensity, full rock band energy"],
    "metal":         ["crushing heavy intensity, thunderous double-kick drum power, aggressive riff energy",
                      "wall of distorted guitars, blastbeat power, overwhelming heavy force",
                      "thunderous double-kick crushing power, heavy riff aggression, overwhelming metal force",
                      "blastbeat drum energy, crushing distorted guitar walls, overwhelming heavy metal force"],
    "afrobeats":     ["infectious afro groove, layered percussion energy, vibrant full arrangement",
                      "driving afrobeats polyrhythm, melodic hook energy, full percussive power",
                      "vibrant West African polyrhythm, layered percussion drive, infectious melodic energy",
                      "infectious afrobeats groove, talking drum energy, vibrant layered percussion power"],
    "reggaeton":     ["punchy dembow drive, thick bass presence, energetic Latin groove",
                      "heavy dembow rhythm, urban bass energy, full reggaeton power",
                      "driving dembow energy, thick urban bass, full Caribbean groove power",
                      "punchy dembow kick energy, urban bass weight, infectious reggaeton groove"],
    "latin":         ["vibrant layered percussion, brass and rhythm energy, full Latin arrangement",
                      "complex polyrhythm power, layered conga energy, vibrant brass Latin force",
                      "vibrant Latin percussion layers, brass stab energy, polyrhythmic groove power",
                      "energetic Latin polyrhythm, layered percussion drive, vibrant full arrangement"],
    "funk":          ["deep funky groove, slapped bass power, full band funk energy",
                      "tight syncopated rhythm, layered funk arrangement, groove-driven force",
                      "deep pocket funk groove, slapped bass intensity, full band syncopated energy",
                      "tight syncopated funk power, ghost note groove, slapped bass deep pocket"],
    "soul":          ["deep soulful groove, full band arrangement, warm powerful energy",
                      "layered soul production, gospel energy, rich harmonic power",
                      "warm deep soulful groove, gospel-influenced energy, rich harmonic arrangement",
                      "deep soul pocket groove, warm powerful energy, rich harmonic full arrangement"],
    "r&b":           ["smooth powerful groove, layered R&B arrangement, warm bass energy",
                      "rich harmonic depth, full R&B production, punchy groove force",
                      "smooth R&B groove power, warm layered arrangement, deep bass pocket energy",
                      "rich harmonic R&B depth, warm groove energy, full soulful arrangement power"],
    "synthwave":     ["pulsing retro energy, driving analog synth power, neon-lit 80s momentum",
                      "relentless synthwave drive, layered analog synth stacks, 80s peak energy",
                      "driving 80s analog synth power, neon atmospheric energy, relentless momentum",
                      "retro analog synth intensity, pulsing bass momentum, neon cinematic power"],
    "pop":           ["hooky powerful energy, full polished arrangement, driving pop momentum",
                      "layered pop production, punchy polished mix, catchy energetic arrangement",
                      "driving hook energy, full polished pop arrangement, catchy momentum",
                      "catchy pop groove power, polished full arrangement, driving energetic hooks"],
    "cinematic":     ["sweeping dramatic power, full orchestral energy, cinematic climax force",
                      "building tension and release, massive cinematic drop, overwhelming orchestral energy",
                      "epic orchestral sweep, dramatic tension release, massive cinematic power",
                      "full orchestral dramatic force, sweeping cinematic energy, overwhelming tension climax"],
    "lo-fi":         ["warm cozy groove, gentle analog warmth, relaxed intimate energy",
                      "soft vinyl crackle atmosphere, mellow laid-back groove, warm nostalgic feel",
                      "gentle warm analog energy, dusty chill groove, relaxed intimate atmosphere",
                      "cozy lo-fi warmth, soft groove energy, mellow analog atmosphere"],
    "amapiano":      ["deep log drum groove, vibrant South African energy, layered piano percussion drive",
                      "infectious amapiano log drum energy, vibrant layered groove, South African power",
                      "deep log drum pulse, piano stab energy, vibrant South African dance groove",
                      "log drum driving groove, layered percussion energy, vibrant amapiano power"],
    "cloud rap":     ["hazy ethereal atmosphere, dreamy reverb-drenched energy, soft ambient trap feel",
                      "misty reverb-soaked groove, soft dreamy energy, hazy atmospheric trap",
                      "ethereal hazy atmosphere, reverb-drenched soft energy, dreamy trap feel",
                      "soft misty trap energy, reverb-soaked dreamy atmosphere, hazy emotional feel"],
    "phonk":         ["dark distorted Memphis energy, heavy crunchy 808 aggression, cowbell-driven raw power",
                      "overdriven distorted bass, Memphis raw phonk power, cowbell rhythm drive",
                      "cassette-saturated distorted energy, Memphis phonk aggression, cowbell force",
                      "heavy overdriven 808 distortion, dark Memphis phonk power, cowbell-driven"],
    "trap soul":     ["slow warm emotional energy, heavy 808 sub pressure, intimate RnB atmosphere",
                      "deep slow 808 groove, warm emotional intimacy, gentle trap soul energy",
                      "intimate slow emotional power, heavy 808 warmth, gentle trap soul groove",
                      "slow heavy 808 emotional energy, warm RnB intimacy, gentle dark atmosphere"],
    "future bass":   ["euphoric festival energy, massive chord swell power, emotional atmospheric drive",
                      "soaring synth euphoria, powerful drop energy, emotional festival atmosphere",
                      "massive chord swell euphoria, driving electronic energy, festival emotional power",
                      "euphoric soaring synth energy, massive drop impact, emotional festival atmosphere"],
    "afro house":    ["spiritual dancefloor power, tribal percussion drive, four-on-the-floor groove energy",
                      "tribal afro groove intensity, spiritual dance energy, driving four-on-the-floor power",
                      "layered tribal percussion energy, spiritual dancefloor drive, afro groove power",
                      "driving spiritual tribal energy, layered afro percussion, peak dancefloor groove"],
    "hyperpop":      ["chaotic maximum distorted energy, glitch overload, hyper-saturated overwhelming force",
                      "explosive hyper-saturated chaos, maximalist glitch energy, distorted overload",
                      "maximum chaos distorted energy, glitch maximalism, overwhelming hyper production",
                      "hyper-saturated glitch energy, maximalist distorted chaos, overwhelming hyper force"],
    "pluggnb":       ["slow emotional melodic energy, heavy 808 weight, intimate dark atmosphere",
                      "deep slow 808 emotion, gentle melodic intimacy, dark atmospheric energy",
                      "slow dark emotional power, heavy 808 groove, intimate pluggnb atmosphere",
                      "gentle slow emotional energy, heavy 808 sub pressure, dark intimate feel"],
    "grime":         ["cold staccato UK energy, hard-hitting dark grime force, aggressive cold power",
                      "staccato cold grime intensity, dark UK aggression, hard forward drive",
                      "aggressive cold UK energy, staccato dark grime power, relentless forward force",
                      "cold hard grime aggression, staccato dark UK intensity, relentless drive"],
    "vaporwave":     ["hazy nostalgic atmosphere, slow dreamlike energy, warm surreal feel",
                      "dreamlike slow nostalgia, hazy warm atmosphere, surreal gentle energy",
                      "nostalgic warm haze, slow dreamlike atmosphere, surreal vintage feel",
                      "slow hazy nostalgic energy, dreamlike warm atmosphere, surreal vintage power"],
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
    "jungle":        (9.5, 10.0),
    "techno":        (9.5, 10.0),
    "tech house":    (9.5, 10.0),
    "edm":           (9.5, 10.0),
    "house":         (9.5, 10.0),
    "deep house":    (9.0,  9.5),
    "electronic":    (9.5, 10.0),
    "hyperpop":      (9.5, 10.0),
    "jersey club":   (9.5, 10.0),
    "uk garage":     (9.0,  9.5),
    "grime":         (9.5, 10.0),
    # ── Synthwave / retro ──────────────────────────────────────────────────
    "synthwave":     (9.0,  9.5),
    "vaporwave":     (8.0,  9.0),
    # ── Hip-hop / rap ──────────────────────────────────────────────────────
    "trap":          (9.5, 10.0),
    "drill":         (9.5, 10.0),
    "hip hop":       (9.0, 10.0),
    "hip-hop":       (9.0, 10.0),
    "rap":           (9.0, 10.0),
    "boom bap":      (9.0, 10.0),
    "phonk":         (9.5, 10.0),
    "cloud rap":     (8.0,  9.0),
    "pluggnb":       (8.0,  9.0),
    # ── R&B / Soul / Funk ──────────────────────────────────────────────────
    "r&b":           (8.5,  9.5),
    "soul":          (8.5,  9.5),
    "funk":          (9.0,  9.5),
    "gospel":        (9.0,  9.5),
    # ── Afro / Caribbean / Latin ───────────────────────────────────────────
    "afrobeats":     (9.0,  9.5),
    "dancehall":     (9.0,  9.5),
    "reggaeton":     (9.0,  9.5),
    "reggae":        (8.5,  9.0),
    "latin":         (9.0,  9.5),
    # ── Live instruments ───────────────────────────────────────────────────
    "rock":          (9.5, 10.0),
    "metal":         (9.5, 10.0),
    "blues":         (8.5,  9.5),
    "jazz":          (8.0,  9.0),
    # ── Atmospheric / chill ────────────────────────────────────────────────
    "lo-fi":         (7.0,  8.0),
    "ambient":       (6.5,  7.5),
    "cinematic":     (8.5,  9.5),
    # ── Pop ────────────────────────────────────────────────────────────────
    "pop":           (8.5,  9.5),
}

# Tags injected when user specifies "energic" / "energetic" in their prompt (neutral/happy mood)
_ENERGIC_BOOST_TAGS = [
    "maximum energy, relentless intensity, explosive powerful production",
    "full power, aggressive high-energy, overwhelming sonic force",
    "peak energy, driving relentless momentum, intense high-impact sound",
    "maximum intensity, hard-hitting explosive production, full-throttle energy",
]

# Tags injected when energic + dark/sad mood — hard/fast/intense but NOT happy, NOT uplifting
_DARK_ENERGIC_BOOST_TAGS = [
    "dark maximum energy, powerful but emotionally heavy, NOT uplifting, NOT happy, NOT bright, minor key, intense sorrowful force",
    "aggressive dark power, hard-hitting and emotionally devastating, NOT major key, NOT cheerful, cold relentless intensity",
    "heavy dark intensity, explosive emotional pain channeled into sonic force, NOT uplifting, minor key, menacing drive",
    "dark full-throttle production, relentless heavy power, NOT happy, NOT bright, emotionally raw intensity, minor key",
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

    # ── Detect dark/sad emotional tone ────────────────────────────────────────
    # Check both explicit mood field and detected prompt keywords
    all_mood_words = set([m] + detected_moods + p_lower.split())
    is_dark_sad = bool(all_mood_words & _DARK_SAD_MOODS)

    # ── Genre-specific negative disambiguation tags ───────────────────────────
    # Prevent ACE-Step from drifting to "default" pop/EDM production
    _GENRE_NEGATIVE_TAGS = {
        "trap":    "trap beat, hip hop street production, NOT pop, NOT EDM, NOT electronic dance, NOT soft",
        "drill":   "drill beat, dark cold drill, NOT pop, NOT trap, NOT EDM, NOT soft",
        "hip hop": "hip hop beat, rap production, NOT pop, NOT EDM, NOT R&B smooth",
        "hip-hop": "hip hop beat, rap production, NOT pop, NOT EDM, NOT R&B smooth",
        "boom bap": "boom bap hip hop, raw underground rap, NOT pop, NOT trap, NOT EDM",
        "phonk":   "phonk beat, Memphis phonk production, cowbell driven, NOT pop, NOT EDM",
        "rap":     "rap beat, hip hop production, NOT pop, NOT EDM",
        "techno":  "techno beat, industrial mechanical dance, NOT pop, NOT trance, NOT melodic EDM",
        "tech house": "tech house groove, underground club beat, NOT pop, NOT trance",
        "dubstep": "dubstep bass music, heavy bass wobble drop, NOT pop, NOT melodic EDM",
        "rock":    "rock music, live guitar drums, NOT pop, NOT electronic",
        "metal":   "metal music, aggressive guitar, NOT pop, NOT electronic",
        "jazz":    "jazz music, live instruments swing, NOT pop, NOT electronic",
        "lo-fi":   "lo-fi hip hop, chill relaxed beat, NOT aggressive, NOT loud",
        "ambient": "ambient music, atmospheric texture, NOT drums, NOT beat-driven",
    }

    parts = []

    # ── Genre anchor (VERY FIRST tag — strongly anchors genre identity) ───────
    neg = _GENRE_NEGATIVE_TAGS.get(g)
    if neg:
        # When dark/sad mood detected, append explicit anti-happy negations to genre anchor
        if is_dark_sad:
            neg = neg + ", NOT uplifting, NOT major key, NOT happy, NOT bright, NOT cheerful, minor key, dark emotional tone"
        parts.insert(0, neg)
    elif g:
        base = f"{g} beat, {g} music production style"
        if is_dark_sad:
            base += ", NOT uplifting, NOT major key, NOT happy, NOT bright, minor key, dark emotional tone"
        parts.insert(0, base)

    # ── Explicit instrument override ──────────────────────────────────────────
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
        if is_dark_sad:
            # Dark + energic = hard/fast/intense but NOT happy — use dark-energy tags
            parts.append(random.choice(_DARK_ENERGIC_BOOST_TAGS))
            parts.append("massive wall of dark sound, relentless heavy-energy arrangement, powerful dark full-mix impact, minor key intensity")
        else:
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

    # ── Atmospheric feel tag (80% chance, energy-appropriate) ─────────────────
    if random.random() < 0.80:
        parts.append(random.choice(_feel_tags_for_genre(g, is_dark_sad=is_dark_sad, is_energic=is_energic)))

    # ── Mix fullness ──────────────────────────────────────────────────────────
    parts.append(random.choice(_FULLNESS_TAGS))

    # ── Base energy boost for all non-chill genres ─────────────────────────
    if g not in _CHILL_GENRES:
        if is_dark_sad:
            parts.append(random.choice([
                "hard-hitting dark production, every drum hit lands with force, powerful heavy mix, NOT uplifting, minor key",
                "aggressive full-power dark mix, hard kick impact, loud punchy drums, menacing dark arrangement, NOT happy",
                "powerful dark driving production, relentless heavy energy, strong punchy drum hits, full loud dark mix",
                "intense hard-hitting dark beat, powerful kick and snare punch, full-throttle driving rhythm, cold and heavy",
            ]))
        else:
            parts.append(random.choice([
                "hard-hitting punchy production, every drum hit lands with force, powerful energetic mix",
                "aggressive full-power mix, hard kick impact, loud punchy drums, high-energy arrangement",
                "powerful driving production, relentless energy, strong punchy drum hits, full loud mix",
                "energetic hard-hitting beat, powerful kick and snare punch, full-throttle driving rhythm",
            ]))

    # ── Production texture ────────────────────────────────────────────────────
    textures = _PRODUCTION_TEXTURES.get(g, _PRODUCTION_TEXTURES["_default"])
    parts.append(random.choice(textures))

    # ── Production technique (filter lo-fi techniques for energetic genres) ──
    _energetic_techniques = [t for t in _PRODUCTION_TECHNIQUES
                              if g not in _CHILL_GENRES or "lo-fi" not in t.lower()]
    parts.append(random.choice(_energetic_techniques if _energetic_techniques else _PRODUCTION_TECHNIQUES))

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
    if is_energic and is_dark_sad:
        parts.append("dark explosive production, powerful heavy dark mix, every element hitting hard in minor key, NOT uplifting, NOT happy, emotionally devastating high-intensity sound")
    elif is_energic:
        parts.append("explosive maximum-energy production, full loud powerful mix, every element hitting hard, peak intensity")
    elif is_dark_sad:
        parts.append("dark emotional studio production, minor key, NOT uplifting, NOT happy, NOT bright, deep somber mix, heavy atmosphere")
    elif g in _CHILL_GENRES:
        parts.append("full rich studio production, warm mix depth, smooth arrangement, emotionally expressive")
    else:
        parts.append("high-energy studio production, loud powerful mix, hard-hitting drums and bass, full aggressive arrangement")

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

    # Detect genre category for structure selection
    _HH_GENRES = {"trap", "drill", "hip hop", "hip-hop", "rap", "boom bap", "phonk",
                  "cloud rap", "pluggnb", "grime", "uk garage"}
    _is_hh = g in _HH_GENRES

    # ── Random structure pool by duration (non-electronic) ───────────────────
    # Hip-hop uses [hook] instead of [chorus] for authenticity
    if _is_hh:
        if duration <= 90:
            pool = [
                ["[intro]", "[verse]", "[hook]", "[outro]"],
                ["[verse]", "[hook]", "[verse]", "[outro]"],
                ["[hook]", "[verse]", "[hook]", "[outro]"],
                ["[verse]", "[verse]", "[hook]", "[outro]"],
                ["[intro]", "[hook]", "[verse]", "[outro]"],
            ]
        elif duration <= 130:
            pool = [
                ["[verse]", "[hook]", "[verse]", "[hook]", "[outro]"],
                ["[intro]", "[verse]", "[hook]", "[bridge]", "[hook]"],
                ["[hook]", "[verse]", "[hook]", "[verse]", "[outro]"],
                ["[intro]", "[verse]", "[verse]", "[hook]", "[outro]"],
                ["[verse]", "[hook]", "[bridge]", "[verse]", "[hook]"],
                ["[intro]", "[hook]", "[verse]", "[hook]", "[outro]"],
                ["[verse]", "[verse]", "[hook]", "[bridge]", "[outro]"],
            ]
        elif duration <= 180:
            pool = [
                ["[intro]", "[verse]", "[hook]", "[verse]", "[hook]", "[bridge]", "[outro]"],
                ["[verse]", "[hook]", "[verse]", "[hook]", "[verse]", "[outro]"],
                ["[intro]", "[hook]", "[verse]", "[hook]", "[verse]", "[hook]", "[outro]"],
                ["[intro]", "[verse]", "[verse]", "[hook]", "[bridge]", "[hook]", "[outro]"],
                ["[verse]", "[verse]", "[hook]", "[verse]", "[hook]", "[outro]"],
                ["[intro]", "[verse]", "[hook]", "[bridge]", "[verse]", "[hook]", "[outro]"],
                ["[hook]", "[verse]", "[verse]", "[hook]", "[bridge]", "[hook]", "[outro]"],
            ]
        else:
            pool = [
                ["[intro]", "[verse]", "[hook]", "[verse]", "[hook]", "[bridge]", "[verse]", "[hook]", "[outro]"],
                ["[verse]", "[hook]", "[verse]", "[hook]", "[bridge]", "[verse]", "[hook]", "[outro]"],
                ["[intro]", "[hook]", "[verse]", "[hook]", "[verse]", "[bridge]", "[hook]", "[outro]"],
                ["[intro]", "[verse]", "[verse]", "[hook]", "[bridge]", "[verse]", "[hook]", "[hook]", "[outro]"],
                ["[verse]", "[hook]", "[verse]", "[hook]", "[verse]", "[hook]", "[outro]"],
            ]
    else:
        if duration <= 90:
            pool = [
                ["[intro]", "[verse]", "[chorus]", "[outro]"],
                ["[verse]", "[chorus]", "[verse]", "[outro]"],
                ["[intro]", "[chorus]", "[verse]", "[chorus]"],
                ["[chorus]", "[verse]", "[chorus]", "[outro]"],
                ["[verse]", "[verse]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[bridge]", "[chorus]"],
            ]
        elif duration <= 130:
            pool = [
                ["[intro]", "[verse]", "[chorus]", "[verse]", "[outro]"],
                ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[outro]"],
                ["[intro]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
                ["[verse]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[verse]", "[chorus]", "[chorus]", "[outro]"],
                ["[chorus]", "[verse]", "[chorus]", "[verse]", "[outro]"],
                ["[intro]", "[verse]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
                ["[verse]", "[bridge]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
                ["[intro]", "[chorus]", "[verse]", "[bridge]", "[outro]"],
            ]
        elif duration <= 180:
            pool = [
                ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
                ["[intro]", "[chorus]", "[verse]", "[pre-chorus]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[chorus]", "[verse]", "[pre-chorus]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[verse]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
                ["[intro]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
                ["[chorus]", "[verse]", "[chorus]", "[verse]", "[bridge]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"],
                ["[verse]", "[chorus]", "[chorus]", "[verse]", "[bridge]", "[outro]"],
                ["[intro]", "[verse]", "[bridge]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
            ]
        else:
            pool = [
                ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[verse]", "[bridge]", "[outro]"],
                ["[intro]", "[chorus]", "[verse]", "[pre-chorus]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"],
                ["[intro]", "[verse]", "[verse]", "[chorus]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[pre-chorus]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"],
                ["[chorus]", "[verse]", "[chorus]", "[verse]", "[bridge]", "[chorus]", "[verse]", "[chorus]", "[outro]"],
                ["[intro]", "[verse]", "[chorus]", "[chorus]", "[bridge]", "[verse]", "[chorus]", "[outro]"],
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


def _job_value(job_input: dict, *keys, default=None):
    for key in keys:
        if key in job_input and job_input[key] is not None:
            return job_input[key]
    return default


def _to_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _native_caption(job_input: dict, bpm: int, key: str) -> str:
    explicit = (_job_value(job_input, "caption", "tags", "prompt") or "").strip()
    if explicit:
        return explicit

    parts = []
    for value in (job_input.get("genre"), job_input.get("mood"), job_input.get("style")):
        value = (value or "").strip()
        if value:
            parts.append(value)
    if bpm:
        parts.append(f"{bpm} BPM")
    if key:
        parts.append(f"key of {key}")
    parts.append("instrumental")
    return ", ".join(parts)


def _native_lyrics(job_input: dict) -> str:
    return "[inst]"


def _native_guidance(job_input: dict, genre: str, prompt: str = "") -> float:
    raw = _job_value(job_input, "guidanceScale", "guidance_scale")
    if raw is not None:
        return float(raw)
    return _get_guidance_scale(genre, prompt)


def _sample_value(sample, *keys, default=None):
    for key in keys:
        if hasattr(sample, key):
            value = getattr(sample, key)
            if value not in (None, ""):
                return value
        if isinstance(sample, dict):
            value = sample.get(key)
            if value not in (None, ""):
                return value
    return default


def _looks_instrumental(job_input: dict, lyrics: str, prompt: str) -> bool:
    if _to_bool(_job_value(job_input, "instrumental", "isInstrumental", "is_instrumental"), default=False):
        return True
    text = f"{lyrics or ''}\n{prompt or ''}".lower()
    return any(tag in text for tag in ("[inst]", "[instrumental]", "instrumental", "no vocals"))


def _enhance_native_params(llm, job_input: dict, caption: str, lyrics: str,
                           prompt: str, genre: str, mood: str, bpm, key: str,
                           duration: float):
    """Use ACE-Step's own LLM helpers to match the hosted simple-mode prompt flow."""
    if llm is None or _ace_create_sample is None or _ace_format_sample is None:
        return caption, lyrics, bpm, key, int(duration), None

    try:
        use_format = bool((lyrics or "").strip() and (lyrics or "").strip() != "[inst]")
        user_metadata = {
            "bpm": bpm or None,
            "keyscale": key or None,
            "duration": int(duration) if duration else None,
            "genre": genre or None,
            "mood": mood or None,
            "instrumental": _looks_instrumental(job_input, lyrics, prompt or caption),
        }
        user_metadata = {k: v for k, v in user_metadata.items() if v not in (None, "")}

        if use_format:
            sample = _ace_format_sample(
                llm_handler=llm,
                caption=caption,
                lyrics=lyrics,
                user_metadata=user_metadata or None,
                use_constrained_decoding=True,
            )
        else:
            query = prompt or caption
            sample = _ace_create_sample(
                llm_handler=llm,
                query=query,
                instrumental=_looks_instrumental(job_input, lyrics, query),
                use_constrained_decoding=True,
            )

        success = _sample_value(sample, "success", default=True)
        if success is False:
            err = _sample_value(sample, "error", default="unknown error")
            print(f"[acestep] Native sample enhancement failed: {err}", flush=True)
            return caption, lyrics, bpm, key, int(duration), None

        out_caption = _sample_value(sample, "caption", default=caption) or caption
        out_lyrics = _sample_value(sample, "lyrics", default=lyrics or "[inst]") or "[inst]"
        out_bpm = _sample_value(sample, "bpm", default=bpm)
        out_key = _sample_value(sample, "keyscale", "key_scale", default=key)
        out_duration = _sample_value(sample, "duration", "audio_duration", default=int(duration))
        out_language = _sample_value(sample, "language", "vocal_language")

        print(f"[acestep] Native prompt enhanced via {'format_sample' if use_format else 'create_sample'}", flush=True)
        print(f"[acestep] Enhanced caption: {out_caption}", flush=True)
        print(f"[acestep] Enhanced lyrics: {out_lyrics}", flush=True)
        print(f"[acestep] Enhanced metas: bpm={out_bpm} key={out_key} duration={out_duration} language={out_language}", flush=True)
        return out_caption, out_lyrics, out_bpm, out_key, out_duration, out_language
    except Exception as e:
        print(f"[acestep] Native prompt enhancement skipped: {e}", flush=True)
        return caption, lyrics, bpm, key, int(duration), None


def _apply_optional_param(params: GenerationParams, attr: str, value):
    if value is None or not hasattr(params, attr):
        return
    try:
        setattr(params, attr, value)
    except Exception as e:
        print(f"[acestep] Optional param '{attr}' skipped: {e}", flush=True)


def _build_generation_params(job_input: dict, duration: float, bpm: int, key: str,
                             genre: str, mood: str, style: str, prompt: str,
                             infer_step: int, seed: int, llm=None) -> tuple[GenerationParams, str, str, str, float]:
    prompt_mode = str(_job_value(
        job_input,
        "promptMode",
        "prompt_mode",
        default=os.environ.get("ACESTEP_PROMPT_MODE", "native"),
    )).strip().lower()
    native_mode = prompt_mode in {"native", "ace", "ace-step", "acestep"}

    if native_mode:
        caption = _native_caption(job_input, bpm, key)
        lyrics = _native_lyrics(job_input)
        guidance_scale = _native_guidance(job_input, genre, prompt)
        caption, lyrics, bpm, key, duration, vocal_language = _enhance_native_params(
            llm=llm,
            job_input=job_input,
            caption=caption,
            lyrics=lyrics,
            prompt=prompt,
            genre=genre,
            mood=mood,
            bpm=bpm,
            key=key,
            duration=duration,
        )
    else:
        caption = build_tags(prompt or style, genre, mood, bpm, key)
        lyrics = _build_lyrics_structure(duration, prompt, genre)
        guidance_scale = _get_guidance_scale(genre, prompt)
        vocal_language = None

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        duration=int(duration),
        bpm=bpm,
        keyscale=key if key else "N/A",
        timesignature="4/4",
        inference_steps=infer_step,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    if vocal_language and hasattr(params, "vocal_language"):
        params.vocal_language = vocal_language
    if native_mode and hasattr(params, "thinking"):
        params.thinking = True
    if native_mode and hasattr(params, "use_cot_metas"):
        params.use_cot_metas = False
    if native_mode and hasattr(params, "use_cot_caption"):
        params.use_cot_caption = False
    if native_mode and hasattr(params, "use_cot_language"):
        params.use_cot_language = False
    if native_mode and hasattr(params, "use_constrained_decoding"):
        params.use_constrained_decoding = True

    if native_mode:
        optional_fields = {
            "scheduler_type": _job_value(job_input, "schedulerType", "scheduler_type"),
            "cfg_type": _job_value(job_input, "cfgType", "cfg_type"),
            "omega_scale": _job_value(job_input, "omegaScale", "omega_scale"),
            "guidance_interval": _job_value(job_input, "guidanceInterval", "guidance_interval"),
            "guidance_interval_decay": _job_value(job_input, "guidanceIntervalDecay", "guidance_interval_decay"),
            "min_guidance_scale": _job_value(job_input, "minGuidanceScale", "min_guidance_scale"),
            "use_erg_tag": _job_value(job_input, "useErgTag", "use_erg_tag"),
            "use_erg_lyric": _job_value(job_input, "useErgLyric", "use_erg_lyric"),
            "use_erg_diffusion": _job_value(job_input, "useErgDiffusion", "use_erg_diffusion"),
            "oss_steps": _job_value(job_input, "ossSteps", "oss_steps"),
            "guidance_scale_text": _job_value(job_input, "guidanceScaleText", "guidance_scale_text"),
            "guidance_scale_lyric": _job_value(job_input, "guidanceScaleLyric", "guidance_scale_lyric"),
        }
        for attr, value in optional_fields.items():
            if attr.startswith("use_erg_"):
                value = _to_bool(value) if value is not None else None
            _apply_optional_param(params, attr, value)

    return params, prompt_mode, caption, lyrics, guidance_scale


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
    bpm        = int(bpm_raw) if bpm_raw not in (None, "", 0, "0") else None
    key        = (job_input.get("key") or "").strip()
    mood       = job_input.get("mood",  "").strip().lower()
    style      = job_input.get("style", "").strip()
    beat_id    = job_input.get("beatId", "")
    infer_step = int(job_input.get("inferSteps") or 40)

    ref_audio_b64  = job_input.get("referenceAudio")
    ref_strength   = float(job_input.get("referenceStrength") or 0.5)

    seed = int(_job_value(job_input, "seed", default=random.randint(0, 2**31 - 1)))
    full_params, prompt_mode, tags, lyrics_str, g_scale = _build_generation_params(
        job_input=job_input,
        duration=duration,
        bpm=bpm,
        key=key,
        genre=genre,
        mood=mood,
        style=style,
        prompt=user_prompt,
        infer_step=infer_step,
        seed=seed,
        llm=llm,
    )

    print(f"[gen] Prompt mode: {prompt_mode}", flush=True)
    print(f"[gen] Caption: {tags}", flush=True)
    print(f"[gen] Lyrics: {lyrics_str}", flush=True)
    print(f"[gen] Duration: {full_params.duration:.0f}s | BPM: {full_params.bpm} | Key: {full_params.keyscale} | Seed: {seed} | guidance={g_scale}", flush=True)

    # Handle reference audio → temp file
    ref_path = None
    if ref_audio_b64:
        ref_path = "/tmp/bh_reference.wav"
        with open(ref_path, "wb") as f:
            f.write(base64.b64decode(ref_audio_b64))
        print(f"[gen] Reference audio saved", flush=True)

    # ── 1. Generate the full beat (text2music) ────────────────────────────────
    print(f"[gen] Generating full beat ({full_params.duration:.0f}s, {full_params.bpm} BPM, seed={seed})...", flush=True)
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
        "bpm":              full_params.bpm,
        "key":              full_params.keyscale,
        "tags":             tags,
        "lyrics":           lyrics_str,
        "prompt_mode":      prompt_mode,
    }


def generate_preview_audio(job_input: dict) -> tuple:
    """Short 60s preview for MIDI mode beat page. Returns (wav_base64, duration_sec)."""
    dit, dit_base, llm = get_handlers()

    genre  = job_input.get("genre",  "").strip().lower()
    mood   = job_input.get("mood",   "").strip().lower()
    bpm_raw = job_input.get("bpm")
    bpm    = int(bpm_raw) if bpm_raw not in (None, "", 0, "0") else None
    key    = (job_input.get("key") or "").strip()
    prompt = (job_input.get("prompt") or job_input.get("style") or "").strip()

    seed = int(_job_value(job_input, "seed", default=random.randint(0, 2**31 - 1)))
    params, prompt_mode, tags, lyrics, guidance_scale = _build_generation_params(
        job_input=job_input,
        duration=60,
        bpm=bpm,
        key=key,
        genre=genre,
        mood=mood,
        style=(job_input.get("style") or "").strip(),
        prompt=prompt,
        infer_step=int(_job_value(job_input, "inferSteps", "infer_step", default=20)),
        seed=seed,
        llm=llm,
    )
    print(f"[midi-preview] Generating 60s preview...", flush=True)
    print(f"[midi-preview] Prompt mode: {prompt_mode} | guidance={guidance_scale}", flush=True)
    print(f"[midi-preview] Caption: {tags}", flush=True)
    print(f"[midi-preview] Lyrics: {lyrics}", flush=True)
    params.duration = 60

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
    bpm_raw = job_input.get("bpm")
    bpm    = int(bpm_raw) if bpm_raw not in (None, "", 0, "0") else None
    key    = (job_input.get("key") or "").strip()
    prompt = (job_input.get("prompt") or job_input.get("style") or "").strip()
    style  = job_input.get("style", "").strip()
    user_p = prompt.lower()

    # Use shorter duration in MIDI mode (extraction + transcription is slower)
    _dur_raw = job_input.get("duration")
    duration = float(_dur_raw) if _dur_raw else 90.0
    duration = max(60.0, min(120.0, duration))

    seed = int(_job_value(job_input, "seed", default=random.randint(0, 2**31 - 1)))
    full_params, prompt_mode, tags, lyrics_str, g_scale = _build_generation_params(
        job_input=job_input,
        duration=duration,
        bpm=bpm,
        key=key,
        genre=genre,
        mood=mood,
        style=style,
        prompt=prompt,
        infer_step=20,
        seed=seed,
        llm=llm,
    )

    # ── 1. Generate full beat ─────────────────────────────────────────────────
    print(f"[midi-gen] Generating {duration:.0f}s beat for transcription...", flush=True)
    print(f"[midi-gen] Prompt mode: {prompt_mode} | guidance={g_scale}", flush=True)
    print(f"[midi-gen] Caption: {tags}", flush=True)
    print(f"[midi-gen] Lyrics: {lyrics_str}", flush=True)
    full_beat_path = _ace_generate(dit, llm, full_params, "/tmp/bh_midi_main.wav")
    main_audio     = _read_audio(full_beat_path)
    actual_dur     = (len(main_audio) if main_audio.ndim == 1 else main_audio.shape[0]) / SAMPLE_RATE
    effective_bpm  = full_params.bpm or 120
    total_beats    = round(actual_dur * effective_bpm / 60)
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
