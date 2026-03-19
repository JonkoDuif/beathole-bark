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

# Multiple variants per genre — one picked randomly each generation so beats
# never sound identical even with the same settings.
GENRE_VARIANTS = {
    "trap": [
        "dark Atlanta trap, thunderous 808 slides, chopped soul sample, ominous minor melody",
        "hard trap banger, distorted 808 sub bass, pitched vocal chop loop, menacing atmosphere",
        "midnight trap, slow dark 808, haunting piano sample, trap hi-hat rolls",
        "cinematic trap, orchestral strings over 808 bass, epic dark production, cold atmosphere",
        "melodic trap, emotional piano loop, heavy sliding 808, introspective dark energy",
    ],
    "drill": [
        "UK drill, cold sliding 808, sparse ominous melody, dark gritty street sound",
        "Brooklyn drill, hard knocking drums, eerie piano loop, aggressive dark production",
        "Chicago drill, menacing minor chords, thunderous kick, off-beat 808 hits",
        "UK drill banger, chilling string stabs, heavy sub bass, dark brooding atmosphere",
        "melodic drill, haunting flute loop over dark 808, cold emotional energy",
    ],
    "hip hop": [
        "soulful hip hop, warm vinyl-sampled piano, deep punchy kick, jazzy groove",
        "dusty hip hop, muffled soul chop, swinging hi-hats, warm analog texture",
        "dark hip hop, cinematic orchestral sample, boom bap rhythm, underground feel",
        "smooth hip hop, lush Rhodes melody, deep groove, relaxed head-nodding vibe",
        "raw hip hop, gritty sampled horn stab, knocking drums, street authenticity",
    ],
    "hip-hop": [
        "soulful hip hop, warm vinyl-sampled piano, deep punchy kick, jazzy groove",
        "dusty hip hop, muffled soul chop, swinging hi-hats, warm analog texture",
        "dark hip hop, cinematic orchestral sample, boom bap rhythm, underground feel",
        "smooth hip hop, lush Rhodes melody, deep groove, relaxed head-nodding vibe",
    ],
    "boom bap": [
        "classic boom bap, dusty vinyl sample, deep rimshot snare, loose swinging hi-hats",
        "golden era boom bap, soulful horn sample, hard punchy kick, jazz-infused groove",
        "gritty boom bap, chopped vocal sample, crackly vinyl texture, underground rawness",
        "cinematic boom bap, orchestral brass sample, booming kick, timeless old-school feel",
    ],
    "r&b": [
        "silky smooth R&B, warm Fender Rhodes chords, lush reverb, intimate late-night feel",
        "neo-soul R&B, live electric guitar licks, soft groove, sensual atmosphere",
        "dark R&B, minor key piano, deep bass pulse, moody emotional tension",
        "contemporary R&B, lush pad harmonies, airy hi-hats, polished modern feel",
        "soulful R&B, gospel-influenced chord progressions, warm vintage keys, heartfelt",
    ],
    "afrobeats": [
        "vibrant afrobeats, talking drum rhythms, bright guitar melody, joyful West African groove",
        "afrobeats banger, layered percussion, catchy synth hook, infectious energy",
        "amapiano-influenced afrobeats, log drum bass, hypnotic groove, South African sound",
        "afro fusion, kora-inspired melody, rich polyrhythmic percussion, warm organic feel",
    ],
    "dancehall": [
        "riddim dancehall, deep digital bass, offbeat guitar skank, Caribbean energy",
        "modern dancehall, trap-influenced rhythm, melodic synth hook, danceable groove",
        "lovers rock dancehall, smooth reggae feel, warm bass, romantic Caribbean vibe",
    ],
    "lo-fi": [
        "lo-fi hip hop, dusty vinyl crackle, mellow Rhodes melody, soft brushed drums, late night studying",
        "rainy day lo-fi, warm tape saturation, sleepy piano chords, soft kick, nostalgic haze",
        "jazzy lo-fi, muted guitar chords, vinyl grain texture, cozy introspective atmosphere",
        "lo-fi soul, gentle upright bass, warm Rhodes, dusty snare, slow meditative groove",
        "lo-fi chill, softly pitched vocal chop, mellow analog warmth, drifting daydream feel",
    ],
    "electronic": [
        "cutting-edge electronic, layered synthesizer arpeggios, explosive driving pulse, massive synth stabs, modern high-energy production",
        "experimental electronic, textured modular synth, evolving complex pads, intense futuristic atmosphere, relentless forward motion",
        "dark electronic, cold industrial synth, hypnotic relentless rhythm, heavy bass punch, underground club power",
        "melodic electronic, soaring emotional synth lead, arpeggiated chords building to euphoric peak, lush atmospheric depth",
        "hard electronic, brutal percussion hits, distorted bass pressure, intense aggressive energy, raw club force",
    ],
    "house": [
        "classic Chicago house, stomping four-on-the-floor kick, punchy soulful piano chords, warm analog bassline, peak-hour dancefloor energy",
        "funky house, driving chopped vocal stab, thick deep bass line, lively relentless dancefloor energy, full arrangement",
        "melodic house, euphoric emotional piano progression, soaring lush pads, powerful four-on-the-floor drive, crowd-lifting energy",
        "afro house, hypnotic tribal percussion layers, heavy groove, driving bassline, spiritual high-energy dancefloor power",
        "peak-hour house, aggressive bass stab, relentless kick drive, intense club energy, full powerful production",
    ],
    "deep house": [
        "deep house, powerful warm sub bass, rich atmospheric pads, deep introspective late-night groove, full mix depth",
        "soulful deep house, dusty organ chords, rolling punchy deep bass, emotional intimate dancefloor energy",
        "minimal deep house, hypnotic groove, deep resonant bass thump, dark underground tension, hypnotic forward drive",
        "vocal deep house, emotional chord progression, lush reverb depth, late night full-bodied warmth",
    ],
    "tech house": [
        "driving tech house, menacing hypnotic bassline, heavy punchy percussion, dark relentless underground groove, peak-hour intensity",
        "funky tech house, thick filtered bass stab, syncopated rhythm, high-energy peak-hour power, full arrangement",
        "industrial tech house, crushing distorted percussion, heavy driving kick, raw club energy, intense mechanical force",
        "hard tech house, aggressive bass pressure, relentless groove, dystopian underground energy, full powerful production",
    ],
    "techno": [
        "dark Berlin techno, crushing industrial kick drum, relentless hypnotic rhythm, cold mechanical energy, intense driving force",
        "hard techno, aggressive distorted kick, relentless hi-hats, dystopian atmosphere, brutal driving intensity, full power",
        "melodic techno, soaring emotional synth lead over relentless industrial rhythm, building cathartic peak-hour energy",
        "minimal techno, hypnotic deep resonant kick, relentless groove, rising tension, dark powerful underground drive",
        "peak techno, massive kick pressure, sweeping filter automation, relentless forward momentum, crowd-peak energy",
    ],
    "edm": [
        "festival EDM, massive soaring synth lead, explosive epic build-up, euphoric drop with crowd energy, full stadium production",
        "progressive EDM, emotional chord progression, huge soaring synth lead, euphoric breakdown into massive drop",
        "electro EDM, retro-futuristic synth stab, punchy four-on-the-floor, high-energy peak-hour intensity, full energetic production",
        "big room EDM, crushing kick drop, huge chord stabs, massive build tension release, overwhelming crowd energy",
        "electro house EDM, punchy distorted bass, relentless driving kick, explosive drop energy, high-intensity production",
    ],
    "dubstep": [
        "heavy riddim dubstep, crushing reese bass wobble, half-time 140 BPM kick, devastating sub-bass drop, filthy modulated bass growl",
        "brostep dubstep, massive distorted growl bass, raging aggressive percussion, face-melting drop intensity, full-frequency destruction",
        "melodic dubstep, soaring emotional synth lead, cinematic orchestral build, explosive huge drop with cathartic energy release",
        "dark dubstep, ominous cinematic tension build, industrial bass growl, bone-crushing drop, dystopian overwhelming energy",
        "neuro dubstep, complex bass sound design, technical rhythmic precision, intense mechanical energy, deep sub pressure",
        "future bass dubstep, lush chord swells building to heavy drop, emotional intensity, rich harmonic bass design",
    ],
    "drum and bass": [
        "liquid drum and bass, soulful Rhodes melody, fast rolling 170 BPM breaks, smooth energetic atmospheric feel, full production",
        "dark drum and bass, ominous aggressive bass wobble, frantic amen break, intense underground energy, relentless forward drive",
        "neurofunk drum and bass, complex technical bass modulation, precision percussion, futuristic intense mechanical energy",
        "jump up drum and bass, heavyweight punchy bass stab, energetic breakbeat, crowd-moving high-intensity energy",
        "hard drum and bass, crushing bass pressure, relentless fast break, intense raw underground power, full energy",
    ],
    "dnb": [
        "liquid DnB, warm atmospheric pads, rolling fast sub bass, soulful melodic forward momentum, full energetic feel",
        "dark DnB, aggressive bass pressure, rapid amen break, relentless underground raw energy, intense driving force",
        "neurofunk DnB, clinical bass design, technical rhythms, cold futuristic atmosphere, complex intense production",
        "jump up DnB, heavy punchy bass, energetic crowd-moving breakbeat, high-intensity dancefloor power",
    ],
    "jungle": [
        "classic jungle, chopped amen break, reggae bass wobble, raw 90s underground energy, frantic rhythmic intensity",
        "dark jungle, ominous bass line, frantic percussion complexity, intense rhythmic drive, powerful underground energy",
    ],
    "reggaeton": [
        "Latin reggaeton, punchy dembow rhythm, bright synth melody, urban Caribbean groove",
        "dark reggaeton, minor key dembow, moody synth, street urban Latin energy",
        "romantic reggaeton, smooth melody, warm production, sensual Latin groove",
    ],
    "latin": [
        "vibrant Latin beat, layered percussion, bright brass stabs, festive rhythmic energy",
        "Latin jazz fusion, complex piano chords, congas and timbales, rich harmonic groove",
        "dark Latin, minor key guitar, moody atmosphere, intense rhythmic tension",
    ],
    "jazz": [
        "late-night jazz, smoky walking bass, complex chord voicings, brushed snare, intimate atmosphere",
        "jazz fusion, electric piano, intricate rhythmic interplay, improvisational energy",
        "dark jazz, dissonant chords, sparse minor melody, noir cinematic atmosphere",
        "upbeat jazz, swinging brass section, lively groove, classic bebop energy",
    ],
    "soul": [
        "deep soul, warm organ chords, gospel-influenced melody, heartfelt emotional expression",
        "classic soul, lush string arrangement, punchy Motown groove, timeless emotional depth",
        "neo-soul, complex jazz chords, live bass pocket, introspective warm atmosphere",
        "southern soul, raw emotional delivery, gritty guitar licks, deep groove",
    ],
    "funk": [
        "classic funk, slapped bass groove, wah guitar, tight syncopated rhythm, James Brown influence",
        "jazz funk, complex chord progression, electric piano, deep pocket groove, sophisticated",
        "dark funk, minor key groove, heavy bass line, tense rhythmic interplay",
        "P-funk inspired, spacey synth over heavy groove, layered percussion, cosmic feel",
    ],
    "pop": [
        "polished pop, catchy piano hook, punchy modern production, hook-driven radio sound",
        "dark pop, minor key emotional melody, lush production, introspective depth",
        "indie pop, organic instrumentation, warm melody, bittersweet emotional atmosphere",
        "electropop, bright synth melody, contemporary production, energetic catchy sound",
    ],
    "cinematic": [
        "epic cinematic score, sweeping orchestral strings, dramatic percussion, powerful tension",
        "dark cinematic, dissonant brass stabs, ominous cello, tense suspenseful atmosphere",
        "emotional cinematic, solo piano over lush strings, heartbreaking melodic theme",
        "action cinematic, driving percussion, heroic brass, relentless forward momentum",
        "ambient cinematic, evolving atmospheric pads, sparse piano motif, vast spacious sound",
    ],
    "ambient": [
        "deep ambient, slowly evolving pad layers, minimal texture, vast spacious dreamscape",
        "dark ambient, ominous drone, unsettling atmospheric tension, cinematic darkness",
        "ambient electronic, crystalline synth arpeggios, peaceful floating atmosphere",
        "nature-inspired ambient, organic field recording texture, gentle melodic motif, serene",
    ],
    "phonk": [
        "dark Memphis phonk, distorted 808, chopped soul vocal, cowbell, vintage cassette texture",
        "aggressive phonk, heavy distorted kick, dark chord sample, raw underground energy",
        "drift phonk, dark synth atmosphere, hard 808, high-energy intense momentum",
        "melodic phonk, haunting vocal chop, distorted bass, eerie dark production",
    ],
    "cloud rap": [
        "hazy cloud rap, washed-out ambient trap, atmospheric reverb-drenched melody, ethereal feel",
        "dark cloud rap, ominous pad texture, distant pitched vocal, cold atmospheric trap",
        "melodic cloud rap, emotional piano under heavy reverb, dreamy introspective atmosphere",
    ],
    "grime": [
        "raw UK grime, staccato synth stabs, rolling 8-bar beat, aggressive dark energy",
        "dark grime, minor key synth riff, sparse industrial rhythm, cold London sound",
        "melodic grime, emotional keyboard lead, driving percussion, intense urban energy",
    ],
    "uk garage": [
        "UK garage, shuffled 2-step beat, pitched vocal chop, warm bass, soulful house influence",
        "speed garage, deep sub bass, skippy hi-hat rhythm, late-night underground groove",
        "dark UK garage, minor key synth, shuffled rhythm, moody introspective atmosphere",
    ],
    "synthwave": [
        "retro synthwave, pulsing analog bass, neon-lit arpeggios, 80s cinematic atmosphere",
        "dark synthwave, cold minor key synth, dystopian Blade Runner atmosphere, driving rhythm",
        "melodic synthwave, emotional lead synth, lush reverb, romantic 80s nostalgia",
        "outrun synthwave, powerful bass pulse, heroic lead melody, action-movie energy",
    ],
    "vaporwave": [
        "vaporwave, slowed pitched-down melody, lush reverb wash, nostalgic 80s corporate dream",
        "dark vaporwave, unsettling slowed sample, dreamlike distortion, lonely atmosphere",
        "mallsoft vaporwave, ambient elevator music deconstructed, surreal nostalgic texture",
    ],
    "hyperpop": [
        "hyperpop, glitchy distorted synth, maximalist layered production, chaotic energy",
        "dark hyperpop, heavy distorted 808, pitched vocal glitch, intense contrast",
        "melodic hyperpop, emotional lead over glitchy production, bittersweet energy",
    ],
    "pluggnb": [
        "dark pluggnb, smooth minor key melody, melodic trap groove, emotional introspective feel",
        "pluggnb, lush pad chords, slow dark 808, hazy romantic atmosphere",
        "melodic pluggnb, warm piano over slow trap rhythm, intimate emotional depth",
    ],
    "jersey club": [
        "jersey club, frantic 4x4 kick, chopped vocal sample, high-energy dancefloor groove",
        "dark jersey club, ominous bass stab, intense percussion, underground club energy",
    ],
    "reggae": [
        "classic reggae, offbeat guitar skank, one-drop rhythm, warm bass, roots vibration",
        "dub reggae, heavy bass, echo-soaked delay, sparse drumming, hypnotic trippy feel",
        "dancehall reggae, digital riddim, modern production, Caribbean energy",
    ],
    "gospel": [
        "powerful gospel, full choir harmony, driving rhythm, uplifting spiritual energy",
        "contemporary gospel, modern production, emotional piano, soulful choir texture",
        "dark gospel, minor key spiritual tension, raw emotional power, building drama",
    ],
    "blues": [
        "delta blues, raw electric guitar riff, gritty 12-bar progression, soulful expression",
        "modern blues, emotional lead guitar, deep groove, heartfelt authentic feel",
        "blues jazz fusion, complex chord progression, expressive guitar, sophisticated groove",
    ],
    "rock": [
        "hard rock, driving electric guitar riff, powerful live drums, raw energy",
        "indie rock, warm guitar tone, melodic hook, authentic organic sound",
        "dark rock, heavy distorted guitar, minor key tension, intense brooding atmosphere",
    ],
    "rap": [
        "hard rap beat, punchy drums, dark sample, street energy, raw production",
        "melodic rap, emotional piano loop, deep bass, introspective atmosphere",
    ],
}

# Rich mood descriptions — multiple variants per mood, one picked randomly
MOOD_VARIANTS = {
    "dark": [
        "dark ominous minor key, menacing brooding tension, cold threatening atmosphere",
        "pitch-black darkness, slow-burning dread, unsettling harmonic dissonance",
        "deeply dark production, haunting minor chords, shadowy oppressive atmosphere",
        "noir darkness, tense cinematic atmosphere, cold emotionless menace",
    ],
    "sad": [
        "heartbreaking melancholy, tearful minor key, deep emotional sorrow, bittersweet pain",
        "devastating sadness, lonely atmosphere, raw emotional vulnerability, quiet grief",
        "melancholic introspection, aching melody, soulful heartbreak, late-night loneliness",
        "mournful sadness, weeping lead melody, slow emotional decay, poignant atmosphere",
    ],
    "emotional": [
        "deeply emotional, raw vulnerability, heartfelt honest expression, moving atmosphere",
        "overwhelming emotion, intense personal feeling, powerful inner journey",
        "cathartic emotional release, deeply touching, resonant human connection",
    ],
    "energetic": [
        "explosive high energy, adrenaline rush, relentless forward momentum, kinetic power",
        "driving intense energy, unstoppable force, electrifying rhythm, powerful momentum",
        "frenetic high-octane energy, fast and furious, heart-pounding intensity",
    ],
    "chill": [
        "mellow laid-back groove, smooth relaxed atmosphere, easygoing Sunday afternoon feel",
        "soft chill vibe, gentle floating melody, unhurried peaceful mood, calm tranquility",
        "warm chill atmosphere, hazy comfort, slow-breathing relaxation, soft and easy",
    ],
    "aggressive": [
        "brutally aggressive, hard-hitting in-your-face, relentless raw force, confrontational",
        "savage intensity, violent energy, crushing heavy production, overwhelming aggression",
        "menacing aggression, cold calculated force, dominant threatening energy",
    ],
    "uplifting": [
        "uplifting hopeful energy, soaring major key, inspiring triumph, bright optimism",
        "euphoric uplift, joyful emotional release, spiritual elevation, warm radiant energy",
        "motivating uplifting power, victorious momentum, hopeful bright atmosphere",
    ],
    "mysterious": [
        "mysterious eerie tension, haunting unknown, suspenseful cinematic atmosphere",
        "enigmatic dark mystery, unsettling curiosity, strange other-worldly feel",
        "deep mysterious groove, hypnotic trance-like unknown, dark compelling atmosphere",
    ],
    "romantic": [
        "intimate romantic warmth, sensual smooth groove, tender emotional closeness",
        "passionate romance, deep feeling, lush warm atmosphere, heartfelt connection",
        "late-night romantic mood, silky smooth production, soft emotional intimacy",
    ],
    "hard": [
        "hard-hitting heavy production, thick punishing bass, aggressive powerful drums",
        "heavy knocking, dense brutal sound, walls of bass, relentless hard energy",
        "hard street sound, heavy low-end weight, tough uncompromising production",
    ],
    "melodic": [
        "richly melodic, expressive lead instrument, deep harmonic layers, strong memorable hooks",
        "beautiful melodic composition, emotional lead melody, lush harmonic depth",
        "soulful melodic expression, singing lead line, rich emotional musical development",
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
    ],
    "nostalgic": [
        "warm nostalgic longing, vintage analog texture, bittersweet memory, timeless past",
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
# These explicitly describe what a FULL, layered production sounds like
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
    combined = (style + " " + user_prompt).lower()
    parts = []
    if any(x in combined for x in ["hard drum", "heavy drum", "aggressive drum"]):
        parts.append("hard-hitting aggressive drums")
    elif any(x in combined for x in ["soft drum", "light drum", "minimal drum"]):
        parts.append("soft minimal drums")
    if any(x in combined for x in ["hard kick", "heavy kick", "punchy kick", "808"]):
        parts.append("heavy punchy kick drum")
    if any(x in combined for x in ["rolling hi-hat", "fast hi-hat", "triplet hi-hat"]):
        parts.append("rolling triplet hi-hats")
    elif any(x in combined for x in ["soft hi-hat", "light hi-hat", "no hi-hat"]):
        parts.append("soft minimal hi-hats")
    if not parts:
        genre_defaults = {
            "trap":          "heavy 808 kick, rolling triplet hi-hats, snare on beat 3",
            "drill":         "heavy booming kick, sparse hi-hats, sliding 808",
            "hip hop":       "punchy kick on 1 and 3, crisp snare on 2 and 4, swinging hi-hats",
            "boom bap":      "deep sampled kick, rimshot snare, loose swinging hi-hats with ghost notes",
            "r&b":           "smooth kick, snare on 2 and 4, shuffled hi-hats",
            "house":         "four-on-the-floor kick every beat, open hi-hat on offbeats, clap on 2 and 4",
            "deep house":    "soft four-on-the-floor kick, open hi-hat offbeats, subtle groove",
            "techno":        "hard industrial kick on every beat, aggressive 16th-note hi-hats",
            "edm":           "four-on-the-floor kick, clap on 2 and 4, build-up snare rolls",
            "dubstep":       "half-time kick, massive snare on beat 3, sparse hi-hats",
            "drum and bass": "fast 170 BPM breakbeat, syncopated kick and snare, rolling hi-hats",
            "phonk":         "heavy distorted kick, trap snare, rolling hi-hats, cowbell accents",
            "lo-fi":         "mellow kick with vinyl grain, soft brushed snare, dusty hi-hats",
            "funk":          "syncopated funk kick, tight snare ghost notes, open hi-hat choke",
            "afrobeats":     "syncopated afrobeat kick, talking drum, layered shaker and conga",
            "reggaeton":     "dembow pattern kick on 1 and and-3, snare on 2 and 4",
        }
        parts.append(genre_defaults.get(genre, "standard drums, kick and snare, hi-hats"))
    return ", ".join(parts)


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

    parts = []

    # ── Explicit instrument override (FIRST so ACE-Step can't ignore it) ─────
    # Check longest keyword first (e.g. "acoustic guitar" before "guitar",
    # "dark piano" before "piano") so the most specific match wins.
    for kw in sorted(_INSTRUMENT_FORCE_TAGS.keys(), key=len, reverse=True):
        if kw in p_lower:
            parts.append(random.choice(_INSTRUMENT_FORCE_TAGS[kw]))
            break  # only inject the most specific match

    # ── Genre variant ─────────────────────────────────────────────────────────
    genre_variants = GENRE_VARIANTS.get(g)
    if genre_variants:
        parts.append(random.choice(genre_variants))
    elif g:
        parts.append(g)

    # ── Genre-specific energy tag ─────────────────────────────────────────────
    energy_tags = _GENRE_ENERGY_TAGS.get(g)
    if energy_tags:
        parts.append(random.choice(energy_tags))

    # ── "Energic" boost — override everything with maximum intensity tags ─────
    if is_energic:
        parts.append(random.choice(_ENERGIC_BOOST_TAGS))
        # Add a second energy descriptor for extra emphasis
        parts.append("massive wall of sound, relentless high-energy arrangement, powerful full-mix impact")

    # ── Mood ──────────────────────────────────────────────────────────────────
    mood_variants = MOOD_VARIANTS.get(m)
    if mood_variants:
        parts.append(random.choice(mood_variants))
    elif m:
        parts.append(m)

    # ── User prompt ───────────────────────────────────────────────────────────
    if prompt:
        parts.append(prompt)

    # ── Atmospheric feel tag (60% chance) ─────────────────────────────────────
    if random.random() < 0.60:
        parts.append(random.choice(_FEEL_TAGS))

    # ── Mix fullness tag — always injected to push toward richer arrangements ─
    parts.append(random.choice(_FULLNESS_TAGS))

    # ── Production texture ────────────────────────────────────────────────────
    textures = _PRODUCTION_TEXTURES.get(g, _PRODUCTION_TEXTURES["_default"])
    parts.append(random.choice(textures))

    # ── Production technique ──────────────────────────────────────────────────
    parts.append(random.choice(_PRODUCTION_TECHNIQUES))

    # ── Harmonic color (40% chance) ───────────────────────────────────────────
    if random.random() < 0.40:
        parts.append(random.choice(_HARMONIC_COLORS))

    # ── Quality / energy footer ───────────────────────────────────────────────
    parts.append("instrumental, no vocals, no singing, no spoken words")
    if bpm:
        parts.append(f"{bpm} BPM")
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
