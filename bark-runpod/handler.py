"""
BeatHole AI — ACE-Step 1.5 RunPod Handler

Audio mode : ACE-Step 1.5 text2music → stem extraction (drums/bass/melody/strings/synth)
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

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48000   # ACE-Step native output sample rate
STEM_SR     = 22050   # stem upload sample rate — good quality, manageable size

_dit_handler = None
_llm_handler = None

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
    global _dit_handler, _llm_handler
    if _dit_handler is not None:
        return _dit_handler, _llm_handler

    project_root = os.environ.get("ACESTEP_PROJECT_ROOT", "/app")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    lm_model     = os.environ.get("ACESTEP_LM_MODEL", "acestep-5Hz-lm-0.6B")

    print(f"[acestep] Loading handlers (device={device}, project_root={project_root})...", flush=True)

    _dit_handler = _DitHandlerClass()
    _llm_handler = _LLMHandlerClass()

    if hasattr(_dit_handler, "initialize_service"):
        # Patch from_pretrained so meta tensors are never created during DiT init.
        orig_fp = _patch_low_cpu_mem()

        # ACE-Step/Ace-Step1.5 on HuggingFace is the turbo model.
        # Try turbo first, fall back to base.
        for config_name in ["acestep-v15-turbo", "acestep-v15-base"]:
            print(f"[acestep] initialize_service(project_root={project_root}, config_path={config_name}, device={device})", flush=True)
            try:
                _dit_handler.initialize_service(
                    project_root=project_root,
                    config_path=config_name,
                    device=device,
                )
            except Exception as e:
                print(f"[acestep] initialize_service raised: {e}", flush=True)

            # initialize_service swallows internal errors; check if model loaded.
            if getattr(_dit_handler, "model", None) is not None:
                print(f"[acestep] DIT model loaded OK (config_path={config_name})", flush=True)
                break
            print(f"[acestep] DIT model is None after config_path={config_name}, trying next...", flush=True)
            # Reset handler state for next attempt
            _dit_handler.model = None
        else:
            _restore_pretrained(orig_fp)
            raise RuntimeError(
                "AceStepHandler: model is None after all config_path attempts. "
                "Check logs above for the underlying error."
            )

        _restore_pretrained(orig_fp)

        # Verify all required components are present
        missing = [c for c in ("model", "vae", "text_encoder", "text_tokenizer")
                   if getattr(_dit_handler, c, None) is None]
        if missing:
            raise RuntimeError(f"AceStepHandler missing components after init: {missing}")
        print("[acestep] All DIT components loaded", flush=True)

        # LLM is optional — generation works without it (no chain-of-thought)
        lm_checkpoint_dir = os.path.join(project_root, "checkpoints")
        try:
            _llm_handler.initialize(
                checkpoint_dir=lm_checkpoint_dir,
                lm_model_path=lm_model,
                device=device,
            )
            print("[acestep] LLM handler initialized", flush=True)
        except Exception as lm_err:
            print(f"[acestep] LLM init failed (non-fatal): {lm_err}", flush=True)
            _llm_handler = None
    else:
        # ACE-Step 1.0 fallback
        _dit_handler = _DitHandlerClass(model_name="acestep-5Hz-dit-base")
        _llm_handler = _LLMHandlerClass(model_name=lm_model)

    print("[acestep] Handlers ready", flush=True)
    return _dit_handler, _llm_handler


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

GENRE_BASES = {
    "trap":          "trap beat, Atlanta trap, hard 808 bass, dark production",
    "drill":         "UK drill, dark gritty production, London drill style",
    "hip hop":       "hip hop beat, boom bap, groove-driven rhythm",
    "hip-hop":       "hip hop beat, boom bap production style",
    "boom bap":      "classic boom bap, vinyl-texture, golden era hip hop",
    "r&b":           "R&B, smooth soulful production, warm polished mix",
    "afrobeats":     "afrobeats, vibrant African percussion groove",
    "dancehall":     "dancehall riddim, Caribbean rhythm, digital production",
    "lo-fi":         "lo-fi hip hop, vintage warm texture, chill nostalgic aesthetic",
    "electronic":    "electronic music, synthesizer-driven, modern production",
    "house":         "house music, four-on-the-floor groove, electronic dance",
    "deep house":    "deep house, warm sub bass, soulful house groove",
    "tech house":    "tech house, punchy percussion, hypnotic groove",
    "techno":        "techno, industrial percussion, dark driving rhythm",
    "edm":           "EDM, energetic build-ups and drops, powerful electronic",
    "dubstep":       "dubstep, massive wobble bass, half-time rhythm, heavy drops",
    "drum and bass": "drum and bass, fast breakbeats, heavy sub bass",
    "dnb":           "drum and bass, fast amen break, rolling sub bass",
    "jungle":        "jungle, chopped amen break, reggae bass, 160 BPM",
    "reggaeton":     "reggaeton, dembow rhythm, Latin urban production",
    "latin":         "Latin beat, percussion-driven, rhythmic Latin groove",
    "jazz":          "jazz-influenced, complex chord harmony, live instrumentation",
    "soul":          "soul music, warm live instrumentation, groove-based",
    "funk":          "funk beat, syncopated groove, punchy bass",
    "pop":           "pop music, catchy hook-driven melody, polished radio production",
    "cinematic":     "cinematic orchestral score, epic dramatic production",
    "ambient":       "ambient electronic, atmospheric pads, spacious textured production",
    "phonk":         "phonk, Memphis rap style, dark distorted 808, cowbell",
    "cloud rap":     "cloud rap, hazy atmospheric production, ambient trap",
    "grime":         "grime, UK urban production, staccato synths, 140 BPM",
    "uk garage":     "UK garage, shuffled 2-step beat, soulful groove",
    "synthwave":     "synthwave, 80s retro electronic, analog synthesizer",
    "vaporwave":     "vaporwave, slowed retro aesthetic, nostalgic electronic",
    "hyperpop":      "hyperpop, glitchy distorted pop, maximalist electronic",
    "pluggnb":       "pluggnb, smooth dark melodic trap, melodic plug beat",
    "jersey club":   "jersey club, chopped samples, 4x4 kick, club energy",
    "reggae":        "reggae, offbeat guitar skank, one-drop rhythm, Caribbean groove",
    "gospel":        "gospel, powerful choir, driving rhythm, uplifting production",
    "blues":         "blues, guitar riff, 12-bar progression, soulful expression",
    "rock":          "rock, live drums, electric guitar, bass-driven groove",
}

MOOD_MAP = {
    "dark":       "dark minor key, ominous tense atmosphere, menacing brooding",
    "sad":        "melancholic sad melody, minor key emotion, heartbreaking",
    "emotional":  "deeply emotional, heartfelt raw atmosphere, touching",
    "energetic":  "high energy driving intensity, powerful momentum, adrenaline",
    "chill":      "chill relaxed laid-back vibe, mellow smooth, easygoing",
    "aggressive": "aggressive hard-hitting intense, powerful in-your-face, raw force",
    "uplifting":  "uplifting positive bright energy, major key hopeful, inspirational",
    "mysterious": "mysterious eerie haunting, suspenseful tension, cinematic dark",
    "romantic":   "romantic warm smooth, sensual soulful expression, intimate",
    "hard":       "hard-hitting heavy punchy, thick powerful bass, aggressive drums",
    "melodic":    "melodic expressive lead, rich harmonic depth, strong hooks",
    "happy":      "happy bright cheerful major key, feel-good positive, joyful",
    "angry":      "angry aggressive intense, distorted heavy energy, raw power",
    "nostalgic":  "nostalgic warm retro, vintage analog texture, timeless",
    "epic":       "epic cinematic power, massive orchestral build, dramatic",
    "dreamy":     "dreamy ethereal floating, hazy reverb-soaked texture, hypnotic",
    "bouncy":     "bouncy playful groove, catchy rhythmic, fun energetic swing",
    "raw":        "raw unpolished gritty energy, underground authentic, street sound",
}

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


def build_tags(prompt: str, genre: str, mood: str, bpm: int, key: str) -> str:
    parts = []
    genre_base = GENRE_BASES.get(genre.lower() if genre else "", f"{genre}" if genre else "")
    if genre_base:
        parts.append(genre_base)
    mood_desc = MOOD_MAP.get(mood.lower() if mood else "", mood if mood else "")
    if mood_desc:
        parts.append(mood_desc)
    if prompt:
        parts.append(prompt)
    parts.append("instrumental, no vocals, no singing, no spoken words")
    if bpm:
        parts.append(f"{bpm} BPM")
    if key:
        parts.append(f"key of {key}")
    parts.append("professional studio quality, clean mix, well-produced beat")
    # Deduplicate preserving order
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return ", ".join(out)


def _build_lyrics_structure(duration: float) -> str:
    """Section markers guide ACE-Step's internal arrangement dynamics."""
    if duration <= 100:
        secs = ["[intro]", "[verse]", "[chorus]", "[verse]", "[outro]"]
    elif duration <= 160:
        secs = ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[outro]"]
    elif duration <= 220:
        secs = ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]", "[bridge]", "[outro]"]
    else:
        secs = ["[intro]", "[verse]", "[chorus]", "[verse]", "[chorus]",
                "[verse]", "[chorus]", "[bridge]", "[outro]"]
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


def generate_audio_with_stems(job_input: dict) -> dict:
    """
    1. Generate the full beat with ACE-Step 1.5 text2music (coherent, properly structured).
    2. Extract stems (drums, bass, melody/instrument) from the generated audio via
       ACE-Step's extract task — stems always match the beat 100%.
    3. Upload stems to backend. Return combined beat WAV + stem URLs.
    """
    dit, llm = get_handlers()

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
    lyrics_str = _build_lyrics_structure(duration)
    seed       = random.randint(0, 2**31 - 1)

    print(f"[gen] Tags: {tags}", flush=True)
    print(f"[gen] Duration: {duration:.0f}s | BPM: {bpm} | Key: {key} | Seed: {seed}", flush=True)

    # Handle reference audio → temp file
    ref_path = None
    if ref_audio_b64:
        ref_path = "/tmp/bh_reference.wav"
        with open(ref_path, "wb") as f:
            f.write(base64.b64decode(ref_audio_b64))
        print(f"[gen] Reference audio saved", flush=True)

    # ── 1. Generate each stem independently ──────────────────────────────────
    # Same seed + BPM + key → stems sit on the same rhythmic grid.
    # The main beat is built by mixing all generated stems together, giving
    # perfectly clean, bleed-free stem files for studio use.
    stems_to_gen = _stems_to_generate(genre, style, user_prompt)
    print(f"[stems] Generating {len(stems_to_gen)} stems: {[s for s,_ in stems_to_gen]}", flush=True)

    stems_audio = {}   # stem_name → np.ndarray (native SAMPLE_RATE, possibly stereo)
    for ace_stem, display_name in stems_to_gen:
        print(f"[stems] Generating '{ace_stem}' stem ({duration:.0f}s)...", flush=True)
        try:
            caption = _stem_tags(ace_stem, tags)
            params  = GenerationParams(
                task_type       = "text2music",
                caption         = caption,
                lyrics          = lyrics_str,
                duration        = int(duration),
                bpm             = bpm,
                keyscale        = key if key else None,
                timesignature   = "4/4",
                inference_steps = infer_step,
                # High guidance_scale forces strict adherence to the stem caption,
                # minimising bleed from other instruments (7.5 is too permissive).
                guidance_scale  = 15.0,
                seed            = seed,  # identical seed → same rhythmic grid
                thinking        = False, # no chain-of-thought for stems (faster)
            )
            # Apply reference audio to every stem so they share the same style
            if ref_path:
                params.audio2audio_enable  = True
                params.ref_audio_input     = ref_path
                params.ref_audio_strength  = ref_strength
            stem_path  = _ace_generate(dit, llm, params, f"/tmp/bh_stem_{ace_stem}.wav")
            stem_audio = _read_audio(stem_path)
            stems_audio[ace_stem] = (stem_audio, display_name)
            print(f"[stems] '{ace_stem}' done — shape: {stem_audio.shape}", flush=True)
        except Exception as e:
            print(f"[stems] '{ace_stem}' FAILED: {e}", flush=True)

    # ── 2. Mix all stems into the main beat ──────────────────────────────────
    def _to_stereo_float32(audio: np.ndarray) -> np.ndarray:
        """Ensure (samples, 2) float32."""
        a = audio.astype(np.float32)
        if a.ndim == 1:
            a = np.stack([a, a], axis=1)
        elif a.ndim == 2 and a.shape[0] < a.shape[1]:
            a = a.T  # (channels, samples) → (samples, channels)
        if a.shape[1] == 1:
            a = np.concatenate([a, a], axis=1)
        return a

    if stems_audio:
        arrays = [_to_stereo_float32(aud) for aud, _ in stems_audio.values()]
        max_len = max(a.shape[0] for a in arrays)
        padded  = [np.pad(a, ((0, max_len - a.shape[0]), (0, 0))) for a in arrays]
        main_audio = np.sum(padded, axis=0)
        # Normalize to prevent clipping
        peak = np.max(np.abs(main_audio))
        if peak > 0.95:
            main_audio = main_audio * (0.90 / peak)
        print(f"[gen] Mixed {len(stems_audio)} stems into main beat, shape: {main_audio.shape}", flush=True)
    else:
        # Fallback: generate a plain full-mix beat if all stems failed
        print("[gen] All stems failed — generating fallback full-mix beat", flush=True)
        fb_params = GenerationParams(
            task_type="text2music", caption=tags, lyrics=lyrics_str,
            duration=int(duration), bpm=bpm, keyscale=key if key else None,
            timesignature="4/4", inference_steps=infer_step, guidance_scale=7.5, seed=seed,
        )
        if ref_path:
            fb_params.audio2audio_enable = True
            fb_params.ref_audio_input    = ref_path
            fb_params.ref_audio_strength = ref_strength
        main_path  = _ace_generate(dit, llm, fb_params, "/tmp/bh_main.wav")
        main_audio = _read_audio(main_path)

    actual_dur = (main_audio.shape[0] if main_audio.ndim == 2 else len(main_audio)) / SAMPLE_RATE

    # ── 3. Upload stems to backend ───────────────────────────────────────────
    backend_url  = os.environ.get("BACKEND_URL", "").rstrip("/")
    internal_key = os.environ.get("INTERNAL_API_KEY", "")
    stem_urls    = {}

    if backend_url and internal_key and beat_id:
        for ace_stem, (audio, _display) in stems_audio.items():
            try:
                audio_down = resample_mono(audio, SAMPLE_RATE, STEM_SR)
                b64  = np_to_wav_b64(audio_down, sr=STEM_SR)
                url  = upload_stem(backend_url, internal_key, beat_id, ace_stem, b64)
                stem_urls[ace_stem] = url
                print(f"[stems] Uploaded '{ace_stem}' → {url}", flush=True)
            except Exception as e:
                print(f"[stems] Upload failed for '{ace_stem}': {e}", flush=True)
    else:
        print("[stems] Skipping upload — env vars not set", flush=True)

    # ── 4. Upload main audio (avoids RunPod 10 MB payload limit) ────────────
    wav_url = None
    if backend_url and internal_key and beat_id:
        try:
            main_b64 = np_to_wav_b64(main_audio)
            wav_url  = upload_main_audio(backend_url, internal_key, beat_id, main_b64)
            print(f"[gen] Uploaded main beat → {wav_url}", flush=True)
        except Exception as e:
            print(f"[gen] Main beat upload failed: {e}", flush=True)

    # ── 5. Return ────────────────────────────────────────────────────────────
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
    dit, llm = get_handlers()

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
        keyscale       = key if key else None,
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
# ── RunPod handler ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def handler(job: dict) -> dict:
    job_input   = job.get("input", {})
    output_mode = job_input.get("output_mode", "audio")

    print(f"[job] mode={output_mode}", flush=True)

    try:
        if output_mode == "midi":
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
