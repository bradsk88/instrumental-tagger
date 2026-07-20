import os
import sys
import glob
import wave
import array
import math
import tempfile
import shutil
from mutagen.flac import FLAC
from mutagen.easyid3 import EasyID3
from mutagen.wave import WAVE
from mutagen.id3 import ID3NoHeaderError, TXXX

# Forcefully silence Essentia's internal C++ logging before standard imports
import essentia
essentia.log.warningActive = False
essentia.log.infoActive = False

# Silence TensorFlow CPU memory warnings and info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import essentia.standard as es
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MUSIC_DIR = "/music"
VGGISH_MODEL = "/app/models/audioset-vggish-3.pb"
CLASSIFIER_MODEL = "/app/models/voice_instrumental-audioset-vggish-1.pb"

def _env_flag(name, default=False):
    """Reads a boolean-ish environment variable (1/true/yes/on)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

# Set this to True to overwrite previously written tags with fresh ML predictions.
# Set it to False once your library is clean to make future scans ultra-fast.
# Override via the FORCE_RETAG env var (e.g. FORCE_RETAG=1 or FORCE_RETAG=true).
FORCE_RETAG = _env_flag("FORCE_RETAG", False)

# Store the raw ML instrumental confidence (0-100%) as its own tag so that the
# label can later be recomputed at a different threshold without re-analyzing.
# Override via the STORE_CONFIDENCE env var (default: enabled).
STORE_CONFIDENCE = _env_flag("STORE_CONFIDENCE", True)

# Confidence percentage (0-100) at or above which a track is considered
# INSTRUMENTAL. Override via the INSTRUMENTAL_THRESHOLD env var (default: 50).
try:
    INSTRUMENTAL_THRESHOLD = float(os.environ.get("INSTRUMENTAL_THRESHOLD", "50"))
except ValueError:
    INSTRUMENTAL_THRESHOLD = 50.0

# When a track already has a stored confidence tag, re-derive its label from
# that value using the current threshold instead of running ML again. This makes
# threshold changes near-instant across a large library. Disable to force fresh
# analysis. Override via the SKIP_ANALYZED env var (default: enabled).
SKIP_ANALYZED = _env_flag("SKIP_ANALYZED", True)
# ==============================================================================

# Register custom tag mappings for EasyID3
EasyID3.RegisterTXXXKey("instrumental", "INSTRUMENTAL")
EasyID3.RegisterTXXXKey("instrumental_confidence", "INSTRUMENTAL_CONFIDENCE")

def get_existing_tag(filepath, ext):
    """
    Checks if a file is already tagged. 
    Returns ('INSTRUMENTAL (1)' or 'VOCAL (0)') if tagged, otherwise None.
    """
    try:
        if ext == '.flac':
            audio = FLAC(filepath)
            if 'INSTRUMENTAL' in audio:
                val = audio['INSTRUMENTAL'][0]
                return "INSTRUMENTAL (1)" if val == "1" else "VOCAL (0)"
        elif ext == '.mp3':
            try:
                audio = EasyID3(filepath)
                if 'instrumental' in audio:
                    val = audio['instrumental'][0]
                    return "INSTRUMENTAL (1)" if val == "1" else "VOCAL (0)"
            except ID3NoHeaderError:
                return None
        elif ext == '.wav':
            audio = WAVE(filepath)
            if audio.tags is not None:
                for frame in audio.tags.getall("TXXX"):
                    if frame.desc == "INSTRUMENTAL":
                        val = frame.text[0]
                        return "INSTRUMENTAL (1)" if val == "1" else "VOCAL (0)"
    except Exception:
        return None
    return None

def get_confidence_tag(filepath, ext):
    """
    Reads the previously stored INSTRUMENTAL_CONFIDENCE tag (a 0-100 percentage).
    Returns the float value if present, otherwise None.
    """
    try:
        if ext == '.flac':
            audio = FLAC(filepath)
            if 'INSTRUMENTAL_CONFIDENCE' in audio:
                return float(audio['INSTRUMENTAL_CONFIDENCE'][0])
        elif ext == '.mp3':
            try:
                audio = EasyID3(filepath)
                if 'instrumental_confidence' in audio:
                    return float(audio['instrumental_confidence'][0])
            except ID3NoHeaderError:
                return None
        elif ext == '.wav':
            audio = WAVE(filepath)
            if audio.tags is not None:
                for frame in audio.tags.getall("TXXX"):
                    if frame.desc == "INSTRUMENTAL_CONFIDENCE":
                        return float(frame.text[0])
    except Exception:
        return None
    return None

def tag_file(filepath, is_instrumental, ext, confidence_pct=None):
    """
    Writes the instrumental tag (1 for true, 0 for false) to the audio file metadata.
    When STORE_CONFIDENCE is enabled and confidence_pct is provided, also writes
    the raw INSTRUMENTAL_CONFIDENCE percentage so the label can be recomputed at a
    different threshold later without re-analyzing.
    """
    tag_val = "1" if is_instrumental else "0"
    write_conf = STORE_CONFIDENCE and confidence_pct is not None
    conf_val = f"{confidence_pct:.1f}" if write_conf else None
    try:
        if ext == '.flac':
            audio = FLAC(filepath)
            audio['INSTRUMENTAL'] = tag_val
            if write_conf:
                audio['INSTRUMENTAL_CONFIDENCE'] = conf_val
            audio.save()
        elif ext == '.mp3':
            try:
                audio = EasyID3(filepath)
            except ID3NoHeaderError:
                audio = EasyID3()
            audio['instrumental'] = tag_val
            if write_conf:
                audio['instrumental_confidence'] = conf_val
            audio.save(filepath)
        elif ext == '.wav':
            audio = WAVE(filepath)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.delall("TXXX:INSTRUMENTAL")
            audio.tags.add(TXXX(encoding=3, desc="INSTRUMENTAL", text=[tag_val]))
            if write_conf:
                audio.tags.delall("TXXX:INSTRUMENTAL_CONFIDENCE")
                audio.tags.add(TXXX(encoding=3, desc="INSTRUMENTAL_CONFIDENCE", text=[conf_val]))
            audio.save()
        return True
    except Exception as e:
        print(f"\n❌ Error tagging {filepath}: {e}", flush=True)
        return False

def analyze_track_ml(filepath):
    """
    Loads 45 seconds of audio, extracts VGGish embeddings, and classifies
    vocal vs instrumental using TensorFlow.

    Returns the instrumental confidence as a 0-100 percentage, or None on failure.
    The caller applies INSTRUMENTAL_THRESHOLD to decide the final label.
    """
    try:
        # Preferred analysis window: 40s-85s (skips intros/outros). Short tracks
        # yield little or no audio there (a 41s song gives ~1s, and anything
        # under 40s gives none), which makes for a weak or empty VGGish input.
        # Fall back to the whole track whenever the window is too short.
        SAMPLE_RATE = 16000
        MIN_FALLBACK_SAMPLES = 10 * SAMPLE_RATE  # need >=10s of audio to trust the window

        loader = es.EasyLoader(filename=filepath, sampleRate=SAMPLE_RATE, startTime=40, endTime=85)
        audio = loader()

        if len(audio) < MIN_FALLBACK_SAMPLES:
            audio = es.MonoLoader(filename=filepath, sampleRate=SAMPLE_RATE)()

        if len(audio) == 0:
            print(f"\n❌ ML Analysis failed for {os.path.basename(filepath)}: "
                  f"no audio samples could be loaded", flush=True)
            return None

        vggish = es.TensorflowPredictVGGish(
            graphFilename=VGGISH_MODEL, 
            output="model/vggish/embeddings"
        )
        embeddings = vggish(audio)
        
        classifier = es.TensorflowPredict2D(
            graphFilename=CLASSIFIER_MODEL,
            input="model/Placeholder",  
            output="model/Softmax"       
        )
        predictions = classifier(embeddings)
        
        # Predictions shape is [frame, class_probabilities]
        # index 0 is Instrumental, index 1 is Vocal
        instrumental_prob = np.mean(predictions[:, 0])
        return float(instrumental_prob) * 100.0

    except Exception as e:
        print(f"\n❌ ML Analysis failed for {os.path.basename(filepath)}: {e}", flush=True)
        return None

def label_for(confidence_pct):
    """A track is INSTRUMENTAL when its confidence meets the threshold."""
    return confidence_pct >= INSTRUMENTAL_THRESHOLD

def process_file(filepath):
    _, ext = os.path.splitext(filepath.lower())
    if ext not in ['.mp3', '.flac', '.wav']:
        return

    filename = os.path.basename(filepath)

    # Fast path: reuse a previously stored confidence value instead of running ML.
    # This lets a threshold change re-label the whole library near-instantly.
    stored_confidence = get_confidence_tag(filepath, ext)
    if stored_confidence is not None and SKIP_ANALYZED and not FORCE_RETAG:
        is_instrumental = label_for(stored_confidence)
        status = "INSTRUMENTAL (1)" if is_instrumental else "VOCAL (0)"
        if tag_file(filepath, is_instrumental, ext, stored_confidence):
            print(f"⚡ Re-labeled from stored confidence "
                  f"({stored_confidence:.1f}% >= {INSTRUMENTAL_THRESHOLD:g}%? "
                  f"-> {status}): {filename}", flush=True)
        else:
            print(f"❌ Failed to re-label: {filename}", flush=True)
        return

    # Otherwise fall back to the existing label tag to decide whether to skip.
    existing_tag = get_existing_tag(filepath, ext)
    if existing_tag is not None and stored_confidence is None and not FORCE_RETAG:
        print(f"⏭️  Already Tagged as {existing_tag}: {filename}", flush=True)
        return

    # Run ML analysis
    prefix = f"🔄 Re-analyzing (Was {existing_tag}): " if existing_tag else "🔍 Analyzing: "
    print(f"{prefix}{filename}...", end="", flush=True)

    confidence = analyze_track_ml(filepath)

    if confidence is not None:
        is_instrumental = label_for(confidence)
        if tag_file(filepath, is_instrumental, ext, confidence):
            status = "INSTRUMENTAL (1)" if is_instrumental else "VOCAL (0)"
            print(f" Done! -> Tagged as {status} (Conf: {confidence:.1f}%)", flush=True)
        else:
            print(" Failed to tag.", flush=True)
    else:
        print(" Failed to analyze.", flush=True)

def main():
    print("==================================================", flush=True)
    print("🚀 Starting Instrumental Tagging Scan...", flush=True)
    print(f"🎚️  Instrumental threshold: {INSTRUMENTAL_THRESHOLD:g}%", flush=True)
    print(f"💾 Store confidence tag: {'ON' if STORE_CONFIDENCE else 'OFF'}", flush=True)
    print(f"⏩ Skip analyzed (reuse confidence): {'ON' if SKIP_ANALYZED else 'OFF'}", flush=True)
    if FORCE_RETAG:
        print("⚠️  FORCE_RETAG is ENABLED. Existing tags will be overwritten.", flush=True)
    print("==================================================", flush=True)
    
    files = []
    for ext in ['*.mp3', '*.flac', '*.wav']:
        files.extend(glob.glob(os.path.join(MUSIC_DIR, '**', ext), recursive=True))
    
    total = len(files)
    print(f"Found {total} audio files in /music.", flush=True)
    
    for idx, f in enumerate(files, 1):
        print(f"\n[{idx}/{total}] ", end="", flush=True)
        process_file(f)
        
    print("\n🎉 Scan completed successfully! Shutting down container.", flush=True)

if __name__ == "__main__":
    main()
