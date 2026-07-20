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

# Set this to True to overwrite previously written tags with fresh ML predictions.
# Set it to False once your library is clean to make future scans ultra-fast.
FORCE_RETAG = True 
# ==============================================================================

# Register custom tag mapping for EasyID3
EasyID3.RegisterTXXXKey("instrumental", "INSTRUMENTAL")

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

def tag_file(filepath, is_instrumental, ext):
    """
    Writes the instrumental tag (1 for true, 0 for false) to the audio file metadata.
    """
    tag_val = "1" if is_instrumental else "0"
    try:
        if ext == '.flac':
            audio = FLAC(filepath)
            audio['INSTRUMENTAL'] = tag_val
            audio.save()
        elif ext == '.mp3':
            try:
                audio = EasyID3(filepath)
            except ID3NoHeaderError:
                audio = EasyID3()
            audio['instrumental'] = tag_val
            audio.save()
        elif ext == '.wav':
            audio = WAVE(filepath)
            if audio.tags is None:
                audio.add_tags()
            audio.tags.delall("TXXX:INSTRUMENTAL")
            audio.tags.add(TXXX(encoding=3, desc="INSTRUMENTAL", text=[tag_val]))
            audio.save()
        return True
    except Exception as e:
        print(f"\n❌ Error tagging {filepath}: {e}", flush=True)
        return False

def analyze_track_ml(filepath):
    """
    Loads 45 seconds of audio, extracts VGGish embeddings, and classifies 
    vocal vs instrumental using TensorFlow.
    """
    try:
        loader = es.EasyLoader(filename=filepath, sampleRate=16000, startTime=40, endTime=85)
        audio = loader()
        
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
        vocal_prob = np.mean(predictions[:, 1])
        
        is_instrumental = instrumental_prob > vocal_prob
        return is_instrumental, instrumental_prob

    except Exception as e:
        print(f"\n❌ ML Analysis failed for {os.path.basename(filepath)}: {e}", flush=True)
        return None, None

def process_file(filepath):
    _, ext = os.path.splitext(filepath.lower())
    if ext not in ['.mp3', '.flac', '.wav']:
        return

    filename = os.path.basename(filepath)

    # Check for an existing tag value first
    existing_tag = get_existing_tag(filepath, ext)
    
    if existing_tag is not None and not FORCE_RETAG:
        print(f"⏭️  Already Tagged as {existing_tag}: {filename}", flush=True)
        return

    # Run ML analysis
    prefix = f"🔄 Re-analyzing (Was {existing_tag}): " if existing_tag else "🔍 Analyzing: "
    print(f"{prefix}{filename}...", end="", flush=True)
    
    is_instrumental, confidence = analyze_track_ml(filepath)
    
    if is_instrumental is not None:
        if tag_file(filepath, is_instrumental, ext):
            status = "INSTRUMENTAL (1)" if is_instrumental else "VOCAL (0)"
            print(f" Done! -> Tagged as {status} (Conf: {confidence:.1%})", flush=True)
        else:
            print(" Failed to tag.", flush=True)
    else:
        print(" Failed to analyze.", flush=True)

def main():
    print("==================================================", flush=True)
    print("🚀 Starting Instrumental Tagging Scan...", flush=True)
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
