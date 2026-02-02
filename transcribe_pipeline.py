#!/usr/bin/env python3
"""
WhisperX Transcription and Diarization Pipeline for BigUppetite Workflow Automation (to be integrated into the main ingestion pipeline)
Processes audio and video files from the input folder and outputs transcriptions with speaker diarization.
"""

import os
import sys
import time
from datetime import timedelta # Used for formatting the duration nicely
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable torch.compile for faster imports
os.environ['TORCH_COMPILE_DEBUG'] = '0'
os.environ['TORCH_COMPILE'] = '0'
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN', '')

import json
from html import escape
from whisperx.utils import SubtitlesWriter
import logging
from pathlib import Path
from datetime import datetime
import whisperx
import librosa
import soundfile as sf
import subprocess
import torch
import omegaconf

# Monkey-patch torch.load to disable weights_only for compatibility
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Patched torch.load that disables weights_only for PyTorch 2.6+ compatibility."""
    kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
DEVICE = "cpu"  # Use "cpu" if CUDA not available, "cuda" for GPU
BATCH_SIZE = 8  # Reduced for CPU
COMPUTE_TYPE = "int8"  # Use "int8" for lower memory usage on CPU
MODEL_NAME = "large-v3" # Defined here so we can log it in metadata later

# Create directories if they don't exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DIR = Path("models")

def initialize_models():
    """Initialize WhisperX model and diarization pipeline."""
    logger.info(f"Initializing WhisperX model: {MODEL_NAME}...")
    
    # Load Whisper model
    model = whisperx.load_model(
        MODEL_NAME, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE
        # download_root=str(MODEL_DIR) 
    )    
    
    # Load diarization pipeline
    logger.info("Loading diarization model...")
    hf_token = HUGGINGFACE_TOKEN

    
    if not hf_token:
        logger.warning("=" * 80)
        logger.warning("HUGGINGFACE_TOKEN not set - speaker diarization will be DISABLED")
        logger.warning("=" * 80)
        diarize_model = None
    else:
        try:
            diarize_model = whisperx.diarize.DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.0",
                use_auth_token=hf_token,
                device=DEVICE
            )
            logger.info("Diarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load diarization model: {str(e)}")
            diarize_model = None
    
    logger.info("Models loaded successfully")
    return model, diarize_model

def convert_mp3_to_wav(mp3_path, wav_path):
    """Convert MP3 to WAV format if needed."""
    if mp3_path.suffix.lower() == ".mp3":
        logger.info(f"Converting {mp3_path.name} to WAV...")
        try:
            # Try using ffmpeg via subprocess first
            subprocess.run([
                "ffmpeg", "-i", str(mp3_path), "-acodec", "pcm_s16le", 
                "-ar", "16000", str(wav_path), "-y"
            ], capture_output=True, check=True)
            logger.info("Conversion successful with ffmpeg")
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fallback: use pydub-like approach with scipy if available
                logger.info("FFmpeg not found, attempting librosa conversion...")
                y, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
                sf.write(str(wav_path), y, sr)
                logger.info("Conversion successful with librosa")
            except Exception as e:
                logger.warning(f"Librosa conversion failed: {e}")
                logger.info("Attempting to use input file as-is...")
                return mp3_path
        return wav_path
    return mp3_path

def transcribe_file(file_path, model, diarize_model):
    """Transcribe, Align, and Diarize a single audio file."""
    # --- Start Timer ---
    start_time_proc = time.time()
    
    try:
        logger.info(f"Processing: {file_path.name}")
        
        # Convert MP3 to WAV if necessary
        wav_path = OUTPUT_DIR / file_path.stem / "audio.wav"
        wav_path.parent.mkdir(exist_ok=True)
        audio_file = convert_mp3_to_wav(file_path, wav_path)
        
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found after conversion: {audio_file}")
        
        logger.info(f"Audio file ready: {audio_file}")
        
        # 1. Transcribe
        logger.info(f"Starting transcription of {file_path.name}...")
        result = model.transcribe(str(audio_file), batch_size=BATCH_SIZE)
        
        if result is None:
            raise ValueError("Transcription returned None")
            
        # 2. Align (CRITICAL STEP FOR WORD SCORES)
        logger.info("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=DEVICE
        )
        
        logger.info("Aligning segments...")
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            str(audio_file), 
            device=DEVICE, 
            return_char_alignments=False
        )
        
        # Free up memory
        import gc
        del model_a
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Alignment complete! Found {len(result['segments'])} segments")
        
        # 3. Diarize if model available
        if diarize_model is not None:
            logger.info(f"Running speaker diarization on {file_path.name}...")
            diarize_segments = diarize_model(str(audio_file))
            result = whisperx.assign_speakers(diarize_segments, result)
        else:
            diarize_segments = None
        
        # --- Stop Timer & Add Metadata ---
        end_time_proc = time.time()
        duration_seconds = end_time_proc - start_time_proc
        
        # Add metadata block to the result dictionary
        result["metadata"] = {
            "model_name": MODEL_NAME,
            "transcription_date": datetime.now().isoformat(),
            "processing_duration_seconds": round(duration_seconds, 2),
            "processing_duration_formatted": str(timedelta(seconds=int(duration_seconds)))
        }
        
        logger.info(f"✓ Completed: {file_path.name} (Time: {result['metadata']['processing_duration_formatted']})")
        return result, diarize_segments
        
    except Exception as e:
        logger.error(f"ERROR processing {file_path.name}: {str(e)}", exc_info=True)
        return None, None
    

def save_results(file_path, result, diarize_segments):
    """Save transcription results in multiple formats."""
    if result is None:
        return
    
    output_base = OUTPUT_DIR / file_path.stem
    output_base.mkdir(exist_ok=True)
    
    try:
        # Save as JSON
        json_path = output_base / "transcript.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON: {json_path}")
        
        # Save as formatted text with speakers
        txt_path = output_base / "transcript.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"File: {file_path.name}\n")
            f.write(f"Processed: {datetime.now().isoformat()}\n")
            
            # Write metadata if available
            if "metadata" in result:
                meta = result["metadata"]
                f.write(f"Model: {meta.get('model_name', 'Unknown')}\n")
                f.write(f"Duration: {meta.get('processing_duration_formatted', 'Unknown')}\n")
                
            f.write("=" * 80 + "\n\n")
            
            if "segments" in result:
                for segment in result["segments"]:
                    # Get speaker if diarization was performed
                    speaker = segment.get("speaker", "Unknown")
                    start_time = format_timestamp(segment["start"])
                    end_time = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    
                    f.write(f"[{start_time} - {end_time}] Speaker {speaker}:\n")
                    f.write(f"{text}\n\n")
        
        logger.info(f"Saved TXT: {txt_path}")

        # --- Colored outputs: HTML and colored SRT ---
        try:
            # Colored HTML
            html_path = output_base / "transcript_colored.html"
            write_colored_html(result, html_path)
            logger.info(f"Saved colored HTML: {html_path}")

            # Colored SRT
            srt_path = output_base / "transcript_colored.srt"
            with open(srt_path, 'w', encoding='utf-8') as srt_f:
                writer = ColoredSRTWriter()
                writer.write_result(result, srt_f)
            logger.info(f"Saved colored SRT: {srt_path}")
        except Exception as e:
            logger.error(f"Error saving colored outputs: {e}")

        # Save diarization segments if available
        if diarize_segments is not None:
            diarize_path = output_base / "diarization.json"
            with open(diarize_path, 'w', encoding='utf-8') as f:
                json.dump(str(diarize_segments), f, indent=2)
            logger.info(f"Saved diarization: {diarize_path}")
        
    except Exception as e:
        logger.error(f"Error saving results for {file_path.name}: {str(e)}")

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _get_color(score):
    """Return Red for low confidence, Green otherwise."""
    try:
        score = float(score)
    except Exception:
        score = 1.0
        
    if score < 0.6:
        return "#FF0000" # Red
    else:
        return "#008000" # Green

def _speaker_color(speaker_label):
    """Deterministically pick a color for a speaker label."""
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    try:
        idx = abs(hash(str(speaker_label))) % len(palette)
        return palette[idx]
    except Exception:
        return "#000000"


class ColoredSRTWriter(SubtitlesWriter):
    extension = "srt"
    always_include_hours = True
    decimal_marker = ","

    def write_result(self, result, file, options=None):
        """Write colored SRT based on word confidence scores"""
        for i, segment in enumerate(result.get("segments", []), 1):
            if "words" in segment:
                # Build colored text from words
                colored_words = []
                for word_data in segment.get("words", []):
                    word = word_data.get("word", "").strip()
                    score = word_data.get("score", 1.0)
                    color = _get_color(score)
                    # escape words for safety
                    colored_words.append(f'<font color="{color}">{escape(word)}</font>')
                
                text = " ".join(colored_words)
            else:
                text = escape(segment.get("text", "").strip())

            # Prefix speaker label if present
            speaker = segment.get("speaker")
            if speaker is not None:
                speaker_label = f"Speaker {speaker}: "
                text = f"<b>{escape(speaker_label)}</b>" + text

            # Write SRT format
            print(i, file=file)
            start = self.format_timestamp(segment.get("start", 0.0))
            end = self.format_timestamp(segment.get("end", 0.0))
            print(f"{start} --> {end}", file=file)
            print(text, file=file)
            print("", file=file)


def write_colored_html(result, html_path):
    """Create an HTML transcript that highlights words by confidence and marks speaker colors."""
    segments = result.get("segments", [])

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n")
        f.write("<meta charset=\"utf-8\">\n<title>Colored Transcript</title>\n")
        f.write("<style>body{font-family:Inter, Arial, Helvetica, sans-serif;line-height:1.5;padding:20px;}\n")
        f.write(".segment{margin:12px 0;padding:8px;border-radius:6px;}\n")
        f.write(".speaker-label{font-weight:700;padding:4px 8px;border-radius:4px;color:#fff;display:inline-block;margin-right:8px;}\n")
        f.write(".word{padding:2px 2px;border-radius:3px;margin-right:2px;display:inline-block;}\n")
        f.write("</style>\n</head>\n<body>\n")

        f.write(f"<h2>Colored Transcript</h2>\n")
        
        # Display metadata in HTML if available
        if "metadata" in result:
             meta = result["metadata"]
             f.write(f"<p>Model: {meta.get('model_name', 'Unknown')} | Time: {meta.get('processing_duration_formatted', 'Unknown')}</p>")
        
        f.write(f"<p>Generated: {datetime.now().isoformat()}</p>\n<hr/>\n")

        for segment in segments:
            start = format_timestamp(segment.get("start", 0.0))
            end = format_timestamp(segment.get("end", 0.0))
            speaker = segment.get("speaker")
            speaker_color = _speaker_color(speaker) if speaker is not None else "#444"

            f.write(f"<div class=\"segment\">\n")
            f.write(f"<div style=\"margin-bottom:6px;\">\n")
            if speaker is not None:
                f.write(f"<span class=\"speaker-label\" style=\"background:{speaker_color}\">{escape(str(speaker))}</span>")
            f.write(f"<small>[{start} - {end}]</small>\n")
            f.write("</div>\n")

            # Words with per-word coloring
            if "words" in segment:
                for w in segment.get("words", []):
                    word = escape(w.get("word", "").strip())
                    score = w.get("score", 1.0)
                    color = _get_color(score)
                    f.write(f"<span class=\"word\" style=\"color:{color}\">{word}</span>")
            else:
                text = escape(segment.get("text", "").strip())
                f.write(f"<span>{text}</span>")

            f.write("\n</div>\n")

        f.write("</body>\n</html>\n")

def process_input_folder(model, diarize_model):
    """Process all MP3 files in the input folder."""
    mp3_files = list(INPUT_DIR.glob("*.mp3"))
    
    if not mp3_files:
        logger.warning("No MP3 files found in input folder")
        return
    
    logger.info(f"Found {len(mp3_files)} MP3 file(s) to process")
    
    for mp3_file in mp3_files:
        result, diarize_segments = transcribe_file(mp3_file, model, diarize_model)
        if result:
            save_results(mp3_file, result, diarize_segments)
    
    logger.info("All files processed!")

def main():
    """Main entry point."""
    logger.info("Starting WhisperX Transcription Pipeline")
    logger.info(f"Input directory: {INPUT_DIR.absolute()}")
    logger.info(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    try:
        model, diarize_model = initialize_models()
        process_input_folder(model, diarize_model)
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()