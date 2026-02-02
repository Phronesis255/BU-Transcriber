"""
Quick start guide for the WhisperX Transcription Pipeline
"""

# STEP 1: Get HuggingFace Token
# 1. Go to https://huggingface.co/settings/tokens
# 2. Create a new token (can be read-only)
# 3. Copy the token

# STEP 2: Set up environment
# On Windows PowerShell:
# $env:HUGGINGFACE_TOKEN = "your_token_here"

# STEP 3: Accept model agreements
# 1. Visit https://huggingface.co/pyannote/speaker-diarization-3.0
# 2. Click "Accept" on the model card
# 3. Visit https://huggingface.co/openai/whisper-base
# 4. Click "Accept" on the model card

# STEP 4: Run the pipeline

# Option A: One-time processing
# python transcribe_pipeline.py

# Option B: Watch mode (processes files as they're added)
# python watch_mode.py

# STEP 5: Check results in output/ folder
