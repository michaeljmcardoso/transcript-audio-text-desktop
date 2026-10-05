#!/usr/bin/env sh
set -eu

PYTHON=python3
if [ -x .venv/bin/python3 ]; then
  PYTHON=.venv/bin/python3
fi

"$PYTHON" -m PyInstaller \
  --noconfirm \
  --clean \
  --onedir \
  --windowed \
  --name TranscricaoDesktop \
  --collect-all whisperx \
  --collect-all pyannote.audio \
  --copy-metadata torchcodec \
  --collect-binaries imageio_ffmpeg \
  --exclude-module triton \
  --hidden-import transcriber \
  main.py
