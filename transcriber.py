"""
Módulo responsável por transcrever áudio com WhisperX + diarização de falantes.
"""
import os
from pathlib import Path
from typing import Callable

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

from ffmpeg_setup import ensure_ffmpeg_available

load_dotenv()

# ---------- Configurações globais ----------
DEVICE = "cpu"
COMPUTE_TYPE = "int8"           # mais rápido na CPU
WHISPER_MODEL = "large-v3"      # alta precisão (lento na CPU!)
BATCH_SIZE = 4                  # baixo para economizar RAM


class TranscriptionCancelled(Exception):
    """Sinaliza o cancelamento cooperativo do processamento."""


def _fmt_time(seconds: float) -> str:
    """Converte segundos para HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_and_diarize(
    audio_path: str,
    hf_token: str,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    language: str | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    cancellation_callback: Callable[[], bool] | None = None,
):
    """
    Transcreve o áudio, faz alinhamento e diarização.

    Retorna um dicionário com:
      - 'segments': lista de segmentos com 'speaker', 'start', 'end', 'text'
      - 'language': idioma detectado
      - 'full_text': texto completo com falantes formatados
      - 'speakers': lista dos IDs de falantes únicos detectados
    """
    if not hf_token:
        raise ValueError("Token do Hugging Face é obrigatório para a diarização.")

    def check_cancelled() -> None:
        if cancellation_callback and cancellation_callback():
            raise TranscriptionCancelled("Transcrição cancelada pelo usuário.")

    audio_path = str(Path(audio_path).resolve())

    def log(msg: str, progress: float):
        if progress_callback:
            progress_callback(msg, progress)
        else:
            print(msg)

    # ---------- 1. Carregar áudio ----------
    check_cancelled()
    ensure_ffmpeg_available()
    log("🎧 Carregando áudio...", 0.05)
    audio = whisperx.load_audio(audio_path)
    check_cancelled()
    log("✅ Áudio carregado.", 0.1)

    # ---------- 2. Transcrição ----------
    check_cancelled()
    log(
        f"📝 Transcrevendo com modelo '{WHISPER_MODEL}' (isso pode demorar um pouco)...",
        0.12,
    )
    model = whisperx.load_model(
        WHISPER_MODEL,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language=language,
    )
    check_cancelled()
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=language)
    check_cancelled()
    detected_lang = result.get("language", language or "?")
    log(f"✅ Transcrição concluída. Idioma detectado: {detected_lang}", 0.52)

    # ---------- 3. Alinhamento (timestamps por palavra) ----------
    check_cancelled()
    log("🎯 Alinhando marcas de tempo...", 0.55)
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=detected_lang, device=DEVICE
        )
        check_cancelled()
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False,
        )
        check_cancelled()
        log("✅ Alinhamento concluído.", 0.7)
    except TranscriptionCancelled:
        raise
    except Exception as e:
        log(
            f"⚠️ Alinhamento falhou ({e}). Prosseguindo sem alinhamento.",
            0.7,
        )

    # ---------- 4. Diarização ----------
    check_cancelled()
    log("🗣️ Identificando falantes (diarização)...", 0.72)
    diarize_model = DiarizationPipeline(
        token=hf_token,
        device=DEVICE,
    )
    diarize_segments = diarize_model(
        audio,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    check_cancelled()
    log("✅ Diarização concluída.", 0.93)

    # ---------- 5. Mesclar transcrição + diarização ----------
    check_cancelled()
    log("🔗 Mesclando transcrição com falantes...", 0.95)
    result = whisperx.assign_word_speakers(diarize_segments, result)
    check_cancelled()
    log("✅ Falantes associados à transcrição.", 0.98)

    # ---------- 6. Formatar saída ----------
    segments = []
    speakers_set = set()
    for seg in result["segments"]:
        check_cancelled()
        spk = seg.get("speaker", "DESCONHECIDO")
        speakers_set.add(spk)
        segments.append({
            "speaker": spk,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "start_hms": _fmt_time(seg["start"]),
            "end_hms": _fmt_time(seg["end"]),
            "text": seg["text"].strip(),
        })

    speakers = sorted(speakers_set)

    # Texto formatado por falante
    lines = []
    current_speaker = None
    buffer = []
    for seg in segments:
        if seg["speaker"] != current_speaker:
            if buffer:
                lines.append(f"[{current_speaker}] {' '.join(buffer).strip()}")
                buffer = []
            current_speaker = seg["speaker"]
        buffer.append(seg["text"])
    if buffer:
        lines.append(f"[{current_speaker}] {' '.join(buffer).strip()}")

    check_cancelled()
    full_text = "\n\n".join(lines)

    log("🎉 Processamento concluído!", 1.0)
    return {
        "segments": segments,
        "language": detected_lang,
        "full_text": full_text,
        "speakers": speakers,
    }