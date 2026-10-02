#!/usr/bin/env python3
"""
Unified Multilingual Translation & Speech Demo CLI
===================================================
Provides unified command-line and interactive demonstrations for:
1. Text Translation (English -> Apatani / Nyishi / Adi / Galo / Tagin)
2. Speech Recognition (Apatani Speech .wav -> Transcribed Text via Whisper)
3. End-to-End Pipeline Evaluation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import soundfile as sf
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_audio_resampled(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Loads and resamples audio to 16 kHz."""
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        gcd = np.gcd(int(sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        data = scipy.signal.resample_poly(data, up, down)
    return data.astype(np.float32)


def run_translation(
    text: str,
    model_dir: str = "./best_apatani_translation",
    src_lang: str = "eng_Latn",
    tgt_lang: str = "hin_Deva",
) -> str:
    """Translates English text into target language."""
    if not os.path.exists(model_dir):
        # Fallback to general best_model if specific model not yet finished
        if os.path.exists("./best_model"):
            model_dir = "./best_model"
        else:
            raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)
    forced_bos_token_id = model.config.forced_bos_token_id
    if forced_bos_token_id is None and hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(tgt_lang)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=128,
            num_beams=4,
        )

    translated = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return translated


def run_transcription(
    audio_path: str,
    asr_model_dir: str = "./best_apatani_asr",
) -> str:
    """Transcribes Apatani audio into text using fine-tuned Whisper."""
    if not os.path.exists(asr_model_dir):
        raise FileNotFoundError(f"ASR model directory '{asr_model_dir}' not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(asr_model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(asr_model_dir).to(device)
    model.eval()
    model.generation_config.forced_decoder_ids = None

    waveform = load_audio_resampled(audio_path, target_sr=16000)
    input_features = processor.feature_extractor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return transcription


def main():
    parser = argparse.ArgumentParser(description="Multilingual Translation & Speech Demo")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["translate", "transcribe", "demo"],
        help="Mode: 'translate', 'transcribe', or 'demo'.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Noah, Shem, Ham, and Japheth.",
        help="Input text for translation.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="Apatani_TTS_Database/wav/APT-0006.wav",
        help="Audio file path for transcription.",
    )
    parser.add_argument(
        "--translation_model",
        type=str,
        default="./best_apatani_translation",
        help="Path to translation model.",
    )
    parser.add_argument(
        "--asr_model",
        type=str,
        default="./best_apatani_asr",
        help="Path to fine-tuned Whisper model.",
    )

    args = parser.parse_args()

    print("\n========================================================")
    print("      Arunachal Multilingual & Speech AI Demo           ")
    print("========================================================")

    if args.mode in ["translate", "demo"]:
        print(f"\n[1] Text Translation Mode:")
        print(f"    Source (English): {args.text}")
        try:
            translation = run_translation(args.text, model_dir=args.translation_model)
            print(f"    Translation:      {translation}")
        except Exception as e:
            print(f"    Translation notice: {e}")

    if args.mode in ["transcribe", "demo"]:
        print(f"\n[2] Speech-to-Text (ASR) Mode:")
        print(f"    Audio Input:      {args.audio}")
        if os.path.exists(args.audio):
            try:
                transcript = run_transcription(args.audio, asr_model_dir=args.asr_model)
                print(f"    Transcription:    {transcript}")
            except Exception as e:
                print(f"    ASR error: {e}")
        else:
            print(f"    Audio file '{args.audio}' not found.")

    print("\n========================================================\n")


if __name__ == "__main__":
    main()
