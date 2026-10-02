#!/usr/bin/env python3
"""
Official Hackathon Submission & Audio Generator
===============================================
Fulfills all requirements from the Hackathon Arunachal guidelines:
1. Machine Translation (MT):
   - Nyishi <-> English Bidirectional generation
   - Accepts released test sentences and exports formatted submission TSV.
2. Text-to-Speech (TTS):
   - Apatani Neural TTS generation using fine-tuned SpeechT5 + HiFi-GAN.
   - Accepts released test sentences and produces normalized 16 kHz .wav files.
   - Automatically packages into a ready-to-submit ZIP archive.
3. Audio Quality Evaluation:
   - Computes Mel-Cepstral Distortion (MCD in dB) comparing synthesized vs ground-truth speech.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.fftpack
import scipy.signal
import soundfile as sf
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_mcep(
    waveform: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    n_mcep: int = 13,
) -> np.ndarray:
    """Computes Mel-Cepstral coefficients from audio waveform."""
    if len(waveform) < win_length:
        waveform = np.pad(waveform, (0, win_length - len(waveform)))

    # Framing with Hann window
    num_frames = 1 + int((len(waveform) - win_length) / hop_length)
    window = np.hanning(win_length)
    frames = np.zeros((num_frames, win_length))
    for t in range(num_frames):
        start = t * hop_length
        frames[t] = waveform[start : start + win_length] * window

    # FFT and magnitude spectrum
    spec = np.abs(np.fft.rfft(frames, n=n_fft))
    spec = np.maximum(spec, 1e-10)

    # Simple Mel filterbank
    n_freqs = spec.shape[1]
    n_mels = 40
    mel_fb = np.zeros((n_mels, n_freqs))
    mel_pts = np.linspace(0, 2595 * np.log10(1 + (sr / 2) / 700), n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)

    for m in range(1, n_mels + 1):
        f_m_minus = bin_pts[m - 1]
        f_m = bin_pts[m]
        f_m_plus = bin_pts[m + 1]
        if f_m > f_m_minus:
            mel_fb[m - 1, f_m_minus:f_m] = (np.arange(f_m_minus, f_m) - f_m_minus) / (f_m - f_m_minus)
        if f_m_plus > f_m:
            mel_fb[m - 1, f_m:f_m_plus] = (f_m_plus - np.arange(f_m, f_m_plus)) / (f_m_plus - f_m)

    mel_spec = np.dot(spec, mel_fb.T)
    log_mel = np.log(np.maximum(mel_spec, 1e-10))

    # DCT to obtain cepstral coefficients (drop c0 energy)
    cep = scipy.fftpack.dct(log_mel, type=2, axis=-1, norm="ortho")
    return cep[:, 1 : n_mcep + 1]


def compute_mcd_distance(ref_wav_path: str, synth_wav_path: str) -> float:
    """Computes Mel Cepstral Distortion (MCD in dB) using Dynamic Time Warping."""
    ref_audio, sr_ref = sf.read(ref_wav_path)
    if ref_audio.ndim > 1:
        ref_audio = np.mean(ref_audio, axis=1)
    if sr_ref != 16000:
        gcd = math.gcd(int(sr_ref), 16000)
        ref_audio = scipy.signal.resample_poly(ref_audio, 16000 // gcd, sr_ref // gcd)

    synth_audio, sr_synth = sf.read(synth_wav_path)
    if synth_audio.ndim > 1:
        synth_audio = np.mean(synth_audio, axis=1)
    if sr_synth != 16000:
        gcd = math.gcd(int(sr_synth), 16000)
        synth_audio = scipy.signal.resample_poly(synth_audio, 16000 // gcd, sr_synth // gcd)

    c_ref = compute_mcep(ref_audio.astype(np.float32))
    c_syn = compute_mcep(synth_audio.astype(np.float32))

    # Dynamic Time Warping (DTW) distance matrix
    N, D = c_ref.shape
    M, _ = c_syn.shape

    dist = np.zeros((N, M))
    for i in range(N):
        diff = c_syn - c_ref[i]
        dist[i] = np.sqrt(np.sum(diff ** 2, axis=1))

    # Cost matrix
    cost = np.full((N + 1, M + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost[i, j] = dist[i - 1, j - 1] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path_len = max(N, M)
    mcd = (10.0 * math.sqrt(2.0) / math.log(10.0)) * (cost[N, M] / path_len)
    return round(float(mcd), 4)


def generate_tts_submission_wavs(
    input_file: str,
    output_dir: str = "submission_tts_wavs",
    model_dir: str = "./best_apatani_tts",
    vocoder_name: str = "microsoft/speecht5_hifigan",
) -> str:
    """Reads test sentences and synthesizes submission .wav files."""
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading TTS model from: {model_dir}")
    processor = SpeechT5Processor.from_pretrained(model_dir)
    model = SpeechT5ForTextToSpeech.from_pretrained(model_dir).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_name).to(device)
    model.eval()
    vocoder.eval()

    spk_path = os.path.join(model_dir, "speaker_embedding.pt")
    speaker_embedding = torch.load(spk_path, map_location=device, weights_only=True) if os.path.exists(spk_path) else torch.randn(1, 512, device=device) * 0.05

    # Read sentences
    sentences = []
    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                sentences.append((item.get("id", f"sample_{len(sentences)+1:03d}"), item.get("normalized_text", item.get("text", ""))))
    elif input_file.endswith(".tsv") or input_file.endswith(".csv"):
        sep = "\t" if input_file.endswith(".tsv") else ","
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=sep)
            for row in reader:
                if len(row) >= 2:
                    sentences.append((row[0].strip(), row[1].strip()))
                elif len(row) == 1:
                    sentences.append((f"sample_{len(sentences)+1:03d}", row[0].strip()))
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append((f"sample_{len(sentences)+1:03d}", line))

    logger.info(f"Synthesizing {len(sentences)} test sentences into '{output_dir}'...")

    generated_paths = []
    for s_id, text in sentences:
        clean_text = text.strip().strip('"').strip("'")
        if not clean_text:
            continue

        inputs = processor(text=clean_text, return_tensors="pt").to(device)
        with torch.no_grad():
            speech = model.generate_speech(inputs["input_ids"], speaker_embedding, vocoder=vocoder)

        audio_np = speech.cpu().numpy()
        # Normalize and trim dead silence
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.95

        wav_path = os.path.join(output_dir, f"{s_id}.wav")
        sf.write(wav_path, audio_np, samplerate=16000)
        generated_paths.append(wav_path)

    # Package into a submission ZIP
    zip_path = f"{output_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in generated_paths:
            zipf.write(file, arcname=os.path.basename(file))

    logger.info(f"Successfully generated {len(generated_paths)} .wav files.")
    logger.info(f"Submission ZIP archive created at: {zip_path}")
    return zip_path


def generate_mt_submission(
    input_file: str,
    output_file: str = "translations_submission.tsv",
    model_dir: str = "./best_bidirectional_model",
    direction: str = "en2nyishi",
) -> str:
    """Translates test sentences and exports formatted submission TSV."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if direction == "en2nyishi":
        src_lang = "eng_Latn"
        tgt_lang = "hin_Deva"
    else:
        src_lang = "hin_Deva"
        tgt_lang = "eng_Latn"

    logger.info(f"Loading MT model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang, tgt_lang=tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    sentences = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    logger.info(f"Translating {len(sentences)} sentences ({direction})...")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    results = []
    batch_size = 16
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for src, pred in zip(batch, decoded):
            results.append((src, pred.strip()))

    with open(output_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source", "translation"])
        for src, pred in results:
            writer.writerow([src, pred])

    logger.info(f"Translations saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Official Hackathon Submission Generator")
    parser.add_argument("--mode", type=str, required=True, choices=["tts", "mt", "mcd"], help="Submission mode.")
    parser.add_argument("--input_file", type=str, default="Apatani_TTS_Database/test_manifest.json", help="Test sentences input file.")
    parser.add_argument("--output_dir", type=str, default="submission_tts_wavs", help="Output directory for .wav files.")
    parser.add_argument("--output_file", type=str, default="translations_submission.tsv", help="Output file for MT translations.")
    parser.add_argument("--ref_wav", type=str, default=None, help="Reference WAV for MCD calculation.")
    parser.add_argument("--synth_wav", type=str, default=None, help="Synthesized WAV for MCD calculation.")
    parser.add_argument("--direction", type=str, default="en2nyishi", choices=["en2nyishi", "nyishi2en"], help="MT translation direction.")

    args = parser.parse_args()

    if args.mode == "tts":
        generate_tts_submission_wavs(args.input_file, output_dir=args.output_dir)
    elif args.mode == "mt":
        generate_mt_submission(args.input_file, output_file=args.output_file, direction=args.direction)
    elif args.mode == "mcd":
        if not args.ref_wav or not args.synth_wav:
            print("Error: --ref_wav and --synth_wav are required for MCD calculation.")
            sys.exit(1)
        score = compute_mcd_distance(args.ref_wav, args.synth_wav)
        print(f"Mel-Cepstral Distortion (MCD): {score:.4f} dB (Lower is better)")


if __name__ == "__main__":
    main()
