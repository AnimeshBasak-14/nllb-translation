#!/usr/bin/env python3
"""
Apatani Speech & TTS Dataset Preprocessing Pipeline
===================================================
Processes raw Apatani speech audio (.wav) and sentences.txt transcripts:
- Validates audio integrity and computes duration and sample rate.
- Cleans and normalizes transcripts (strips quotation artifacts, formatting noise).
- Produces LJSpeech standard metadata (metadata.csv) for Coqui / ESPnet / VITS.
- Creates 85% train / 15% validation manifests for Hugging Face (SpeechT5 / Whisper).
- Exports comprehensive dataset statistics (character vocabulary, audio duration).
"""

import argparse
import json
import logging
import os
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import soundfile as sf

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Apatani TTS / ASR Dataset.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="Apatani_TTS_Database",
        help="Path to extracted Apatani_TTS_Database directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Apatani_TTS_Database",
        help="Directory to save preprocessed manifests and stats.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction of data reserved for validation (default: 0.15 = 15%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split.",
    )
    return parser.parse_args()


def clean_text(raw_text: str) -> str:
    """Cleans quotation artifacts, leading verse numbers, and excessive whitespace."""
    text = raw_text.strip()
    # Strip enclosing quotes and backslashes
    text = re.sub(r'^["\'\s]+|["\'\s]+$', '', text)
    # Remove leading numbering like "23hojalo" -> "hojalo" or stray digit headers
    text = re.sub(r'^\d+\s*', '', text)
    text = re.sub(r'\s+\d+(?=[a-zA-Z])', ' ', text)
    # Normalize multiple whitespace characters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_audio_info(wav_path: str) -> Dict[str, Any]:
    """Inspects audio headers for sample rate, duration, and channel count."""
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sr = wf.getframerate()
        frames = wf.getnframes()
        duration = round(frames / float(sr), 3)
    return {
        "duration": duration,
        "sample_rate": sr,
        "channels": n_channels,
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    wav_dir = data_dir / "wav"
    sentences_file = data_dir / "sentences.txt"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found at: {wav_dir}")
    if not sentences_file.exists():
        raise FileNotFoundError(f"Sentences file not found at: {sentences_file}")

    logger.info(f"Loading transcript mappings from '{sentences_file}'...")
    transcript_map = {}
    with open(sentences_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            parts = line.split("\t", 1)
            clip_id = parts[0].strip()
            text = parts[1].strip()
            transcript_map[clip_id] = text

    logger.info(f"Found {len(transcript_map)} transcripts in sentences.txt.")

    # Match audio files
    wav_files = sorted(wav_dir.glob("*.wav"))
    logger.info(f"Found {len(wav_files)} .wav audio files in '{wav_dir}'.")

    records = []
    missing_audio = 0
    missing_text = 0

    for wav_path in wav_files:
        clip_id = wav_path.stem
        if clip_id not in transcript_map:
            missing_text += 1
            logger.warning(f"Audio file '{wav_path.name}' has no matching transcript. Skipping.")
            continue

        raw_text = transcript_map[clip_id]
        normalized_text = clean_text(raw_text)

        if not normalized_text:
            continue

        try:
            info = get_audio_info(str(wav_path))
            records.append({
                "id": clip_id,
                "audio_path": str(wav_path.resolve()),
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "duration": info["duration"],
                "sample_rate": info["sample_rate"],
                "channels": info["channels"],
            })
        except Exception as e:
            logger.error(f"Failed to read audio '{wav_path}': {e}")

    df = pd.DataFrame(records)
    logger.info(f"Successfully processed and validated {len(df)} audio-transcript pairs.")

    # Dataset Statistics
    total_duration_sec = df["duration"].sum()
    total_duration_min = round(total_duration_sec / 60.0, 2)
    total_duration_hr = round(total_duration_sec / 3600.0, 2)
    avg_duration = round(df["duration"].mean(), 2)
    min_duration = round(df["duration"].min(), 2)
    max_duration = round(df["duration"].max(), 2)

    # Vocabulary extraction
    all_chars = sorted(list(set("".join(df["normalized_text"].tolist()))))

    # Split into train and validation
    train_df = df.sample(frac=1.0 - args.val_ratio, random_state=args.seed).sort_index()
    val_df = df.drop(train_df.index).sort_index()

    stats = {
        "total_clips": len(df),
        "total_duration_minutes": total_duration_min,
        "total_duration_hours": total_duration_hr,
        "avg_clip_duration_sec": avg_duration,
        "min_duration_sec": min_duration,
        "max_duration_sec": max_duration,
        "sample_rate_hz": int(df["sample_rate"].iloc[0]),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "vocab_char_count": len(all_chars),
        "vocabulary_chars": all_chars,
    }

    # Save outputs
    # 1. Complete manifest CSV
    manifest_csv = out_dir / "apatani_manifest.csv"
    df.to_csv(manifest_csv, index=False)
    logger.info(f"Saved complete manifest to: {manifest_csv}")

    # 2. LJSpeech standard format (id|raw_text|normalized_text)
    ljspeech_dir = out_dir / "ljspeech"
    ljspeech_dir.mkdir(parents=True, exist_ok=True)
    ljspeech_metadata = ljspeech_dir / "metadata.csv"
    with open(ljspeech_metadata, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"{row['id']}|{row['raw_text']}|{row['normalized_text']}\n")
    logger.info(f"Saved standard LJSpeech format metadata to: {ljspeech_metadata}")

    # 3. Train and Validation JSON manifests
    train_manifest = out_dir / "train_manifest.json"
    val_manifest = out_dir / "val_manifest.json"
    train_df.to_json(train_manifest, orient="records", indent=2)
    val_df.to_json(val_manifest, orient="records", indent=2)
    logger.info(f"Saved Train ({len(train_df)} samples) & Val ({len(val_df)} samples) manifests.")

    # 4. Summary statistics JSON
    stats_json = out_dir / "dataset_stats.json"
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved dataset statistics to: {stats_json}")

    print("\n========================================================")
    print("           Apatani Speech Dataset Profile               ")
    print("========================================================")
    print(f"Total Paired Samples:     {len(df)}")
    print(f"Total Audio Duration:     {total_duration_min} minutes ({total_duration_hr} hours)")
    print(f"Average Clip Duration:    {avg_duration} seconds (Range: {min_duration}s - {max_duration}s)")
    print(f"Sampling Frequency:       {stats['sample_rate_hz']} Hz (Mono)")
    print(f"Train / Val Split:        {len(train_df)} train / {len(val_df)} val (15%)")
    print(f"Vocabulary Size:          {len(all_chars)} unique characters")
    print("========================================================\n")


if __name__ == "__main__":
    main()
