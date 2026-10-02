#!/usr/bin/env python3
"""
Apatani Text-to-Speech (TTS) Fine-Tuning Pipeline
=================================================
Fine-tunes a neural Text-to-Speech model (Microsoft SpeechT5 + HiFi-GAN Vocoder)
on the indigenous Apatani Speech Database (251 utterances):
- Converts Apatani text into natural spoken speech (.wav).
- Trains on log-mel spectrogram acoustic features with attention alignment.
- Handles audio resampling (22.05 kHz -> 16 kHz) and variable-length padding.
- Saves best TTS model checkpoint, processor, and speaker embeddings for inference.
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    get_cosine_schedule_with_warmup,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SpeechT5 TTS on Apatani dataset.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/speecht5_tts",
        help="Base SpeechT5 model path or Hugging Face hub id.",
    )
    parser.add_argument(
        "--vocoder_name_or_path",
        type=str,
        default="microsoft/speecht5_hifigan",
        help="HiFi-GAN vocoder model path or Hugging Face hub id.",
    )
    parser.add_argument(
        "--train_manifest",
        type=str,
        default="Apatani_TTS_Database/train_manifest.json",
        help="Path to training manifest JSON.",
    )
    parser.add_argument(
        "--val_manifest",
        type=str,
        default="Apatani_TTS_Database/val_manifest.json",
        help="Path to validation manifest JSON.",
    )
    parser.add_argument(
        "--test_manifest",
        type=str,
        default="Apatani_TTS_Database/test_manifest.json",
        help="Path to test manifest JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./best_apatani_tts",
        help="Directory to save the fine-tuned TTS model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=300,
        help="Maximum text character length to consider.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 mixed precision training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


class ApataniTTSDataset(Dataset):
    """PyTorch Dataset for paired Apatani text and speech audio."""

    def __init__(
        self,
        manifest_path: str,
        processor: SpeechT5Processor,
        target_sr: int = 16000,
        max_duration: float = 25.0,
    ):
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

        self.processor = processor
        self.target_sr = target_sr
        # Filter out extreme outliers
        self.items = [
            item for item in self.items
            if item.get("duration", 0) <= max_duration and len(item.get("normalized_text", "")) > 2
        ]
        logger.info(f"Loaded {len(self.items)} valid samples from {manifest_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        audio_path = item["audio_path"]
        text = item["normalized_text"]

        # Load and resample audio
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != self.target_sr:
            gcd = math.gcd(int(sr), int(self.target_sr))
            up = int(self.target_sr // gcd)
            down = int(sr // gcd)
            data = scipy.signal.resample_poly(data, up, down)

        data = data.astype(np.float32)

        # Normalize audio amplitude
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val * 0.95

        # Extract mel spectrogram frames
        audio_features = self.processor(audio_target=data, sampling_rate=self.target_sr, return_tensors="pt")
        labels = audio_features["input_values"][0]  # shape: [seq_len, 80]

        # Truncate by 1 if odd for reduction_factor = 2
        if labels.shape[0] % 2 != 0:
            labels = labels[:-1, :]

        # Encode text
        text_inputs = self.processor(text=text, return_tensors="pt")
        input_ids = text_inputs["input_ids"][0]  # shape: [text_len]

        return {
            "id": item["id"],
            "input_ids": input_ids,
            "labels": labels,
            "text": text,
        }


def collate_tts_fn(batch: List[Dict[str, Any]], pad_token_id: int = 1) -> Dict[str, torch.Tensor]:
    """Collate and pad variable-length texts and spectrograms."""
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    # Pad text input_ids
    max_text_len = max(len(ids) for ids in input_ids_list)
    padded_input_ids = torch.full((len(batch), max_text_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_text_len), dtype=torch.long)

    for i, ids in enumerate(input_ids_list):
        padded_input_ids[i, :len(ids)] = ids
        attention_mask[i, :len(ids)] = 1

    # Pad spectrograms
    max_spec_len = max(l.shape[0] for l in labels_list)
    # Ensure divisible by 2
    if max_spec_len % 2 != 0:
        max_spec_len += 1

    padded_labels = torch.zeros((len(batch), max_spec_len, 80), dtype=torch.float32)
    for i, spec in enumerate(labels_list):
        padded_labels[i, :spec.shape[0], :] = spec

    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("========================================================")
    logger.info("      Apatani Text-to-Speech (TTS) Training Run         ")
    logger.info("========================================================")
    logger.info(f"Device: {device}")
    logger.info(f"Model:  {args.model_name_or_path}")

    # Load processor, model, and vocoder
    processor = SpeechT5Processor.from_pretrained(args.model_name_or_path)
    model = SpeechT5ForTextToSpeech.from_pretrained(args.model_name_or_path).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(args.vocoder_name_or_path).to(device)
    vocoder.eval()

    # Learnable or fixed representative speaker embedding for Apatani speaker
    speaker_embedding = nn.Parameter(
        torch.randn(1, 512, device=device) * 0.05,
        requires_grad=True,
    )

    # Prepare datasets and loaders
    train_dataset = ApataniTTSDataset(args.train_manifest, processor)
    val_dataset = ApataniTTSDataset(args.val_manifest, processor)

    pad_id = processor.tokenizer.pad_token_id or 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_tts_fn(b, pad_token_id=pad_id),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_tts_fn(b, pad_token_id=pad_id),
        num_workers=2,
        pin_memory=True,
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [speaker_embedding],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(10, int(total_steps * 0.05)),
        num_training_steps=max(20, total_steps),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and torch.cuda.is_available())

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    logger.info(f"Starting training for {args.num_train_epochs} epochs ({total_steps} update steps)...")

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_train_epochs} [Train]")
        for step, batch in enumerate(pbar, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Expand speaker embedding to batch size
            bsz = input_ids.shape[0]
            spk_emb = speaker_embedding.repeat(bsz, 1)

            with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    speaker_embeddings=spk_emb,
                    labels=labels,
                )
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            train_loss_total += loss.item() * args.gradient_accumulation_steps
            train_steps += 1

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            pbar.set_postfix({"loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"})

        avg_train_loss = train_loss_total / max(1, train_steps)

        # Validation Loop
        model.eval()
        val_loss_total = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                bsz = input_ids.shape[0]
                spk_emb = speaker_embedding.repeat(bsz, 1)

                with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        speaker_embeddings=spk_emb,
                        labels=labels,
                    )
                val_loss_total += outputs.loss.item()
                val_steps += 1

        avg_val_loss = val_loss_total / max(1, val_steps)
        logger.info(
            f"Epoch {epoch}/{args.num_train_epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # Checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving checkpoint to {args.output_dir}...")
            model.save_pretrained(args.output_dir)
            processor.save_pretrained(args.output_dir)
            torch.save(speaker_embedding.detach().cpu(), os.path.join(args.output_dir, "speaker_embedding.pt"))

            # Synthesize sample audio for validation
            sample_text = "Hopa Ngo nunumi lukoso, nunuka sangomi hena siiyo."
            try:
                inputs = processor(text=sample_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    speech = model.generate_speech(
                        inputs["input_ids"],
                        speaker_embedding,
                        vocoder=vocoder,
                    )
                sample_wav_path = os.path.join(args.output_dir, "sample_synthesized.wav")
                sf.write(sample_wav_path, speech.cpu().numpy(), samplerate=16000)
                logger.info(f"Synthesized sample audio saved to: {sample_wav_path}")
            except Exception as e:
                logger.warning(f"Could not generate sample speech: {e}")

    # Test Set Evaluation
    if os.path.exists(args.test_manifest):
        logger.info(f"Evaluating best model on Test Set ({args.test_manifest})...")
        test_dataset = ApataniTTSDataset(args.test_manifest, processor)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_tts_fn(b, pad_token_id=pad_id),
            num_workers=2,
        )
        test_loss_total = 0.0
        test_steps = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                bsz = input_ids.shape[0]
                spk_emb = speaker_embedding.repeat(bsz, 1)

                with torch.cuda.amp.autocast(enabled=args.fp16 and torch.cuda.is_available()):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        speaker_embeddings=spk_emb,
                        labels=labels,
                    )
                test_loss_total += outputs.loss.item()
                test_steps += 1

        avg_test_loss = test_loss_total / max(1, test_steps)
        logger.info(f"Test Set Loss: {avg_test_loss:.4f}")

    logger.info(f"Training complete! Best model saved at: {args.output_dir} (Val Loss: {best_val_loss:.4f})")


if __name__ == "__main__":
    main()
