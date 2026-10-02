#!/usr/bin/env python3
"""
Apatani Automatic Speech Recognition (ASR) Pipeline
===================================================
Fine-tunes OpenAI Whisper (whisper-small / whisper-base) on the Apatani Speech Database:
- Loads paired 22.05 kHz audio and transcripts from preprocessed manifests.
- Resamples audio on-the-fly to 16 kHz using high-fidelity polyphase filtering.
- Computes Word Error Rate (WER) and Character Error Rate (CER) via jiwer.
- Employs Seq2SeqTrainer with mixed precision (FP16) on GPU.
- Automatically saves the best-performing ASR model based on validation WER.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import jiwer
import numpy as np
import scipy.signal
import soundfile as sf
import torch
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper ASR on Apatani Speech.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/whisper-small",
        help="Pretrained Whisper checkpoint (e.g. openai/whisper-small, openai/whisper-base).",
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
        "--output_dir",
        type=str,
        default="./checkpoints_asr",
        help="Directory to save intermediate training checkpoints.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="./best_apatani_asr",
        help="Directory to save the best-performing ASR model.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=5.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Train batch size per GPU.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Eval batch size per GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup ratio over total steps.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate train samples for fast iteration.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Truncate eval samples for fast iteration.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mixed precision training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def load_audio_resampled(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Loads audio file and resamples to target_sr using polyphase filtering."""
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)  # convert stereo to mono

    if sr != target_sr:
        gcd = np.gcd(int(sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(sr // gcd)
        data = scipy.signal.resample_poly(data, up, down)

    return data.astype(np.float32)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous steps
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(manifest_path: str, processor: WhisperProcessor, max_samples: int = None) -> Dataset:
    """Loads manifest JSON and processes audio waveforms and transcripts."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    logger.info(f"Loaded {len(data)} items from '{manifest_path}'. Processing waveforms...")

    processed_data = []
    for item in data:
        audio_path = item["audio_path"]
        text = item["normalized_text"]

        try:
            waveform = load_audio_resampled(audio_path, target_sr=16000)
            input_features = processor.feature_extractor(
                waveform, sampling_rate=16000
            ).input_features[0]

            labels = processor.tokenizer(text).input_ids

            processed_data.append({
                "input_features": input_features,
                "labels": labels,
            })
        except Exception as e:
            logger.warning(f"Error processing {audio_path}: {e}")

    return Dataset.from_list(processed_data)


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available() and not args.fp16:
        args.fp16 = True
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}. FP16 enabled.")

    logger.info(f"Initializing Whisper processor and model: '{args.model_name_or_path}'...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name_or_path)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name_or_path, language="english", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name_or_path, language="english", task="transcribe")

    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Prepare datasets
    train_dataset = prepare_dataset(args.train_manifest, processor, max_samples=args.max_train_samples)
    eval_dataset = prepare_dataset(args.val_manifest, processor, max_samples=args.max_eval_samples)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Compute metrics (WER & CER)
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        clean_preds = [p.strip() for p in pred_str]
        clean_labels = [l.strip() for l in label_str]

        wer_score = jiwer.wer(clean_labels, clean_preds)
        cer_score = jiwer.cer(clean_labels, clean_preds)

        return {
            "wer": round(float(wer_score) * 100, 2),
            "cer": round(float(cer_score) * 100, 2),
        }

    strategy_kwarg = {"eval_strategy": "epoch"}
    try:
        Seq2SeqTrainingArguments(output_dir=args.output_dir, eval_strategy="epoch")
    except (TypeError, ValueError):
        strategy_kwarg = {"evaluation_strategy": "epoch"}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        **strategy_kwarg,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=225,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    logger.info("Starting Whisper ASR fine-tuning...")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    logger.info("Evaluating best-performing ASR model...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("\n========================================================")
    print("            Apatani ASR Evaluation Results              ")
    print("========================================================")
    print(f"Word Error Rate (WER):      {eval_metrics.get('eval_wer', 'N/A')}%")
    print(f"Character Error Rate (CER): {eval_metrics.get('eval_cer', 'N/A')}%")
    print("========================================================\n")

    os.makedirs(args.best_model_dir, exist_ok=True)
    logger.info(f"Saving best ASR model and processor to: {args.best_model_dir}")
    trainer.save_model(args.best_model_dir)
    processor.save_pretrained(args.best_model_dir)

    logger.info(f"ASR pipeline complete! Model ready at '{args.best_model_dir}'.")


if __name__ == "__main__":
    main()
