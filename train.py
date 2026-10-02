#!/usr/bin/env python3
"""
NLLB Fine-Tuning and Evaluation Pipeline
=========================================
Fine-tunes Meta's NLLB (No Language Left Behind) translation model using
Hugging Face Transformers, PyTorch, and Evaluate.

Key Features:
- Default model: facebook/nllb-200-distilled-600M (configurable)
- Strict 95% training / 5% validation data split
- Evaluation with SacreBLEU and ChrF++ (word_order=2)
- Automatic checkpointing and saving of the best-performing model
- Mixed-precision training support (FP16 / BF16)
- Ready for local datasets (TSV, CSV, JSON, Parquet) and Hugging Face Hub
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    NllbTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Meta NLLB model with SacreBLEU & ChrF++ evaluation."
    )

    # Model configuration
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Path or Hugging Face model identifier for the NLLB model.",
    )

    # Data configuration
    parser.add_argument(
        "--data_file",
        type=str,
        default="nyishi_train_cleaned.tsv",
        help="Path to local translation data file (.tsv, .csv, .json, .parquet).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Hugging Face Hub dataset name (if not using local data_file).",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="english",
        help="Name of the source language text column in the dataset.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="nyishi",
        help="Name of the target language text column in the dataset.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="NLLB source language code (e.g., 'eng_Latn').",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="NLLB target language code (e.g., 'hin_Deva' or FLORES-200 code).",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Fraction of data reserved for validation (strictly 0.05 = 5%%).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="Maximum input sequence length in tokens.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Maximum target sequence length in tokens.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to store training intermediate checkpoints.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="./best_model",
        help="Directory to export the best-performing model after training.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of update steps to accumulate before backward/update pass.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Linear warmup ratio over total training steps.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for regularization.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=50,
        help="Log training metrics every X steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Limit the total amount of checkpoints. Deletes older checkpoints.",
    )

    # Evaluation & Model Checkpointing
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="chrf++",
        choices=["chrf++", "bleu"],
        help="Validation metric to determine the best model checkpoint ('chrf++' or 'bleu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across splits and initialization.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (recommended for CUDA).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bfloat16 precision (if supported by hardware).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help="Tracking integrations ('tensorboard', 'wandb', or 'none').",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loading and Splitting (95% Train / 5% Validation)
# ---------------------------------------------------------------------------
def load_and_split_data(
    data_file: Optional[str],
    dataset_name: Optional[str],
    source_col: str,
    target_col: str,
    val_split: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Loads dataset from local file or Hugging Face Hub, cleans missing/null rows,
    and applies a strict train/validation split (95% train, 5% val by default).
    """
    if data_file and os.path.exists(data_file):
        logger.info(f"Loading local dataset from: {data_file}")
        ext = Path(data_file).suffix.lower()

        if ext == ".tsv":
            df = pd.read_csv(data_file, sep="\t")
        elif ext == ".csv":
            df = pd.read_csv(data_file)
        elif ext in [".json", ".jsonl"]:
            df = pd.read_json(data_file, lines=(ext == ".jsonl"))
        elif ext == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    elif dataset_name:
        logger.info(f"Loading dataset from Hugging Face Hub: {dataset_name}")
        hf_dataset = load_dataset(dataset_name)
        if isinstance(hf_dataset, DatasetDict):
            if "train" in hf_dataset and "validation" in hf_dataset:
                return hf_dataset["train"], hf_dataset["validation"]
            df = pd.DataFrame(hf_dataset["train"])
        else:
            df = pd.DataFrame(hf_dataset)
    else:
        raise FileNotFoundError(
            f"Dataset file '{data_file}' not found and no '--dataset_name' specified."
        )

    # Validate column presence
    if source_col not in df.columns or target_col not in df.columns:
        raise KeyError(
            f"Required columns ('{source_col}', '{target_col}') not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Data Quality: drop NaN and empty strings
    initial_count = len(df)
    df = df.dropna(subset=[source_col, target_col]).copy()
    df[source_col] = df[source_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()
    df = df[(df[source_col] != "") & (df[target_col] != "")].reset_index(drop=True)
    cleaned_count = len(df)

    dropped = initial_count - cleaned_count
    if dropped > 0:
        logger.warning(f"Filtered out {dropped} null or empty translation pairs.")

    logger.info(f"Total valid samples: {cleaned_count:,}")

    # Convert to Hugging Face Dataset
    raw_dataset = Dataset.from_pandas(df)

    # Strict train/val split (95% / 5%)
    train_pct = (1.0 - val_split) * 100
    val_pct = val_split * 100
    logger.info(f"Performing strict {train_pct:.1f}% train / {val_pct:.1f}% validation split (seed={seed})...")

    split_dataset = raw_dataset.train_test_split(
        test_size=val_split,
        seed=seed,
        shuffle=True,
    )

    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    logger.info(
        f"Data split complete: Train samples = {len(train_data):,} ({train_pct:.1f}%), "
        f"Validation samples = {len(val_data):,} ({val_pct:.1f}%)"
    )

    return train_data, val_data


# ---------------------------------------------------------------------------
# Preprocessing and Tokenization
# ---------------------------------------------------------------------------
def prepare_tokenized_datasets(
    train_dataset: Dataset,
    val_dataset: Dataset,
    tokenizer: Any,
    source_col: str,
    target_col: str,
    src_lang: str,
    tgt_lang: str,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> Tuple[Dataset, Dataset]:
    """Tokenizes source and target sentences for NLLB Seq2Seq modeling."""

    def preprocess_batch(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        sources = [str(s).strip() for s in examples[source_col]]
        targets = [str(t).strip() for t in examples[target_col]]

        # Set source language on the tokenizer
        tokenizer.src_lang = src_lang

        # Tokenize source texts
        model_inputs = tokenizer(
            sources,
            max_length=max_source_length,
            truncation=True,
            padding=False,  # dynamic padding in collator
        )

        # Tokenize target texts with target language
        # Modern Hugging Face tokenizers support text_target
        tokenizer.src_lang = tgt_lang
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )

        # Restore source language on tokenizer
        tokenizer.src_lang = src_lang

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing training dataset...")
    tokenized_train = train_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )

    logger.info("Tokenizing validation dataset...")
    tokenized_val = val_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data",
    )

    return tokenized_train, tokenized_val


# ---------------------------------------------------------------------------
# Metrics Computation: SacreBLEU and ChrF++
# ---------------------------------------------------------------------------
def build_compute_metrics_fn(tokenizer: Any):
    """
    Creates the compute_metrics function computing SacreBLEU and ChrF++ (word_order=2).
    """
    sacrebleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        preds, labels = eval_preds

        # In case the model returns extra outputs (e.g. logits tuple)
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace label -100 masking with pad_token_id before decoding
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Decode tokens to strings
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip extra whitespace
        cleaned_preds: List[str] = []
        cleaned_refs: List[List[str]] = []

        for pred, label in zip(decoded_preds, decoded_labels):
            p = pred.strip()
            l = label.strip()
            if len(l) > 0:
                cleaned_preds.append(p if len(p) > 0 else " ")
                cleaned_refs.append([l])

        if not cleaned_preds or not cleaned_refs:
            return {"bleu": 0.0, "chrf++": 0.0}

        # Compute SacreBLEU
        bleu_res = sacrebleu_metric.compute(
            predictions=cleaned_preds,
            references=cleaned_refs,
        )

        # Compute ChrF++ (word_order=2 corresponds to ChrF++)
        chrf_res = chrf_metric.compute(
            predictions=cleaned_preds,
            references=cleaned_refs,
            word_order=2,
        )

        metrics = {
            "bleu": round(float(bleu_res["score"]), 4),
            "chrf++": round(float(chrf_res["score"]), 4),
        }
        return metrics

    return compute_metrics


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Detect hardware acceleration
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        logger.info(f"CUDA is available. Device: {torch.cuda.get_device_name(0)}")
        if not args.fp16 and not args.bf16:
            # Enable fp16 automatically on CUDA if not explicitly overridden
            args.fp16 = True
            logger.info("Automatically enabling FP16 mixed precision for CUDA acceleration.")
    else:
        logger.info("Running on CPU. Mixed precision (FP16/BF16) disabled.")
        args.fp16 = False
        args.bf16 = False

    logger.info("=== Configuration ===")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Source Language: {args.src_lang} | Target Language: {args.tgt_lang}")
    logger.info(f"Data File: {args.data_file}")
    logger.info(f"Validation Split: {args.val_split * 100:.1f}%")
    logger.info(f"Metric for Best Model: {args.metric_for_best_model}")
    logger.info(f"Output Checkpoints Dir: {args.output_dir}")
    logger.info(f"Best Model Save Dir: {args.best_model_dir}")

    # 1. Load and strictly split dataset (95% train / 5% val)
    train_raw, val_raw = load_and_split_data(
        data_file=args.data_file,
        dataset_name=args.dataset_name,
        source_col=args.source_column,
        target_col=args.target_column,
        val_split=args.val_split,
        seed=args.seed,
    )

    # 2. Load Tokenizer & Model
    logger.info(f"Loading tokenizer for '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )

    logger.info(f"Loading pretrained model for '{args.model_name_or_path}'...")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=config,
    )

    # Determine forced_bos_token_id for target language generation in NLLB
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and args.tgt_lang in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]
    elif args.tgt_lang in tokenizer.get_vocab():
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)
    else:
        # If target language token is new, add it to tokenizer
        logger.info(f"Adding custom target language token '{args.tgt_lang}' to tokenizer...")
        tokenizer.add_special_tokens({"additional_special_tokens": [args.tgt_lang]})
        model.resize_token_embeddings(len(tokenizer))
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    if forced_bos_token_id is not None:
        model.config.forced_bos_token_id = forced_bos_token_id
        logger.info(f"Configured forced_bos_token_id: {forced_bos_token_id} for target '{args.tgt_lang}'")

    # 3. Preprocess datasets
    tokenized_train, tokenized_val = prepare_tokenized_datasets(
        train_dataset=train_raw,
        val_dataset=val_raw,
        tokenizer=tokenizer,
        source_col=args.source_column,
        target_col=args.target_column,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )

    # 4. Data Collator with Dynamic Padding
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    # 5. Metrics computation function
    compute_metrics_fn = build_compute_metrics_fn(tokenizer)

    # 6. Seq2Seq Training Arguments with Best-Model Checkpointing
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
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        seed=args.seed,
    )

    # 7. Initialize Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    # 8. Start Training
    logger.info("Starting training loop...")
    train_result = trainer.train()

    # Log & Save training metrics
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 9. Evaluate Best Model
    logger.info("Evaluating best-performing model on validation set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    logger.info("================ Final Validation Results ================")
    logger.info(f"SacreBLEU Score: {eval_metrics.get('eval_bleu', 'N/A')}")
    logger.info(f"ChrF++ Score:    {eval_metrics.get('eval_chrf++', 'N/A')}")
    logger.info("==========================================================")

    # 10. Save the Best-Performing Model for Future Reuse
    os.makedirs(args.best_model_dir, exist_ok=True)
    logger.info(f"Saving best model checkpoint and tokenizer to: {args.best_model_dir}")
    trainer.save_model(args.best_model_dir)
    tokenizer.save_pretrained(args.best_model_dir)

    logger.info(
        f"Training pipeline successfully completed! Best model is ready at '{args.best_model_dir}'."
    )


if __name__ == "__main__":
    main()
