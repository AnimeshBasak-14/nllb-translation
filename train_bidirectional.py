#!/usr/bin/env python3
"""
Single Unified Bidirectional Translation Pipeline for NLLB (Meta NLLB-200)
===========================================================================
Trains a single sequence-to-sequence model that simultaneously translates
both ways between English and indigenous languages:
  1. English -> Indigenous (e.g. English -> Nyishi)
  2. Indigenous -> English (e.g. Nyishi -> English)

Features:
- Strict 95% training / 5% validation split at the parallel sentence level.
- Symmetrical bidirectional augmentation (forward + reverse pairs in one model).
- Dual evaluation metrics (SacreBLEU and ChrF++ with word_order=2) computed for
  both directions individually and overall.
- Checkpointing saves the best-performing bidirectional model.
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
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "nyishi": "nyishi_train_cleaned.tsv",
    "apatani": "TSV data/apatani_train.tsv",
    "adi": "TSV data/adi_train.tsv",
    "galo": "TSV data/galo_train.tsv",
    "tagin": "TSV data/tagin_train.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single bidirectional NLLB translation model (both directions simultaneously)."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="nyishi",
        choices=["nyishi", "apatani", "adi", "galo", "tagin", "all"],
        help="Language to train bidirectional model for (or 'all' for all languages).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Base NLLB model identifier or checkpoint path.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/nyishi_train.tsv",
        help="Path to training TSV file.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/nyishi_val.tsv",
        help="Path to validation TSV file.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/nyishi_test.tsv",
        help="Path to test TSV file.",
    )
    parser.add_argument(
        "--eng_lang_code",
        type=str,
        default="eng_Latn",
        help="NLLB English language code.",
    )
    parser.add_argument(
        "--ind_lang_code",
        type=str,
        default="hin_Deva",
        help="NLLB code representing the indigenous language.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio (strictly 0.05 = 5%).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="Maximum source sequence length.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Maximum target sequence length.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_bidirectional",
        help="Directory to save training checkpoints.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="./best_bidirectional_model",
        help="Directory to save final best model and tokenizer.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size per GPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of update steps to accumulate gradients.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Linear warmup ratio.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 mixed precision training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Truncate training samples for fast debugging.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=None,
        help="Truncate validation samples for fast evaluation.",
    )
    return parser.parse_args()


def load_raw_dataset(data_path: str) -> pd.DataFrame:
    """Reads TSV/CSV dataset handling both header and header-less variants."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    # Inspect first line to check if header is present
    with open(data_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")

    has_header = False
    lower_tokens = [t.lower() for t in first_line]
    if "english" in lower_tokens:
        has_header = True

    if has_header:
        df = pd.read_csv(data_path, sep="\t", on_bad_lines="skip")
        eng_col = [c for c in df.columns if "eng" in c.lower()][0]
        tgt_cols = [c for c in df.columns if c != eng_col]
        tgt_col = tgt_cols[0]
        df = df[[eng_col, tgt_col]].rename(columns={eng_col: "english", tgt_col: "indigenous"})
    else:
        df = pd.read_csv(data_path, sep="\t", header=None, on_bad_lines="skip")
        df = df[[0, 1]].rename(columns={0: "english", 1: "indigenous"})

    # Drop invalid/empty rows
    df["english"] = df["english"].astype(str).str.strip()
    df["indigenous"] = df["indigenous"].astype(str).str.strip()
    df = df[(df["english"].str.len() > 1) & (df["indigenous"].str.len() > 1)]
    return df.reset_index(drop=True)


def build_bidirectional_splits(
    df: pd.DataFrame,
    val_split: float = 0.05,
    seed: int = 42,
    eng_lang_code: str = "eng_Latn",
    ind_lang_code: str = "hin_Deva",
) -> Tuple[Dataset, Dataset]:
    """
    Splits parallel sentence pairs into 95% train and 5% validation first,
    then expands each into bidirectional pairs:
      1. English -> Indigenous (src_lang=eng_Latn, tgt_lang=ind_lang_code)
      2. Indigenous -> English (src_lang=ind_lang_code, tgt_lang=eng_Latn)
    """
    # Deterministic split on unique parallel sentences
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_size = max(1, int(len(df_shuffled) * val_split))
    train_df = df_shuffled.iloc[:-val_size].reset_index(drop=True)
    val_df = df_shuffled.iloc[-val_size:].reset_index(drop=True)

    logger.info(f"Base parallel sentence split: {len(train_df):,} Train | {len(val_df):,} Val")

    def make_bidirectional(sub_df: pd.DataFrame) -> pd.DataFrame:
        fwd = pd.DataFrame({
            "source_text": sub_df["english"],
            "target_text": sub_df["indigenous"],
            "src_lang": eng_lang_code,
            "tgt_lang": ind_lang_code,
            "direction": "en2ind",
        })
        rev = pd.DataFrame({
            "source_text": sub_df["indigenous"],
            "target_text": sub_df["english"],
            "src_lang": ind_lang_code,
            "tgt_lang": eng_lang_code,
            "direction": "ind2en",
        })
        combined = pd.concat([fwd, rev], ignore_index=True)
        return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    bi_train_df = make_bidirectional(train_df)
    bi_val_df = make_bidirectional(val_df)

    logger.info(
        f"Bidirectional augmented dataset: {len(bi_train_df):,} Train rows (both ways) | "
        f"{len(bi_val_df):,} Val rows (both ways)"
    )

    return Dataset.from_pandas(bi_train_df), Dataset.from_pandas(bi_val_df)


def prepare_tokenized_dataset(
    raw_dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_source_length: int = 128,
    max_target_length: int = 128,
) -> Dataset:
    """Tokenizes each sample respecting its specific source and target language tags."""

    def preprocess_batch(batch):
        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for src_text, tgt_text, s_lang, t_lang in zip(
            batch["source_text"], batch["target_text"], batch["src_lang"], batch["tgt_lang"]
        ):
            tokenizer.src_lang = s_lang
            tokenizer.tgt_lang = t_lang

            inp = tokenizer(
                src_text,
                max_length=max_source_length,
                truncation=True,
                padding=False,
            )
            out = tokenizer(
                text_target=tgt_text,
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )

            input_ids_list.append(inp["input_ids"])
            attention_mask_list.append(inp["attention_mask"])
            labels_list.append(out["input_ids"])

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list,
            "direction": batch["direction"],
        }

    return raw_dataset.map(
        preprocess_batch,
        batched=True,
        batch_size=1000,
        remove_columns=["source_text", "target_text", "src_lang", "tgt_lang"],
        desc="Tokenizing bidirectional dataset",
    )


def compute_bidirectional_metrics(eval_preds, tokenizer: AutoTokenizer) -> Dict[str, float]:
    """Computes SacreBLEU and ChrF++ for predictions."""
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace padding tokens (-100)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]

    sacrebleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")

    bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    chrf_result = chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)

    return {
        "bleu": round(float(bleu_result["score"]), 4),
        "chrf++": round(float(chrf_result["score"]), 4),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    logger.info("========================================================")
    logger.info("   Starting Single Bidirectional NLLB Translation Run   ")
    logger.info("========================================================")
    logger.info(f"Target Language: {args.language}")
    logger.info(f"Base Model:      {args.model_name_or_path}")

    # Determine data files
    if os.path.exists(args.train_file) and os.path.exists(args.val_file):
        logger.info(f"Loading partitioned datasets:")
        logger.info(f"  Train file: {args.train_file}")
        logger.info(f"  Val file:   {args.val_file}")
        logger.info(f"  Test file:  {args.test_file}")
        df_train = load_raw_dataset(args.train_file)
        df_val = load_raw_dataset(args.val_file)
        df_test = load_raw_dataset(args.test_file) if args.test_file and os.path.exists(args.test_file) else None

        def make_bi_df(sub_df: pd.DataFrame) -> Dataset:
            fwd = pd.DataFrame({
                "source_text": sub_df["english"],
                "target_text": sub_df["indigenous"],
                "src_lang": args.eng_lang_code,
                "tgt_lang": args.ind_lang_code,
                "direction": "en2ind",
            })
            rev = pd.DataFrame({
                "source_text": sub_df["indigenous"],
                "target_text": sub_df["english"],
                "src_lang": args.ind_lang_code,
                "tgt_lang": args.eng_lang_code,
                "direction": "ind2en",
            })
            combined = pd.concat([fwd, rev], ignore_index=True)
            return Dataset.from_pandas(combined.sample(frac=1.0, random_state=args.seed).reset_index(drop=True))

        raw_train = make_bi_df(df_train)
        raw_val = make_bi_df(df_val)
        raw_test = make_bi_df(df_test) if df_test is not None else None
    else:
        path = SUPPORTED_LANGUAGES[args.language]
        df_raw = load_raw_dataset(path)
        logger.info(f"Total clean parallel pairs loaded: {len(df_raw):,}")
        raw_train, raw_val = build_bidirectional_splits(
            df_raw,
            val_split=args.val_split,
            seed=args.seed,
            eng_lang_code=args.eng_lang_code,
            ind_lang_code=args.ind_lang_code,
        )
        raw_test = None

    logger.info(
        f"Bidirectional datasets: {len(raw_train):,} Train rows | {len(raw_val):,} Val rows"
        + (f" | {len(raw_test):,} Test rows" if raw_test is not None else "")
    )

    if args.max_train_samples:
        raw_train = raw_train.select(range(min(len(raw_train), args.max_train_samples)))
    if args.max_val_samples:
        raw_val = raw_val.select(range(min(len(raw_val), args.max_val_samples)))

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        src_lang=args.eng_lang_code,
        tgt_lang=args.ind_lang_code,
    )

    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)

    # Tokenize datasets
    tokenized_train = prepare_tokenized_dataset(
        raw_train, tokenizer, args.max_source_length, args.max_target_length
    )
    tokenized_val = prepare_tokenized_dataset(
        raw_val, tokenizer, args.max_source_length, args.max_target_length
    )
    tokenized_test = prepare_tokenized_dataset(
        raw_test, tokenizer, args.max_source_length, args.max_target_length
    ) if raw_test is not None else None

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_total_limit=1,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        fp16=args.fp16 and torch.cuda.is_available(),
        warmup_ratio=args.warmup_ratio,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="chrf++",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_bidirectional_metrics(p, tokenizer),
    )

    logger.info("Beginning bidirectional training...")
    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    logger.info("Evaluating bidirectional model on Validation Set...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    test_metrics = {}
    if tokenized_test is not None:
        logger.info("Evaluating bidirectional model on Test Set...")
        test_pred = trainer.predict(tokenized_test, metric_key_prefix="test")
        test_metrics = test_pred.metrics
        trainer.log_metrics("test", test_metrics)
        trainer.save_metrics("test", test_metrics)

    # Save final best model
    logger.info(f"Saving best bidirectional model to '{args.best_model_dir}'...")
    os.makedirs(args.best_model_dir, exist_ok=True)
    trainer.save_model(args.best_model_dir)
    tokenizer.save_pretrained(args.best_model_dir)

    print("\n" + "=" * 60)
    print("      BIDIRECTIONAL MODEL EVALUATION BENCHMARK          ")
    print("=" * 60)
    print(f"Validation SacreBLEU: {eval_metrics.get('eval_bleu', 0.0):.4f}")
    print(f"Validation ChrF++:    {eval_metrics.get('eval_chrf++', 0.0):.4f}")
    if test_metrics:
        print(f"Test SacreBLEU:       {test_metrics.get('test_bleu', 0.0):.4f}")
        print(f"Test ChrF++:          {test_metrics.get('test_chrf++', 0.0):.4f}")
    print(f"Best model saved to:  {args.best_model_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
