#!/usr/bin/env python3
"""
Unified Multilingual NLLB Translation Pipeline for Arunachal Pradesh Languages
================================================================================
Supports all 5 indigenous languages (Adi, Apatani, Galo, Nyishi, Tagin):
- Loads individual datasets or builds a unified multilingual model (97,408 total pairs).
- Auto-detects header and header-less TSV formats.
- Applies strict 95% training / 5% validation splitting.
- Evaluates with SacreBLEU and ChrF++ (word_order=2) via evaluate.
- Saves best-performing model checkpoint via Seq2SeqTrainer with mixed precision.
"""

import argparse
import glob
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
    "adi": "TSV data/adi_train.tsv",
    "apatani": "TSV data/apatani_train.tsv",
    "galo": "TSV data/galo_train.tsv",
    "nyishi": "TSV data/nyishi_train.tsv",
    "tagin": "TSV data/tagin_train.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NLLB translation models for Arunachal Pradesh languages."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="all",
        choices=["all", "adi", "apatani", "galo", "nyishi", "tagin"],
        help="Target language to train on, or 'all' for joint multilingual training.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Pretrained NLLB model identifier.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="Optional custom dataset file path. Overrides --language dataset lookup.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="NLLB source language code.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="NLLB target language code or base token.",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.05,
        help="Validation split ratio (strictly 0.05 = 5%%).",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=128,
        help="Max input tokens.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="Max target tokens.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints_multilingual",
        help="Directory for intermediate checkpoints.",
    )
    parser.add_argument(
        "--best_model_dir",
        type=str,
        default="./best_multilingual_model",
        help="Directory to save the best model.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Train batch size per GPU.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Eval batch size per GPU.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional training set sample limit for rapid iteration.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional validation set sample limit.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="chrf++",
        choices=["chrf++", "bleu"],
        help="Validation selection metric.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 mixed precision.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def load_dataset_for_language(language: str, custom_file: Optional[str] = None) -> pd.DataFrame:
    """Loads and formats single or combined multilingual parallel datasets."""
    if custom_file and os.path.exists(custom_file):
        files_to_load = [(language, custom_file)]
    elif language == "all":
        files_to_load = [(lang, path) for lang, path in SUPPORTED_LANGUAGES.items() if os.path.exists(path)]
    else:
        path = SUPPORTED_LANGUAGES.get(language)
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Dataset for language '{language}' not found at '{path}'.")
        files_to_load = [(language, path)]

    all_dfs = []
    for lang, fpath in files_to_load:
        with open(fpath, "r", encoding="utf-8", errors="replace") as fp:
            first_line = fp.readline().strip().split("\t")
        has_header = any(h.lower() in ["english", "source", "src"] for h in first_line)

        if has_header:
            df_cur = pd.read_csv(fpath, sep="\t")
            # Normalize column names to 'english' and 'target'
            cols = list(df_cur.columns)
            df_cur = df_cur.rename(columns={cols[0]: "english", cols[1]: "target"})
        else:
            df_cur = pd.read_csv(fpath, sep="\t", header=None, names=["english", "target"])

        # Data cleaning: drop NaN and empty strings
        df_cur = df_cur.dropna(subset=["english", "target"]).copy()
        df_cur["english"] = df_cur["english"].astype(str).str.strip()
        df_cur["target"] = df_cur["target"].astype(str).str.strip()
        df_cur = df_cur[(df_cur["english"] != "") & (df_cur["target"] != "")]

        if language == "all":
            # Prepend language token to source text: e.g. "[apatani] <english text>"
            df_cur["english"] = f"[{lang}] " + df_cur["english"]

        df_cur["lang"] = lang
        all_dfs.append(df_cur)
        logger.info(f"Loaded {len(df_cur):,} valid pairs for '{lang}' from '{fpath}'.")

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Total dataset size: {len(combined):,} parallel sentence pairs.")
    return combined


def build_compute_metrics_fn(tokenizer: Any):
    sacrebleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    def compute_metrics(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        cleaned_preds: List[str] = []
        cleaned_refs: List[List[str]] = []

        for p, l in zip(decoded_preds, decoded_labels):
            pred_str = p.strip()
            ref_str = l.strip()
            if len(ref_str) > 0:
                cleaned_preds.append(pred_str if len(pred_str) > 0 else " ")
                cleaned_refs.append([ref_str])

        if not cleaned_preds:
            return {"bleu": 0.0, "chrf++": 0.0}

        bleu_res = sacrebleu_metric.compute(predictions=cleaned_preds, references=cleaned_refs)
        chrf_res = chrf_metric.compute(predictions=cleaned_preds, references=cleaned_refs, word_order=2)

        return {
            "bleu": round(float(bleu_res["score"]), 4),
            "chrf++": round(float(chrf_res["score"]), 4),
        }

    return compute_metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available() and not args.fp16:
        args.fp16 = True
        logger.info(f"CUDA detected: {torch.cuda.get_device_name(0)}. FP16 enabled.")

    # 1. Load Data
    df = load_dataset_for_language(args.language, custom_file=args.data_file)
    raw_dataset = Dataset.from_pandas(df[["english", "target"]])

    # Strict 95% Train / 5% Validation Split
    split_dataset = raw_dataset.train_test_split(test_size=args.val_split, seed=args.seed, shuffle=True)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    logger.info(
        f"Data split: {len(train_data):,} train ({(1-args.val_split)*100:.1f}%) / "
        f"{len(val_data):,} val ({args.val_split*100:.1f}%)"
    )

    # 2. Tokenizer & Model
    logger.info(f"Loading tokenizer & model: '{args.model_name_or_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, src_lang=args.src_lang)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, config=config)

    # Add language prefix tokens if joint multilingual
    if args.language == "all":
        special_tokens = [f"[{lang}]" for lang in SUPPORTED_LANGUAGES.keys()]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Registered multilingual tokens: {special_tokens}")

    tokenizer.tgt_lang = args.tgt_lang
    if hasattr(tokenizer, "lang_code_to_id") and args.tgt_lang in tokenizer.lang_code_to_id:
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[args.tgt_lang]
    elif args.tgt_lang in tokenizer.get_vocab():
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    # 3. Preprocessing
    def preprocess_batch(examples):
        inputs = [str(s).strip() for s in examples["english"]]
        targets = [str(t).strip() for t in examples["target"]]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, truncation=True, padding=False)
        labels = tokenizer(text_target=targets, max_length=args.max_target_length, truncation=True, padding=False)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing train and validation splits...")
    tokenized_train = train_data.map(preprocess_batch, batched=True, remove_columns=train_data.column_names)
    tokenized_val = val_data.map(preprocess_batch, batched=True, remove_columns=val_data.column_names)

    if args.max_train_samples:
        tokenized_train = tokenized_train.select(range(min(len(tokenized_train), args.max_train_samples)))
    if args.max_eval_samples:
        tokenized_val = tokenized_val.select(range(min(len(tokenized_val), args.max_eval_samples)))

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if args.fp16 else None,
    )

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
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=0.1,
        fp16=args.fp16,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics_fn(tokenizer),
    )

    logger.info(f"Starting training on {args.language.upper()} dataset...")
    train_res = trainer.train()
    trainer.log_metrics("train", train_res.metrics)
    trainer.save_metrics("train", train_res.metrics)

    logger.info("Evaluating best checkpoint...")
    eval_res = trainer.evaluate()
    trainer.log_metrics("eval", eval_res)
    trainer.save_metrics("eval", eval_res)

    print("\n========================================================")
    print(f"        {args.language.upper()} Translation Evaluation Results")
    print("========================================================")
    print(f"SacreBLEU Score: {eval_res.get('eval_bleu', 'N/A')}")
    print(f"ChrF++ Score:    {eval_res.get('eval_chrf++', 'N/A')}")
    print("========================================================\n")

    os.makedirs(args.best_model_dir, exist_ok=True)
    logger.info(f"Saving best model to '{args.best_model_dir}'...")
    trainer.save_model(args.best_model_dir)
    tokenizer.save_pretrained(args.best_model_dir)
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
