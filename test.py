#!/usr/bin/env python3
"""
NLLB Standalone Model Evaluation & Inference Script
===================================================
Evaluates a saved/fine-tuned NLLB model on a dataset or custom sentences.
Computes SacreBLEU and ChrF++ (word_order=2) metrics, prints sample outputs,
and exports translation predictions to a JSON report.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned NLLB model with SacreBLEU & ChrF++."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./best_model",
        help="Path to the saved fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/nyishi_test.tsv",
        help="Path to dataset file (.tsv, .csv, .json, .parquet).",
    )
    parser.add_argument(
        "--source_column",
        type=str,
        default="english",
        help="Name of the source text column.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="nyishi",
        help="Name of the target reference column.",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        default="eng_Latn",
        help="Source language FLORES-200 code.",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="hin_Deva",
        help="Target language FLORES-200 code.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference evaluation.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Max number of samples to evaluate on (set to 0 for all).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum generation sequence length.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation metrics and sample predictions.",
    )
    return parser.parse_args()


def load_evaluation_data(data_file: str, src_col: str, tgt_col: str, max_samples: int = 0) -> pd.DataFrame:
    if not os.path.exists(data_file):
        for fallback in ["data/nyishi_test.tsv", "data/nyishi_val.tsv", "nyishi_train_cleaned.tsv", "TSV data/nyishi_train.tsv"]:
            if os.path.exists(fallback):
                data_file = fallback
                break
    ext = Path(data_file).suffix.lower()
    if ext == ".tsv":
        with open(data_file, "r", encoding="utf-8", errors="replace") as fp:
            first_line = fp.readline().strip().split("\t")
        has_header = any(h.lower() in [src_col.lower(), tgt_col.lower(), "source", "target", "src", "tgt"] for h in first_line)
        if has_header:
            df = pd.read_csv(data_file, sep="\t")
        else:
            df = pd.read_csv(data_file, sep="\t", header=None, names=[src_col, tgt_col])
    elif ext == ".csv":
        with open(data_file, "r", encoding="utf-8", errors="replace") as fp:
            first_line = fp.readline().strip().split(",")
        has_header = any(h.lower() in [src_col.lower(), tgt_col.lower()] for h in first_line)
        if has_header:
            df = pd.read_csv(data_file)
        else:
            df = pd.read_csv(data_file, header=None, names=[src_col, tgt_col])
    elif ext in [".json", ".jsonl"]:
        df = pd.read_json(data_file, lines=(ext == ".jsonl"))
    elif ext == ".parquet":
        df = pd.read_parquet(data_file)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    df = df.dropna(subset=[src_col, tgt_col]).copy()
    df[src_col] = df[src_col].astype(str).str.strip()
    df[tgt_col] = df[tgt_col].astype(str).str.strip()
    df = df[(df[src_col] != "") & (df[tgt_col] != "")].reset_index(drop=True)

    if max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    return df


def evaluate_model(
    model_dir: str,
    data_file: str,
    src_col: str,
    tgt_col: str,
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 16,
    max_samples: int = 1000,
    max_length: int = 128,
    output_file: str = "evaluation_results.json",
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model and tokenizer from '{model_dir}' on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()

    # Determine forced_bos_token_id
    forced_bos_token_id = None
    if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    elif tgt_lang in tokenizer.get_vocab():
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    elif model.config.forced_bos_token_id is not None:
        forced_bos_token_id = model.config.forced_bos_token_id

    logger.info(f"Loading dataset from '{data_file}'...")
    df = load_evaluation_data(data_file, src_col, tgt_col, max_samples=max_samples)
    logger.info(f"Evaluating on {len(df):,} test pairs...")

    sources = df[src_col].tolist()
    references = df[tgt_col].tolist()

    all_predictions: List[str] = []

    # Batch generation
    num_batches = (len(sources) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating translations"):
            batch_src = sources[i * batch_size : (i + 1) * batch_size]
            tokenizer.src_lang = src_lang
            inputs = tokenizer(
                batch_src,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)

            gen_kwargs = {
                "max_length": max_length,
                "num_beams": 4,
                "early_stopping": True,
            }
            if forced_bos_token_id is not None:
                gen_kwargs["forced_bos_token_id"] = forced_bos_token_id

            generated_tokens = model.generate(**inputs, **gen_kwargs)
            decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_predictions.extend([p.strip() for p in decoded])

    # Compute metrics using evaluate
    sacrebleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    formatted_refs = [[ref] for ref in references]
    bleu_score = sacrebleu_metric.compute(predictions=all_predictions, references=formatted_refs)["score"]
    chrf_score = chrf_metric.compute(predictions=all_predictions, references=formatted_refs, word_order=2)["score"]

    results = {
        "metrics": {
            "sacrebleu": round(float(bleu_score), 4),
            "chrf++": round(float(chrf_score), 4),
        },
        "sample_count": len(sources),
        "samples": [
            {
                "source": s,
                "reference": r,
                "prediction": p,
            }
            for s, r, p in zip(sources[:10], references[:10], all_predictions[:10])
        ],
    }

    logger.info("================ Evaluation Results ================")
    logger.info(f"SacreBLEU: {results['metrics']['sacrebleu']}")
    logger.info(f"ChrF++:    {results['metrics']['chrf++']}")
    logger.info("====================================================")

    # Print first few samples
    logger.info("Sample Translations:")
    for idx, sample in enumerate(results["samples"][:5]):
        logger.info(f"[{idx+1}] Source:     {sample['source']}")
        logger.info(f"    Reference:  {sample['reference']}")
        logger.info(f"    Prediction: {sample['prediction']}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved detailed results to '{output_file}'.")

    return results


def main():
    args = parse_args()
    evaluate_model(
        model_dir=args.model_dir,
        data_file=args.data_file,
        src_col=args.source_column,
        tgt_col=args.target_column,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_length=args.max_length,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
