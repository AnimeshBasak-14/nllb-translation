#!/usr/bin/env python3
"""
Dataset Partitioning Script (Train / Validation / Test Splits)
==============================================================
Partitions raw datasets into strict Train, Validation, and Test sets:
1. Nyishi <-> English Translation (MLT):
   - 90% Train (26,466 pairs)
   - 5% Validation (1,470 pairs)
   - 5% Test (1,470 pairs)
   Saved to: data/nyishi_train.tsv, data/nyishi_val.tsv, data/nyishi_test.tsv

2. Apatani Text-to-Speech (TTS):
   - 80% Train (~201 audio clips)
   - 10% Validation (~25 audio clips)
   - 10% Test (~25 audio clips)
   Saved to: Apatani_TTS_Database/train_manifest.json, val_manifest.json, test_manifest.json
"""

import csv
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def split_nyishi_translation(
    input_tsv: str = "nyishi_train_cleaned.tsv",
    output_dir: str = "data",
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(input_tsv):
        if os.path.exists("TSV data/nyishi_train.tsv"):
            input_tsv = "TSV data/nyishi_train.tsv"
        else:
            raise FileNotFoundError(f"Cannot find Nyishi dataset at {input_tsv}")

    # Inspect first line to check header
    with open(input_tsv, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split("\t")

    has_header = "english" in [t.lower() for t in first_line]
    if has_header:
        df = pd.read_csv(input_tsv, sep="\t", on_bad_lines="skip")
        eng_col = [c for c in df.columns if "eng" in c.lower()][0]
        nyi_col = [c for c in df.columns if c != eng_col][0]
        df = df[[eng_col, nyi_col]].rename(columns={eng_col: "english", nyi_col: "nyishi"})
    else:
        df = pd.read_csv(input_tsv, sep="\t", header=None, on_bad_lines="skip")
        df = df[[0, 1]].rename(columns={0: "english", 1: "nyishi"})

    df["english"] = df["english"].astype(str).str.strip()
    df["nyishi"] = df["nyishi"].astype(str).str.strip()
    df = df[(df["english"].str.len() > 1) & (df["nyishi"].str.len() > 1)].drop_duplicates()
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_total = len(df)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]

    train_path = os.path.join(output_dir, "nyishi_train.tsv")
    val_path = os.path.join(output_dir, "nyishi_val.tsv")
    test_path = os.path.join(output_dir, "nyishi_test.tsv")

    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)
    test_df.to_csv(test_path, sep="\t", index=False)

    logger.info("================ Nyishi Translation Split ================")
    logger.info(f"Total Unique Pairs: {n_total:,}")
    logger.info(f"Train Split (90%):  {len(train_df):,} pairs -> {train_path}")
    logger.info(f"Val Split   (5%):   {len(val_df):,} pairs -> {val_path}")
    logger.info(f"Test Split  (5%):   {len(test_df):,} pairs -> {test_path}")
    logger.info("==========================================================")


def split_apatani_tts(
    manifest_csv: str = "Apatani_TTS_Database/apatani_manifest.csv",
    output_dir: str = "Apatani_TTS_Database",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
):
    if not os.path.exists(manifest_csv):
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    items = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "id": row["id"],
                "audio_path": row["audio_path"],
                "raw_text": row["raw_text"],
                "normalized_text": row["normalized_text"],
                "duration": float(row["duration"]),
                "sample_rate": int(row["sample_rate"]),
                "channels": int(row["channels"]),
            })

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(items))

    n_total = len(items)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    train_items = [items[i] for i in indices[:n_train]]
    val_items = [items[i] for i in indices[n_train : n_train + n_val]]
    test_items = [items[i] for i in indices[n_train + n_val :]]

    train_manifest = os.path.join(output_dir, "train_manifest.json")
    val_manifest = os.path.join(output_dir, "val_manifest.json")
    test_manifest = os.path.join(output_dir, "test_manifest.json")

    with open(train_manifest, "w", encoding="utf-8") as f:
        json.dump(train_items, f, indent=2, ensure_ascii=False)
    with open(val_manifest, "w", encoding="utf-8") as f:
        json.dump(val_items, f, indent=2, ensure_ascii=False)
    with open(test_manifest, "w", encoding="utf-8") as f:
        json.dump(test_items, f, indent=2, ensure_ascii=False)

    logger.info("================ Apatani TTS Split =======================")
    logger.info(f"Total Audio Samples: {n_total}")
    logger.info(f"Train Split (80%):   {len(train_items)} samples ({sum(x['duration'] for x in train_items)/60:.1f} mins) -> {train_manifest}")
    logger.info(f"Val Split   (10%):   {len(val_items)} samples ({sum(x['duration'] for x in val_items)/60:.1f} mins) -> {val_manifest}")
    logger.info(f"Test Split  (10%):   {len(test_items)} samples ({sum(x['duration'] for x in test_items)/60:.1f} mins) -> {test_manifest}")
    logger.info("==========================================================")


if __name__ == "__main__":
    split_nyishi_translation()
    split_apatani_tts()
