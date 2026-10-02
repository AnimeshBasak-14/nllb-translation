# Machine Translation Engine (MTE) – Nyishi <-> English

This directory contains the single unified bidirectional sequence-to-sequence neural translation pipeline for **Nyishi <-> English**.

## Overview
- **Base Architecture**: Meta NLLB-200 (`facebook/nllb-200-distilled-600M`)
- **Parameters**: 615 Million
- **Training Strategy**: Single parameter-shared model translating both ways simultaneously (`English -> Nyishi` and `Nyishi -> English`)
- **Vocabulary**: 256,206 subwords (FLORES-200 language code routing)
- **Checkpoints**: Saved to `best_bidirectional_model/`

## Key Files
- `train_bidirectional.py`: Unified bidirectional training pipeline with symmetric data pairing and evaluation.
- `train.py`: Single-direction baseline training engine.
- `test.py`: Standalone evaluation benchmark script for SacreBLEU and ChrF++ (word_order=2).
- `evaluation_results.json`: Output metrics and sample translation logs.

## Checkpoint Architecture & Weight Reassembly
- The model is sharded into 18 `.safetensors` files in `best_bidirectional_model/` to comply with GitHub's 100 MB per-file limit without Git LFS.
- Shards `00002` through `00018` are each $\le 80.1$ MB.
- Shard 1 (`model-00001-of-00018.safetensors`, 1.0 GB) is chunked into 13 `.part_*` files (`part_00` to `part_12`, each 80 MB).
- **Auto-Reassembly**: `mte/test.py`, `demo.py`, and `generate_hackathon_submission.py` automatically detect if shard 1 is missing and reassemble it in $<1.5$ seconds upon execution.
- **Manual Reassembly**:
  ```bash
  cat best_bidirectional_model/model-00001-of-00018.safetensors.part_* > best_bidirectional_model/model-00001-of-00018.safetensors
  ```

## Benchmarks
- **Validation SacreBLEU**: **22.64**
- **Validation ChrF++**: **42.47**
- **Test SacreBLEU (Held-Out)**: **22.81**
- **Test ChrF++ (Held-Out)**: **42.29**
- **Validation Cross-Entropy Loss**: **1.7940**
- **Test Cross-Entropy Loss**: **1.8007**

## Execution Commands

### Training the Bidirectional Model
```bash
python mte/train_bidirectional.py \
    --train_file data/nyishi_train.tsv \
    --val_file data/nyishi_val.tsv \
    --test_file data/nyishi_test.tsv \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --output_dir ./checkpoints_bidirectional \
    --best_model_dir ./best_bidirectional_model \
    --fp16
```

### Evaluating on Test Set
```bash
python mte/test.py \
    --model_dir ./best_bidirectional_model \
    --data_file data/nyishi_test.tsv \
    --src_lang eng_Latn \
    --tgt_lang hin_Deva \
    --batch_size 16 \
    --output_file mte/evaluation_results.json
```
