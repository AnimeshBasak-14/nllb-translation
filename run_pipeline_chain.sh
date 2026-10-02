#!/usr/bin/env bash
set -e

echo "=== Pipeline Chain: Waiting for initial bidirectional model evaluation to complete ==="
while kill -0 2784782 2>/dev/null; do
    sleep 10
done
echo "=== Initial bidirectional model evaluation finished and saved! ==="

echo "=== Step 1: Starting 20-Epoch High-Fidelity Apatani TTS Training ==="
.venv/bin/python -u tts/train_tts.py \
    --train_manifest Apatani_TTS_Database/train_manifest.json \
    --val_manifest Apatani_TTS_Database/val_manifest.json \
    --test_manifest Apatani_TTS_Database/test_manifest.json \
    --output_dir ./best_apatani_tts \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 2 \
    --fp16 > train_tts_20ep.log 2>&1

echo "=== Step 1 Complete: 20-Epoch Apatani TTS model successfully trained and saved to ./best_apatani_tts ==="

echo "=== Step 2: Generating sample speech from updated TTS model ==="
.venv/bin/python tts/synthesize_tts.py \
    --text "Hopa Ngo nunumi lukoso, nunuka sangomi hena siiyo." \
    --output ./best_apatani_tts/sample_synthesized_20ep.wav

echo "=== Step 3: Starting Extended 3-Epoch Bidirectional Translation Training ==="
.venv/bin/python -u mte/train_bidirectional.py \
    --train_file data/nyishi_train.tsv \
    --val_file data/nyishi_val.tsv \
    --test_file data/nyishi_test.tsv \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --output_dir ./checkpoints_bidirectional \
    --best_model_dir ./best_bidirectional_model \
    --fp16 > train_bidirectional_3ep.log 2>&1

echo "=== All Pipeline Steps Successfully Completed! ==="
