# Neural Text-to-Speech Engine (TTS) – Apatani

This directory contains the acoustic modeling and speech synthesis pipeline for the indigenous **Apatani** language.

## Overview
- **Acoustic Model**: Microsoft SpeechT5 (`microsoft/speecht5_tts`)
- **Neural Vocoder**: HiFi-GAN (`microsoft/speecht5_hifigan`)
- **Audio Specifications**: 16,000 Hz, Single-Channel (Mono), 16-bit PCM WAV
- **Speaker Adaptation**: Continuous 512-dimensional speaker timbre embedding vector $\mathbf{s} \in \mathbb{R}^{512}$
- **Audio Preprocessing**: Energy-based Voice Activity Detection (VAD) silence trimming, polyphase rational resampling, and reduction factor ($r=2$) frame alignment.

## Key Files
- `train_tts.py`: Fine-tuning pipeline for SpeechT5 acoustic model with dynamic mel padding, VAD trimming, and checkpointing.
- `synthesize_tts.py`: Command-line interface for synthesizing arbitrary Apatani text into spoken WAV audio.
- `preprocess_apatani.py`: Audio preprocessing, header inspection, rational resampling (22.05 kHz -> 16 kHz), and LJSpeech format metadata generation.

## Benchmarks
- **Zero-Shot Initial Loss**: 3.1086
- **Validation Spectrogram Loss (20 Epochs)**: **0.3026** *(90.3% relative error reduction)*
- **Held-Out Test Loss (25 clips)**: **0.3122**
- **Synthesized Audio Proof**: [`best_apatani_tts/sample_synthesized_20ep.wav`](../best_apatani_tts/sample_synthesized_20ep.wav)

## Execution Commands

### Training the TTS Acoustic Model
```bash
python tts/train_tts.py \
    --train_manifest Apatani_TTS_Database/train_manifest.json \
    --val_manifest Apatani_TTS_Database/val_manifest.json \
    --test_manifest Apatani_TTS_Database/test_manifest.json \
    --output_dir ./best_apatani_tts \
    --batch_size 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 20 \
    --gradient_accumulation_steps 2 \
    --fp16
```

### Direct Audio Synthesis
```bash
python tts/synthesize_tts.py \
    --text "Hopa Ngo nunumi lukoso, nunuka sangomi hena siiyo." \
    --output ./synthesized_apatani.wav \
    --model_dir ./best_apatani_tts
```
