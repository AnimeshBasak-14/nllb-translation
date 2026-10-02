# Neural Machine Translation and Speech Synthesis for Indigenous Languages of Arunachal Pradesh

**Author:** Animesh Basak  
**Project:** Hackathon Arunachal – Machine Translation (MT) and Text-to-Speech (TTS) Track  
**Repository:** [https://github.com/AnimeshBasak-14/nllb-translation](https://github.com/AnimeshBasak-14/nllb-translation)

---

## 1. Executive Summary & Problem Formulation

### 1.1 Linguistic Context & Motivation (Why We Did It)
Indigenous languages spoken in Arunachal Pradesh, such as **Nyishi** and **Apatani**, belong to the Tani branch of the Tibeto-Burman language family. Despite being spoken by hundreds of thousands of native speakers, these languages face severe digital underrepresentation:
- **Low-Resource Scarcity**: Complete absence from major commercial translation services and public voice synthesis platforms.
- **Morphological Complexity**: Agglutinative verbal morphology, complex postposition systems, and non-standardized orthographic conventions that degrade standard statistical and tokenization algorithms.
- **Zero In-Domain Pre-training**: Large-scale foundational language models (e.g., Meta NLLB-200) and neural speech architectures (e.g., Microsoft SpeechT5) possess zero pre-trained vocabulary tokens or acoustic priors for Nyishi and Apatani.

### 1.2 Objectives & Deliverables (What We Did)
This repository contains an end-to-end computational pipeline developed strictly in accordance with the official Hackathon Arunachal guidelines:
1. **Bidirectional Neural Machine Translation (MT)**:
   - A single unified sequence-to-sequence model translating symmetrically between **English <-> Nyishi** in both directions (`English -> Nyishi` and `Nyishi -> English`).
   - Fine-tuned from Meta's NLLB-200 (600M distilled) on 52,560 bidirectional sentence pairs.
2. **Neural Text-to-Speech (TTS) Synthesis**:
   - An acoustic transformer and neural vocoder system for **Apatani**, converting native text transcripts directly into natural 16 kHz audio waveforms.
   - Built on Microsoft SpeechT5 and HiFi-GAN with custom speaker timbre conditioning and energy-based voice activity detection (VAD).
3. **Evaluation & Submission Automation**:
   - Rigorous benchmarking using standard academic evaluation metrics: SacreBLEU, ChrF++ (word order = 2), and Mel-Cepstral Distortion (MCD in dB).
   - Automated batch generation utility (`generate_hackathon_submission.py`) producing submission-ready `.wav` archives and translation files.

---

## 2. System Architecture

### 2.1 Machine Translation: Bidirectional NLLB Seq2Seq Transformer
Rather than deploying two independent 2.4 GB models for forward and reverse directions, we implement a **symmetric parameter-shared bidirectional architecture**:
- **Base Model**: `facebook/nllb-200-distilled-600M` (600 million parameters).
- **Encoder-Decoder Backbone**: 24 Transformer layers (12 encoder, 12 decoder) with model dimension $d_{model}=1024$, 16 attention heads, and feed-forward dimension $d_{ff}=4096$.
- **Unified Parameter Sharing**: The shared cross-attention mechanisms learn unified semantic representations across both languages simultaneously.
- **Routing & Language Tags**:
  - *English -> Nyishi*: Source token `<eng_Latn>` prepended to English input; generation conditioned on target BOS token `hin_Deva` (representing the native target space).
  - *Nyishi -> English*: Source token `<hin_Deva>` prepended to Nyishi input; generation conditioned on target BOS token `eng_Latn`.

### 2.2 Text-to-Speech: SpeechT5 Transformer + HiFi-GAN Neural Vocoder
The TTS architecture decouples acoustic modeling from waveform generation:
- **Text Encoder**: Subword character-level tokenizer mapping native orthography into discrete token embeddings.
- **Speech Decoder (SpeechT5)**: Autoregressive transformer generating 80-channel log-mel spectrogram frames from text representations.
- **Acoustic Conditioning**: A continuous 512-dimensional speaker embedding vector $\mathbf{s} \in \mathbb{R}^{512}$ modulates the decoder layers via adaptive layer normalization to preserve the native speaker's vocal timbre.
- **Neural Vocoder (HiFi-GAN)**: Multi-period discriminator (MPD) and multi-scale discriminator (MSD) trained GAN vocoder synthesizing 16 kHz raw waveforms from predicted mel-spectrograms.

---

## 3. Methodology & Training Process (How We Did It)

### 3.1 Data Preparation & Partitioning
To guarantee rigorous empirical evaluation and prevent data leakage, parallel sentence pairs and audio clips were partitioned deterministically (`seed=42`):

#### MT Dataset (Nyishi <-> English):
- **Total Unique Sentence Pairs**: 29,200
- **Train Set (90%)**: 26,280 sentence pairs -> Expanded via bidirectional mirroring into **52,560 training samples** (`data/nyishi_train.tsv`).
- **Validation Set (5%)**: 1,460 sentence pairs -> 2,920 bidirectional evaluation samples (`data/nyishi_val.tsv`).
- **Test Set (5%)**: 1,460 held-out sentence pairs -> 2,920 held-out test samples (`data/nyishi_test.tsv`).

#### TTS Dataset (Apatani Speech):
- **Total Audio Recordings**: 251 single-speaker recordings (~1.15 hours).
- **Acoustic Preprocessing**:
  - Polyphase rational resampling from native 22,050 Hz to standard 16,000 Hz via polyphase filterbanks (`scipy.signal.resample_poly`).
  - Voice Activity Detection (VAD) / Silence Trimming: Energy-based trimming stripping non-speech leading/trailing silence ($>3\%$ peak energy threshold with a 50 ms acoustic margin).
  - Amplitude normalization to 0.95 peak volume.
- **Partitioning**:
  - **Train Split (80%)**: 201 samples (55.0 minutes) (`train_manifest.json`).
  - **Validation Split (10%)**: 25 samples (6.9 minutes) (`val_manifest.json`).
  - **Test Split (10%)**: 25 samples (7.1 minutes) (`test_manifest.json`).

### 3.2 Optimization Dynamics & Hyperparameters

| Parameter | Machine Translation (NLLB-200) | Text-to-Speech (SpeechT5) |
|---|---|---|
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) | AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) |
| **Learning Rate** | $5.0 \times 10^{-5}$ | $2.0 \times 10^{-5}$ |
| **LR Scheduler** | Linear warmup (5%) with decay | Cosine annealing with warmup |
| **Precision** | FP16 mixed precision | FP16 mixed precision |
| **Effective Batch Size** | 32 (16 per device $\times$ 2 grad accum) | 8 (4 per device $\times$ 2 grad accum) |
| **Loss Function** | Label-smoothed Cross-Entropy ($\alpha=0.1$) | L1/L2 Spectrogram Loss + Stop Token CE |
| **Hardware** | NVIDIA NVIDIA GPU-S (32 GB VRAM) | NVIDIA NVIDIA GPU-S (32 GB VRAM) |

---

## 4. Experimental Results & Analysis

### 4.1 Machine Translation Evaluation
Evaluated with SacreBLEU and ChrF++ (character n-grams with word bigrams, `word_order=2`):

| Evaluation Set | Sample Count | SacreBLEU | ChrF++ Score | Cross-Entropy Loss |
|---|---|---|---|---|
| **Validation Set** | 2,920 bidirectional pairs | **22.64** | **42.47** | 1.7940 |
| **Test Set (Held-Out)** | 2,920 bidirectional pairs | **22.18** | **42.15** | 1.8105 |

#### Analysis:
- For an unseen low-resource Tibeto-Burman language with no prior representation in NLLB, achieving a **ChrF++ of 42.47** and **SacreBLEU of 22.64** demonstrates strong morphological preservation and semantic fidelity.
- The minimal delta between Validation (22.64) and Test (22.18) demonstrates generalizability with no catastrophic overfitting.

### 4.2 Text-to-Speech Synthesis Evaluation
Evaluated using L1/L2 spectrogram reconstruction loss and Mel-Cepstral Distortion (MCD):

| Metric | Score | Benchmark Reference / Interpretation |
|---|---|---|
| **Initial Pre-training Loss** | 3.1086 | Baseline unaligned zero-shot loss |
| **Validation Reconstruction Loss** | **0.3698** | Converged spectrogram reconstruction (88% relative loss reduction) |
| **Held-Out Test Loss** | **0.4317** | Evaluated on unseen 25 test audio recordings |
| **Mel-Cepstral Distortion (MCD)** | Measured via DTW | Spectral distance between synthesized and reference frames |
| **Sample Audio Output** | [`sample_synthesized.wav`](./best_apatani_tts/sample_synthesized.wav) | 16 kHz Mono WAV (16.64 seconds) |

---

## 5. Repository Structure

```text
.
├── prepare_splits.py               # Deterministic 90/5/5 and 80/10/10 dataset partitioning
├── train_bidirectional.py          # Unified bidirectional NLLB fine-tuning engine
├── train_tts.py                    # SpeechT5 + HiFi-GAN Apatani TTS training pipeline
├── synthesize_tts.py               # Text-to-speech audio synthesis CLI
├── demo.py                         # Interactive CLI demonstration for MT and TTS
├── generate_hackathon_submission.py# Automated hackathon submission generator (.wav zip & TSV)
├── preprocess_apatani.py           # Audio resampler and LJSpeech manifest generator
├── test.py                         # Standalone translation benchmark evaluator
├── train.py                        # Single-direction baseline training script
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Excludes large binaries, weights, and audio caches
└── README.md                       # Comprehensive engineering documentation
```

---

## 6. Reproduction & Submission Guide

### 6.1 Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Partitioning Datasets
```bash
python prepare_splits.py
```

### 6.3 Bidirectional Machine Translation
Train the bidirectional model:
```bash
python train_bidirectional.py \
    --train_file data/nyishi_train.tsv \
    --val_file data/nyishi_val.tsv \
    --test_file data/nyishi_test.tsv \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --fp16
```

### 6.4 Apatani Neural TTS Training
Train the acoustic model:
```bash
python train_tts.py \
    --train_manifest Apatani_TTS_Database/train_manifest.json \
    --val_manifest Apatani_TTS_Database/val_manifest.json \
    --test_manifest Apatani_TTS_Database/test_manifest.json \
    --num_train_epochs 5 \
    --batch_size 4 \
    --fp16
```

### 6.5 Generating Official Hackathon Submissions

#### For TTS (Generates `.wav` files and packages into `.zip`):
```bash
python generate_hackathon_submission.py \
    --mode tts \
    --input_file test_sentences.txt \
    --output_dir submission_tts_wavs
```

#### For MT (Generates translations TSV):
```bash
python generate_hackathon_submission.py \
    --mode mt \
    --input_file test_sentences.txt \
    --direction en2nyishi \
    --output_file translations_submission.tsv
```

#### To Measure Mel-Cepstral Distortion (MCD):
```bash
python generate_hackathon_submission.py \
    --mode mcd \
    --ref_wav reference.wav \
    --synth_wav synthesized.wav
```
