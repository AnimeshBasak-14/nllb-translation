# Meta NLLB Fine-Tuning & Evaluation Pipeline

A robust, production-grade PyTorch and Hugging Face `transformers` pipeline to fine-tune and evaluate Meta's **NLLB (No Language Left Behind)** sequence-to-sequence translation models (default: `facebook/nllb-200-distilled-600M`).

---

## 🚀 Key Features

- **Model Flexibility**: Defaults to `facebook/nllb-200-distilled-600M`, easily configurable to any NLLB-200 variant (e.g., `facebook/nllb-200-1.3B`, `facebook/nllb-200-3.3B`).
- **Strict Data Splitting**: Guarantees a **95% training** and **5% validation** split with deterministic seeding and data hygiene (handling null/empty translation pairs).
- **Dual Validation Metrics**: Computes **SacreBLEU** and **ChrF++** (character n-grams + word bigrams, `word_order=2`) during validation loops via Hugging Face `evaluate`.
- **Automatic Best-Model Checkpointing**: Utilizes `Seq2SeqTrainer` with `load_best_model_at_end=True` and `metric_for_best_model="chrf++"` (or `"bleu"`). Automatically exports the best-performing model and tokenizer for immediate reuse.
- **Hardware Acceleration**: Automatic mixed-precision training (`fp16` / `bf16`) with dynamic padding for memory efficiency.
- **Custom Language Token Support**: Automatically recognizes or registers custom language codes if fine-tuning on low-resource languages not present in the default vocabulary.

---

## 📁 Project Structure

```text
.
├── train.py                 # Single-language translation pipeline
├── train_multilingual.py    # Joint multilingual / multi-language translation pipeline
├── test.py                  # Standalone translation evaluation & inference script
├── preprocess_apatani.py    # Apatani speech validation & LJSpeech manifest generator
├── train_asr.py             # Whisper ASR fine-tuning pipeline with WER & CER
├── requirements.txt         # Pinned project dependencies
├── .gitignore               # Ignores large model weights, audio, data & archives
├── README.md                # Project documentation and CLI usage guide
├── evaluation_results.json  # Exported test benchmark results
└── TSV data/                # Parallel translation datasets (Adi, Apatani, Galo, Nyishi, Tagin)
```

---

## 🛠️ Installation & Setup

### 1. Create a Virtual Environment

Using standard Python `venv`:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or using `uv` (fast package manager):
```bash
uv venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Supported Datasets

The repository includes support for 5 major indigenous languages of Arunachal Pradesh:

| Language | Dataset File | Parallel Sentence Pairs | Script |
|---|---|---|---|
| **Adi** | `TSV data/adi_train.tsv` | **28,766** | Latin |
| **Apatani** | `TSV data/apatani_train.tsv` | **16,811** | Latin |
| **Galo** | `TSV data/galo_train.tsv` | **6,450** | Latin |
| **Nyishi** | `TSV data/nyishi_train.tsv` | **29,406** | Latin |
| **Tagin** | `TSV data/tagin_train.tsv` | **15,975** | Latin |
| **Total** | | **97,408** | |

Additionally, an **Apatani Speech & TTS Database** is supported (`Apatani_TTS_Database/`):
- **251 audio clips** (~1.15 hours) at 22,050 Hz Mono with paired transcriptions in `sentences.txt`.

---

## ⚡ Running the Pipelines

### 1. Unified Multilingual Translation (Any / All 5 Languages)

Train on all 5 languages simultaneously with language tags:
```bash
python train_multilingual.py \
    --language all \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --fp16
```

Or train specifically on a single language (e.g., Apatani):
```bash
python train_multilingual.py \
    --language apatani \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --fp16
```

### 2. Apatani Speech Preprocessing & LJSpeech Manifest Generation

Preprocess audio, normalize transcripts, and create standard LJSpeech metadata:
```bash
python preprocess_apatani.py \
    --data_dir Apatani_TTS_Database \
    --val_ratio 0.15
```

### 3. Apatani Automatic Speech Recognition (Whisper Fine-Tuning)

Fine-tune OpenAI Whisper with Word Error Rate (WER) and Character Error Rate (CER) tracking:
```bash
python train_asr.py \
    --model_name_or_path openai/whisper-base \
    --train_manifest Apatani_TTS_Database/train_manifest.json \
    --val_manifest Apatani_TTS_Database/val_manifest.json \
    --num_train_epochs 5 \
    --fp16
```

### 4. Unified Multilingual & Speech Demo CLI

Test both fine-tuned translation and speech-to-text with a single command:
```bash
# Run full demo (translates sample sentence & transcribes sample audio):
python demo_multilingual_speech.py --mode demo

# Translate custom English text to Apatani / Nyishi:
python demo_multilingual_speech.py --mode translate --text "In the beginning God created the heaven and the earth."

# Transcribe any Apatani audio file (.wav):
python demo_multilingual_speech.py --mode transcribe --audio Apatani_TTS_Database/wav/APT-0006.wav
```

---

## ⚡ Running the Training Pipeline

### Quick Start (Default Settings)

To train on your dataset with default settings:
```bash
python train.py \
    --data_file nyishi_train_cleaned.tsv \
    --source_column english \
    --target_column nyishi \
    --src_lang eng_Latn \
    --tgt_lang hin_Deva \
    --output_dir ./checkpoints \
    --best_model_dir ./best_model \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --fp16
```

### Key Command-Line Options

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model_name_or_path` | `str` | `facebook/nllb-200-distilled-600M` | Hugging Face model repository or local directory |
| `--data_file` | `str` | `nyishi_train_cleaned.tsv` | Path to local TSV/CSV/JSON dataset |
| `--source_column` | `str` | `english` | Column name for source language text |
| `--target_column` | `str` | `nyishi` | Column name for target language text |
| `--src_lang` | `str` | `eng_Latn` | NLLB source language code |
| `--tgt_lang` | `str` | `hin_Deva` | NLLB target language code |
| `--val_split` | `float` | `0.05` | Validation split ratio (strictly 5% validation, 95% training) |
| `--metric_for_best_model` | `str` | `chrf++` | Metric used to select best checkpoint (`chrf++` or `bleu`) |
| `--num_train_epochs` | `float` | `3.0` | Number of training epochs |
| `--per_device_train_batch_size` | `int` | `8` | Training batch size per device |
| `--learning_rate` | `float` | `2e-5` | Initial learning rate |
| `--best_model_dir` | `str` | `./best_model` | Destination directory for the saved best model |
| `--fp16` | flag | auto | Enable 16-bit mixed precision (GPU) |

---

## 📈 Benchmark & Evaluation Results

### 1. Translation Benchmarks (Meta NLLB-200-Distilled-600M)
Strict 95% train / 5% validation split:

| Task / Language | Train Split | Validation Split | Val SacreBLEU | Val ChrF++ | Test BLEU |
|---|---|---|---|---|---|
| **English -> Nyishi** | 27,935 pairs | 1,471 pairs | **18.18** | **42.12** | **17.60** |
| **English -> Apatani** | 15,970 pairs | 841 pairs | **13.18** | **37.63** | **13.25** |

### 2. Speech-to-Text Benchmark (OpenAI Whisper-Base)
Evaluated on unseen Apatani speech validation split (38 audio recordings):

| Metric | Score | Note |
|---|---|---|
| **Character Error Rate (CER)** | **7.33%** | Highly accurate character-level transcription |
| **Word Error Rate (WER)** | **34.53%** | Low-resource indigenous language recognition |
| **Validation Loss** | **0.547** | Down from 3.8+ baseline pre-training loss |

Detailed translations and evaluation scores are exported to [`evaluation_results.json`](./evaluation_results.json).

---

## 🧪 Testing & Standalone Evaluation

You can run automated testing and translation sample generation using `test.py`:

```bash
python test.py \
    --model_dir ./best_model \
    --data_file nyishi_train_cleaned.tsv \
    --source_column english \
    --target_column nyishi \
    --src_lang eng_Latn \
    --tgt_lang hin_Deva \
    --max_samples 100 \
    --output_file evaluation_results.json
```

---

## 🔄 Using the Fine-Tuned Model for Inference

Once training is complete, the best model and tokenizer are saved in `--best_model_dir` (`./best_model`). You can load and use them directly:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

input_text = "Noah, Shem, Ham, and Japheth."
inputs = tokenizer(input_text, return_tensors="pt")

# Generate translation using target language forced BOS token
outputs = model.generate(
    **inputs,
    forced_bos_token_id=model.config.forced_bos_token_id,
    max_length=128
)

translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Translation:", translated_text)
```

---

## 📜 License

This project is licensed under the Apache 2.0 License.
