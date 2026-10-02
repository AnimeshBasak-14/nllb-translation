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
    --model_name_or_path openai/whisper-small \
    --train_manifest Apatani_TTS_Database/train_manifest.json \
    --val_manifest Apatani_TTS_Database/val_manifest.json \
    --num_train_epochs 5 \
    --fp16
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

On the Nyishi translation dataset (`nyishi_train_cleaned.tsv` with strict 95% train / 5% validation split):

| Metric | Score | Details |
|---|---|---|
| **Training Loss** | `2.02` (down from `7.22`) | 1 Epoch fine-tuning (1,734 steps) on NVIDIA GPU |
| **Validation SacreBLEU** | **18.18** | Full 5% validation set (1,461 samples) |
| **Validation ChrF++** | **42.12** | Character n-grams with word order = 2 |
| **Test Sample SacreBLEU** | **17.60** | Independent sample evaluation |
| **Test Sample ChrF++** | **42.89** | Independent sample evaluation |

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
