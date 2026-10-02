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
├── train.py                 # Main training, evaluation, and checkpointing script
├── requirements.txt         # Project dependencies (transformers, evaluate, sacrebleu, etc.)
├── .gitignore               # Machine learning & Python ignores (data, checkpoints, venvs)
├── README.md                # Project documentation and usage guide
└── nyishi_train_cleaned.tsv # (Local dataset, excluded from git)
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

## 📊 Dataset Format

The pipeline natively supports `.tsv`, `.csv`, `.json`, `.parquet`, or Hugging Face Hub datasets. 

For TSV/CSV, the file should contain source and target columns (e.g., `nyishi_train_cleaned.tsv`):
```text
english	nyishi
Adam, Seth, Enosh;	Adam, Set, Inos;
Kenan, Mahalalel, Jared;	Kenan, Mahalalel, Jared;
...
```

The script automatically splits any provided dataset into:
- **95%** Training
- **5%** Validation

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

## 📈 Evaluation Metrics

During each evaluation epoch, the pipeline decodes model predictions and computes:
1. **SacreBLEU (`bleu`)**: Standard corpus-level BLEU score.
2. **ChrF++ (`chrf++`)**: Character n-gram F-score augmented with word bigrams (`word_order=2`), especially effective for morphologically rich and low-resource languages.

The best checkpoint based on the selected metric (`--metric_for_best_model`) is tracked and automatically loaded at the end of training.

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
