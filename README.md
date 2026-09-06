# Qwen2.5-1.5B — Continued Pretraining with LoRA on an ML Textbook

Continued pretraining (non-instruction fine-tuning) of **Qwen2.5-1.5B** on a
400-page machine learning textbook to inject domain knowledge.
Trained on a single Kaggle T4 GPU using **transformers + PEFT + TRL**.

## 🔍 Overview
The goal is **domain-adaptive pretraining**: teaching a base language model the
content of a specific ML book by training next-token prediction on raw book text
(not instruction tuning).

## 🔧 Pipeline
1. **PDF extraction** — PyMuPDF, 400 pages → raw text
2. **Cleaning** — OCR artifact fixes (`1n[` → `In[`, `0ut[` → `Out[`), page-number removal via regex
3. **Paragraph reconstruction** — rebuild logical paragraphs, filter garbage (<100 chars)
4. **Tokenization + packing** — Qwen tokenizer, **EOS tokens** at paragraph boundaries, packed into exact **1024-token blocks** (zero padding waste)
5. **LoRA training** — adapters (r=32, α=64, dropout=0.1) on all attention + MLP projections
6. **Eval monitoring** — 90/10 split, per-epoch eval, best checkpoint auto-selected by `eval_loss`

## 📁 Repository Structure
| File | Purpose |
|---|---|
| `data_prep.py` | PDF extraction, cleaning, tokenization, packing |
| `train.py` | Model loading, LoRA config, SFTTrainer training |
| `demo_before_after.py` | A/B demo: base vs fine-tuned via `disable_adapter()` |
| `intro-to-ml.pdf` | Source textbook (dataset) |

## ⚙️ Key Technical Decisions
- **Base model (not Instruct)** → knowledge injection needs continued pretraining, not chat alignment
- **LoRA over full fine-tuning** → <1% trainable params, fits a single 16GB T4
- **Token-level packing + EOS** → 100% GPU utilization, clean document boundaries
- **Gradient checkpointing + accumulation** → 1024 seq length on limited VRAM
- **Best-checkpoint selection** → `load_best_model_at_end` on eval_loss prevents overfitting

## 📊 Training Config
| Hyperparameter | Value |
|---|---|
| LoRA r / alpha | 32 / 64 |
| Target modules | q,k,v,o + gate,up,down projections |
| Learning rate | 5e-5, cosine + 10% warmup |
| Epochs | 4 |
| Effective batch | 1 × 8 accumulation |
| Seq length | 1024 |
| Precision | fp16 |

## 📈 Results
- Train loss: **1.71 → ~1.3**, best epoch auto-selected
- Fine-tuned model completes ML sentences in the book's style (run `demo_before_after.py`)

## ▶️ How to Run
```bash
pip install -r requirements.txt
python data_prep.py           # build dataset
python train.py               # train LoRA adapter
python demo_before_after.py   # see before/after


🔗 Live training notebook with GPU logs: [https://www.kaggle.com/code/mohdmaaz036/fine-tuning-1]
