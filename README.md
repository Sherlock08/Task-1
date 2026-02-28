# Task 1: Comparative LLM Fine-Tuning for Medical Question Answering

**Module:** 7043SCN — Generative AI and Reinforcement Learning

This project compares **baseline (zero-shot)** vs **LoRA fine-tuned** performance of Flan-T5-Small on medical question answering. It trains on the Medical Meadow MedQA dataset and evaluates using ROUGE and BLEU metrics.

---

## Overview

- **Base Model:** [google/flan-t5-small](https://huggingface.co/google/flan-t5-small)
- **Dataset:** [medalpaca/medical_meadow_medqa](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) (fallback: PubMed QA)
- **Fine-tuning:** LoRA (Low-Rank Adaptation) with PEFT
- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L, BLEU

---

## Setup

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install transformers datasets peft torch evaluate rouge_score sacrebleu matplotlib
```

---

## Usage

### Full pipeline (baseline + LoRA training + evaluation)

```bash
python train_evaluate.py
```

### Stage-specific runs

```bash
# Evaluate baseline only (zero-shot)
python train_evaluate.py --stage baseline

# Train LoRA only (loads existing baseline)
python train_evaluate.py --stage lora
```

### Generate plots

After training completes, generate publication-quality figures:

```bash
python generate_plots.py
```

Plots are saved to `outputs/plots/`:
- `fig1_metric_comparison.png` — Baseline vs LoRA metric comparison
- `fig2_training_curves.png` — Training loss and validation ROUGE-L
- `fig3_parameter_efficiency.png` — Frozen vs trainable parameters
- `fig4_improvement_delta.png` — Performance improvement over baseline

---

## Project Structure

| File | Description |
|------|-------------|
| `train_evaluate.py` | Main script: loads data, evaluates baseline, trains LoRA, saves results |
| `generate_plots.py` | Generates 4 IEEE-style figures from `outputs/results.json` |
| `requirements.txt` | Python dependencies |

---

## Configuration

Key parameters in `train_evaluate.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `google/flan-t5-small` | Base seq2seq model |
| `train_size` | 2000 | Training samples |
| `val_size` | 400 | Validation samples |
| `test_size` | 400 | Test samples |
| `lora_r` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA alpha |
| `batch_size` | 8 | Per-device batch size |
| `num_epochs` | 3 | Training epochs |

---

## Outputs

- **`outputs/results.json`** — Full results: baseline metrics, LoRA training stats, evaluation metrics, improvement deltas, qualitative samples
- **`outputs/lora_finetuned/`** — Saved LoRA model and tokenizer
- **`outputs/plots/`** — Generated figures

---

## Hardware Note

The code is configured for CPU training (Intel Core i5, 8 GB RAM). Full fine-tuning (FFT) requires >12 GB VRAM for optimizer states; LoRA enables efficient training on CPU, though it is slower than GPU.
