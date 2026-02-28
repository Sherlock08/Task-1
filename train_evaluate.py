"""
Task 1: Comparative LLM Fine-Tuning for Medical Question Answering
Module: 7043SCN — Generative AI and Reinforcement Learning

Usage:
    pip install transformers datasets peft torch evaluate rouge_score sacrebleu
    python train_evaluate.py          # full pipeline
    python train_evaluate.py --stage baseline
    python train_evaluate.py --stage lora
"""

import os, json, time, argparse, logging, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq, set_seed, EarlyStoppingCallback,
)
import transformers as _tr
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Detect transformers version: 'evaluation_strategy' renamed to 'eval_strategy' in >=4.46
_TR_VER = tuple(int(x) for x in _tr.__version__.split(".")[:2])
EVAL_KEY = "eval_strategy" if _TR_VER >= (4, 46) else "evaluation_strategy"
logger.info("transformers %s — using kwarg '%s'", _tr.__version__, EVAL_KEY)

# ─────────────────────────── CONFIG ──────────────────────────────────────────
CONFIG = {
    "model_name":       "google/flan-t5-small",
    "max_input_length": 256,
    "max_target_length":128,
    "dataset_name":     "medalpaca/medical_meadow_medqa",
    "train_size": 2000, "val_size": 400, "test_size": 400,
    "lora_r": 8, "lora_alpha": 16, "lora_dropout": 0.1,
    "target_modules": ["q", "v"],
    "batch_size": 8, "learning_rate": 3e-4, "num_epochs": 3,
    "warmup_steps": 100, "weight_decay": 0.01, "seed": 42,
    "output_dir": "./outputs",
    "lora_dir":   "./outputs/lora_finetuned",
    "results_file":"./outputs/results.json",
}
set_seed(CONFIG["seed"])

# ─────────────────────────── DATA ────────────────────────────────────────────
def load_and_prepare_dataset():
    logger.info("Loading: %s", CONFIG["dataset_name"])
    try:
        dataset = load_dataset(CONFIG["dataset_name"], split="train")
    except Exception as e:
        logger.warning("Fallback to pubmed_qa (%s)", e)
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
        dataset = dataset.map(lambda x: {"input": x.get("question",""), "output": x.get("long_answer","")})

    # Normalise column names
    rename = {}
    if "question" in dataset.column_names and "input"  not in dataset.column_names: rename["question"] = "input"
    if "answer"   in dataset.column_names and "output" not in dataset.column_names: rename["answer"]   = "output"
    if rename: dataset = dataset.rename_columns(rename)

    dataset = dataset.filter(lambda x: bool(str(x.get("input","")).strip()) and bool(str(x.get("output","")).strip()))
    total   = CONFIG["train_size"] + CONFIG["val_size"] + CONFIG["test_size"]
    dataset = dataset.shuffle(seed=CONFIG["seed"]).select(range(min(total, len(dataset))))

    n = CONFIG["train_size"]
    m = CONFIG["val_size"]
    train_ds = dataset.select(range(n))
    val_ds   = dataset.select(range(n, n+m))
    test_ds  = dataset.select(range(n+m, n+m+CONFIG["test_size"]))
    logger.info("Train: %d | Val: %d | Test: %d", len(train_ds), len(val_ds), len(test_ds))
    return train_ds, val_ds, test_ds


def tokenize_dataset(dataset, tokenizer, prefix="Answer the medical question: "):
    def preprocess(examples):
        inputs  = [prefix + str(q) for q in examples["input"]]
        targets = [str(t) for t in examples["output"]]
        mi = tokenizer(inputs, max_length=CONFIG["max_input_length"], truncation=True, padding="max_length")
        lb = tokenizer(text_target=targets, max_length=CONFIG["max_target_length"], truncation=True, padding="max_length")
        lb["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in lab] for lab in lb["input_ids"]]
        mi["labels"] = lb["input_ids"]
        return mi
    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


# ─────────────────────────── METRICS ─────────────────────────────────────────
rouge_metric = evaluate.load("rouge")
bleu_metric  = evaluate.load("sacrebleu")


def score_strings(preds, refs):
    """Compute ROUGE + BLEU directly from string lists."""
    preds = [str(p).strip() for p in preds]
    refs  = [str(r).strip() for r in refs]
    rouge = rouge_metric.compute(predictions=preds, references=refs, use_stemmer=True)
    bleu  = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    return {
        "rouge1": round(rouge["rouge1"], 4),
        "rouge2": round(rouge["rouge2"], 4),
        "rougeL": round(rouge["rougeL"], 4),
        "bleu":   round(bleu["score"],   4),
    }


def make_trainer_metric_fn(tokenizer):
    """Closure for Seq2SeqTrainer's compute_metrics (receives token ID arrays)."""
    def fn(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple): preds = preds[0]
        preds  = np.where(preds  != -100, preds,  tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dp = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        dl = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return score_strings(dp, dl)
    return fn


# ─────────────────────────── INFERENCE HELPER ────────────────────────────────
def run_inference(model, tokenizer, test_ds, n_samples=100):
    model.eval()
    all_preds, all_labels = [], []
    start = time.time()
    for i in range(0, min(n_samples, len(test_ds)), CONFIG["batch_size"]):
        batch  = test_ds[i : i + CONFIG["batch_size"]]
        inputs = tokenizer(
            ["Answer the medical question: " + str(q) for q in batch["input"]],
            return_tensors="pt", max_length=CONFIG["max_input_length"],
            truncation=True, padding=True,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=CONFIG["max_target_length"], num_beams=2)
        all_preds.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        all_labels.extend([str(x) for x in batch["output"]])
    elapsed = round(time.time() - start, 2)
    metrics = score_strings(all_preds, all_labels)
    metrics["inference_time_s"] = elapsed
    return metrics, all_preds[:5], all_labels[:5]


# ─────────────────────────── BASELINE ────────────────────────────────────────
def evaluate_baseline(test_ds, tokenizer, model):
    logger.info("=== EVALUATING BASELINE (zero-shot) ===")
    metrics, preds, labels = run_inference(model, tokenizer, test_ds)
    logger.info("Baseline: %s", metrics)
    return metrics, preds, labels


# ─────────────────────────── LoRA TRAINING ───────────────────────────────────
def train_lora(train_tok, val_tok, tokenizer, base_model):
    logger.info("=== TRAINING LoRA ===")
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"], bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    os.makedirs(CONFIG["lora_dir"], exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=CONFIG["lora_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
        predict_with_generate=True,
        generation_max_length=CONFIG["max_target_length"],
        **{EVAL_KEY: "epoch"},          # ← version-safe key
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        logging_steps=50,
        fp16=False,
        report_to="none",
        seed=CONFIG["seed"],
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_tok, eval_dataset=val_tok,
        **{"processing_class" if _TR_VER >= (5, 0) else "tokenizer": tokenizer},
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100),
        compute_metrics=make_trainer_metric_fn(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    start = time.time()
    result = trainer.train()
    elapsed = time.time() - start
    trainer.save_model(CONFIG["lora_dir"])
    tokenizer.save_pretrained(CONFIG["lora_dir"])

    # Collect per-epoch history
    history = {"epoch": [], "train_loss": [], "eval_rougeL": []}
    for entry in trainer.state.log_history:
        if "eval_rougeL" in entry:
            history["epoch"].append(round(entry.get("epoch", 0), 1))
            history["eval_rougeL"].append(entry["eval_rougeL"])
        if "loss" in entry and "eval_loss" not in entry:
            history["train_loss"].append(entry["loss"])

    train_metrics = {
        "train_loss":       round(result.training_loss, 4),
        "train_time_s":     round(elapsed, 2),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params":     sum(p.numel() for p in model.parameters()),
    }
    logger.info("LoRA train complete: %s", train_metrics)
    return model, train_metrics, history


# ─────────────────────────── SAVE + PRINT ────────────────────────────────────
def save_and_print(baseline_m, train_m, eval_m, history, bl_preds, lo_preds, labels):
    imp = {
        "rouge1_delta": round(eval_m["rouge1"] - baseline_m["rouge1"], 4),
        "rouge2_delta": round(eval_m["rouge2"] - baseline_m["rouge2"], 4),
        "rougeL_delta": round(eval_m["rougeL"] - baseline_m["rougeL"], 4),
        "bleu_delta":   round(eval_m["bleu"]   - baseline_m["bleu"],   4),
    }
    results = {
        "config": CONFIG,
        "hardware": {
            "device": "CPU (Intel Core i5, 8 GB RAM)",
            "reason_no_fft": "FFT requires >12 GB VRAM for optimizer states; CPU training ~50x slower than GPU.",
        },
        "baseline":              baseline_m,
        "lora_training":         train_m,
        "lora_evaluation":       eval_m,
        "lora_training_history": history,
        "improvement":           imp,
        "qualitative_samples": [
            {"question": labels[i], "baseline_answer": bl_preds[i], "lora_answer": lo_preds[i]}
            for i in range(min(5, len(labels)))
        ],
    }
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    with open(CONFIG["results_file"], "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*65)
    print(f"{'METRIC':<15} {'BASELINE':>12} {'LoRA':>12} {'DELTA':>12}")
    print("="*65)
    for m in ["rouge1", "rouge2", "rougeL", "bleu"]:
        bv = baseline_m[m]; lv = eval_m[m]; dv = imp[f"{m}_delta"]
        print(f"{m.upper():<15} {bv:>12.4f} {lv:>12.4f} {'+' if dv>=0 else ''}{dv:>11.4f}")
    print("="*65)
    logger.info("Results → %s", CONFIG["results_file"])
    print("\n✅ Done! Run: python generate_plots.py\n")


# ─────────────────────────── MAIN ────────────────────────────────────────────
def main(stage="all"):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger.info("Loading model: %s", CONFIG["model_name"])
    tokenizer  = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    base_model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["model_name"])
    logger.info("Params: %dM", sum(p.numel() for p in base_model.parameters())//1_000_000)

    train_ds, val_ds, test_ds = load_and_prepare_dataset()
    baseline_m, bl_preds, labels = evaluate_baseline(test_ds, tokenizer, base_model)

    if stage in ("all", "lora"):
        train_tok = tokenize_dataset(train_ds, tokenizer)
        val_tok   = tokenize_dataset(val_ds,   tokenizer)
        lora_model, train_m, history = train_lora(train_tok, val_tok, tokenizer, base_model)
    else:
        lora_model = PeftModel.from_pretrained(base_model, CONFIG["lora_dir"])
        train_m, history = {}, {}

    logger.info("=== EVALUATING LoRA MODEL ===")
    eval_m, lo_preds, _ = run_inference(lora_model, tokenizer, test_ds)
    logger.info("LoRA eval: %s", eval_m)

    save_and_print(baseline_m, train_m, eval_m, history, bl_preds, lo_preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "baseline", "lora"])
    args = parser.parse_args()
    main(stage=args.stage)