import os
import csv
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate as eval_lib

BASE_MODEL = "distilbert-base-uncased"
TASKS = ["sst2", "mrpc", "rte"]
LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

TASK_CONFIG = {
    "sst2": {
        "dataset": "glue",
        "subset":  "sst2",
        "text_col": "sentence",
        "label_col": "label",
        "metric": "accuracy",
    },
    "mrpc": {
        "dataset": "glue",
        "subset":  "mrpc",
        "text_col": ["sentence1", "sentence2"],
        "label_col": "label",
        "metric": "f1",
    },
    "rte": {
        "dataset": "glue",
        "subset":  "rte",
        "text_col": ["sentence1", "sentence2"],
        "label_col": "label",
        "metric": "accuracy",
    },
}

# ── helpers ──────────────────────────────────────────────────────────────────

def get_tokenizer():
    local = "models/base_tokenizer"
    if os.path.exists(local):
        return AutoTokenizer.from_pretrained(local)
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def load_model(model_path, local_only=True):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        local_files_only=local_only,
    )
    model.eval()
    return model


def evaluate_task(model, tokenizer, task_name, max_samples=400):
    """Evaluate model on one GLUE task. Returns a single float score."""
    cfg = TASK_CONFIG[task_name]
    dataset = load_dataset(cfg["dataset"], cfg["subset"])
    val = dataset["validation"].select(range(min(max_samples, len(dataset["validation"]))))

    metric = eval_lib.load(cfg["dataset"], cfg["subset"])
    all_preds, all_labels = [], []

    with torch.no_grad():
        for i in range(0, len(val), 32):
            batch = val.select(range(i, min(i + 32, len(val))))

            if isinstance(cfg["text_col"], list):
                enc = tokenizer(
                    batch[cfg["text_col"][0]],
                    batch[cfg["text_col"][1]],
                    truncation=True, max_length=128,
                    padding=True, return_tensors="pt",
                )
            else:
                enc = tokenizer(
                    batch[cfg["text_col"]],
                    truncation=True, max_length=128,
                    padding=True, return_tensors="pt",
                )

            logits = model(**enc).logits
            preds  = torch.argmax(logits, dim=-1).tolist()
            labels = batch[cfg["label_col"]]

            all_preds.extend(preds)
            all_labels.extend(labels)

    result = metric.compute(predictions=all_preds, references=all_labels)
    score  = result.get("f1", result.get("accuracy", 0.0))
    return round(score * 100, 2)


def eval_model_all_tasks(model, tokenizer, label):
    """Run eval on all 3 tasks and return a result dict."""
    scores = {}
    for task in TASKS:
        print(f"    {task} ...", end=" ", flush=True)
        s = evaluate_task(model, tokenizer, task)
        scores[task] = s
        print(s)
    avg = round(float(np.mean(list(scores.values()))), 2)
    print(f"    avg: {avg}")
    return {"model": label, "lambda": "N/A", **scores, "avg": avg}


# ── main ─────────────────────────────────────────────────────────────────────

def evaluate_all():
    os.makedirs("results", exist_ok=True)
    tokenizer = get_tokenizer()
    rows = []
    headers = ["model", "lambda"] + TASKS + ["avg"]

    # ── 1. Base model ──────────────────────────────────────────────────
    print("\n=== Base model (no fine-tuning) ===")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2
    )
    base_model.eval()
    rows.append(eval_model_all_tasks(base_model, tokenizer, "base_distilbert"))
    del base_model

    # ── 2. Individual fine-tuned models ───────────────────────────────
    print("\n=== Individual fine-tuned models ===")
    for task in TASKS:
        path = f"models/finetuned_{task}"
        if not os.path.exists(path):
            print(f"  Skipping {task} — folder not found")
            continue
        print(f"  individual_{task}")
        model = load_model(path)
        row   = eval_model_all_tasks(model, tokenizer, f"individual_{task}")
        row["lambda"] = "N/A"
        rows.append(row)
        del model

    # ── 3. Lambda sweep (task arithmetic merged models) ────────────────
    print("\n=== Task arithmetic — lambda sweep ===")
    for lam in LAMBDAS:
        folder = f"models/merged/lambda_{str(lam).replace('.', '_')}"
        if not os.path.exists(folder):
            print(f"  Skipping λ={lam} — folder not found")
            continue
        print(f"  λ = {lam}")
        model = load_model(folder)
        row   = eval_model_all_tasks(model, tokenizer, "task_arithmetic")
        row["lambda"] = lam
        rows.append(row)
        del model

    # ── 4. Write CSV ───────────────────────────────────────────────────
    csv_path = "results/lambda_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved → {csv_path}")

    # ── 5. Print summary table ─────────────────────────────────────────
    print("\n{:<30} {:>8} {:>8} {:>8} {:>8}".format("model", "lambda", *TASKS[:2], "avg"))
    print("-" * 66)
    for r in rows:
        print("{:<30} {:>8} {:>8} {:>8} {:>8}".format(
            r["model"], str(r["lambda"]),
            str(r.get(TASKS[0], "-")),
            str(r.get(TASKS[1], "-")),
            str(r["avg"])
        ))


if __name__ == "__main__":
    evaluate_all()



