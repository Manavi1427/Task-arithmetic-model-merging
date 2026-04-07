 #fine-tuning base model on each task

import argparse
import time
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,  
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate
import os 
import numpy as np

BASE_MODEL = "distilbert-base-uncased"         #66M params 

TASK_CONFIG = {
    #sentiment analysis
    "sst2": {
        "dataset": "glue",
        "subset": "sst2",
        "text_col": "sentence",
        "num_labels": 2,
        "metric": "accuracy",
    },
    #semantic equivalence
    "mrpc": {
        "dataset": "glue",
        "subset": "mrpc",
        "text_col": ["sentence1","sentence2"],
        "num_labels": 2,
        "metric": "f1",
    },
    #recongnizing textual entailement
    "rte": {
        "dataset": "glue",
        "subset": "rte",
        "text_col": ["sentence1","sentence2"],
        "num_labels": 2,
        "metric": "accuracy",
    }
}

def tokenize(batch,tokenizer,text_col):
    if isinstance(text_col,list):
        return tokenizer(
            batch[text_col[0]], batch[text_col[1]], truncation=True,max_length=128
        )
    return tokenizer(batch[text_col],truncation=True,max_length=128)

def finetune(Task_name):
    cfg=TASK_CONFIG[Task_name]
    print(f"\n--Fine tuning on Task: {Task_name.upper()}--")
    start=time.time()

    tokenizer=AutoTokenizer.from_pretrained(BASE_MODEL)


    dataset=load_dataset(cfg['dataset'], cfg['subset'])
    tokenized=dataset.map(
        lambda b: tokenize(b, tokenizer, cfg["text_col"]),
        batched=True
    )

    tokenized["train"] = tokenized["train"].select(range(min(3000, len(tokenized["train"]))))
    tokenized["validation"] = tokenized["validation"].select(range(min(400, len(tokenized["validation"]))))

    model=AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=cfg["num_labels"]
    )

    model.config.pad_token_id=tokenizer.pad_token_id

    metric=evaluate.load(cfg["dataset"],cfg["subset"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return metric.compute(predictions=preds, references=labels)
    
    out_dir=f"models/finetuned_{Task_name}"

    args = TrainingArguments(
        output_dir= out_dir,
        num_train_epochs= 2,
        per_device_train_batch_size= 32,
        per_device_eval_batch_size= 64,
        eval_strategy= "epoch",
        save_strategy= "epoch",
        load_best_model_at_end=  True,
        learning_rate= 2e-5,
        weight_decay= 0.01,
        logging_steps= 50,
        use_cpu= True,
        report_to= []
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator = DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    elapsed = round ((time.time() - start )/60, 2)
    print(f"Done. Time : {elapsed} min -> saved to {out_dir}")

    #Log Compute time 

    os.makedirs("results",exist_ok=True)
    with open("results/compute_log.txt", "a") as f:
        f.write(f"finetune,{Task_name},{elapsed}\n")

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["sst2","mrpc","rte","all"], default="all")
    args= parser.parse_args()

    tasks = ["sst2", "mrpc", "rte"] if args.task == "all" else [args.task]
    for t in tasks:
        finetune(t)



    


