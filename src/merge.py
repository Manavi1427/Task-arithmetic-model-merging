#writes yaml file for every combination of tasks and lambda

import subprocess
import yaml
import os
import time
import sys

BASE_MODEL = "distilbert-base-uncased"

tasks=["sst2","mrpc","rte"]

def task_config(tasks, lam, method= "task-arithmetic"):
    models=[{"model": BASE_MODEL}]           #theta - Base model no weights

    for task in tasks:
        models.append({
            "model": f"models/finetuned_{task}",
            "parameter": {"weight": lam}
        })
    
    config={
        "method": method,
            "base_model": BASE_MODEL,
            "models": models,
            "parameters": {"normalize: False"},
            "dtype": "float32"
    }

    os.makedirs("configs", exist_ok=True)
    tag="_".join(tasks)
    path=f"configs/{method}_{tag}_1{str(lam).replace(".","")}.yaml"

    with open(path, "w") as f:
        yaml.dump(config,f)

    return path

def run_merge(config_path, output_dir):
    os.makedirs("output_dir", exist_ok=True)
    start=time.time()

    # find mergekit-yaml relative to current python executable
    python_dir = os.path.dirname(sys.executable)
    mergekit_cmd = os.path.join(python_dir, "mergekit-yaml.exe")

    subprocess.run([
        mergekit_cmd, config_path, output_dir,
        "--copy-tokenizer",
        "--allow-crimes",
        "--out-shard-size", "1B",
        "--lazy-unpickle",
        "--device", "cpu"
    ], check=True)

    #subprocess.run([
        #"mergekit-yaml", config_path, output_dir,
        #"--copy-tokenizer",      # copies tokenizer from base model to output
        #"--allow-crimes",        # enables CPU merging
        #"--out-shard-size", "1B",  # chunks output to avoid RAM overflow
        #"--lazy-unpickle",      # loads weights one at a time, saves RAM
        #"--device", "cpu"       # explicit CPU
    #], check=True)

    elapsed = round((time.time()-start)/60,2)

    #log compute time
    os.makedirs("results",exist_ok=True)
    with open("results/compute_log.txt", "a") as f:
        f.write(f"merge,{output_dir},{elapsed}\n")

    print(f"Merged-> {output_dir} ({elapsed} min)")

if __name__=="__main__":
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tasks=["sst2","mrpc","rte"]

    for lam in lambdas:
        cfg= task_config(tasks,lam)
        out_dir=f"models/merged/lambda_{str(lam).replace(".","_")}"
        run_merge(cfg, out_dir)


