import os 
import time
import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL= "distilbert-base-uncased"

#loads model from disk and returns its weights as dict (key=layerName: Value=tensors)
def load_weights(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,num_labels=2
    )
    return {name: param.clone() for name, param in model.named_parameters()}

def task_arithmetic_merge(base_weights, finetuned_weights_list, lam):
    merged={}
    for name, base_param in base_weights.items():
        task_vector_sum = torch.zeros_like(base_param)    #for task vectors
        for ft_weights in finetuned_weights_list:
            if name in ft_weights:
                tau= ft_weights[name] - base_param    
                task_vector_sum +=tau

        merged[name] = base_param + lam*task_vector_sum
    return merged 

def save_merged_model(base_path, merged_weights, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_path, num_labels=2
    )
    state_dict=model.state_dict()
    for name, param in merged_weights.items():
        if name in state_dict:
            state_dict[name] = param
        model.load_state_dict(state_dict)
        model.save_pretrained(output_dir)

        #save tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        tokenizer.save_pretrained(output_dir)

def run_merge(tasks, lam, output_dir):
    start= time.time()
    print(f"\nMerging {tasks} at λ={lam}")

    print("  Loading base model weights...")
    base_weights= load_weights(BASE_MODEL)

    print("  loading finetuned weights...")
    finetuned_weights_list=[]
    for task in tasks:
        path = f"models/finetuned_{task}"
        print(f"  Loading {path}")
        finetuned_weights_list.append(load_weights(path))
    
    print("  Computing task vectors and merging...")
    merged_weights = task_arithmetic_merge(base_weights, finetuned_weights_list, lam)
    
    print(f"  Saving merged model to {output_dir}...")
    save_merged_model(BASE_MODEL, merged_weights, output_dir)
    elapsed = round((time.time() - start)/60,2)
    with open("results/compute_log.txt", "a") as f:
        f.write(f"merged,{output_dir},{elapsed}\n")
    print(f"Done → {output_dir} ({elapsed} min)")


if __name__=="__main__":
    lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tasks=["sst2","mrpc","rte"]

    for lam in lambdas:
        out_dir = f"models/merged/lambda_{str(lam).replace('.','_')}"
        run_merge(tasks, lam, out_dir)
