import os 
import json
import csv
import torch
import time
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate as eval_lib

BASE_MODEL = "distilbert-base-uncased"
tasks = ["sst2","mrpc","rte"]
lambdas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

