
from collections import OrderedDict
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer, MistralForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, MistralConfig
import torch
import json
from tqdm.notebook import tqdm
import math
config_file = "./config.json"

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

config = load_config_from_json(config_file)

model = MistralForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None, 
    config=config, 
    state_dict=OrderedDict(),
    attn_implementation="flash_attention_2",
    # torch_dtype=torch.float16
)

model_size = sum(t.numel() for t in model.parameters())
print(f"size: {model_size/1000**2:.1f}M parameters")
