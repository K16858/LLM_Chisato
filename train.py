from collections import OrderedDict
from datasets import Dataset
from transformers import LlamaTokenizer, MistralForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, MistralConfig
import torch
import json
from tqdm import tqdm
import math
import numpy as np
import gc
import os

# ファイルパスを指定
tokenizer_file = "./tokenizer"  # トークナイザーファイルのパス
config_file = "./config.json"
train_file = "./corpus/train_tokens.txt"
val_file = "./corpus/val_tokens.txt"

# トークナイザーの読み込み
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_file)
tokenizer.pad_token = tokenizer.eos_token

def load_tokenized_dataset_from_file(file_path):
    """テキスト形式からトークンデータを読み込む"""
    token_sequences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            if line.strip():
                # スペース区切りの文字列を整数のリストに変換
                tokens = [int(token) for token in line.strip().split()]
                token_sequences.append({"input_ids": tokens})
    
    return Dataset.from_list(token_sequences)

def get_streaming_dataset(file_path, block_size=1024):
    """ストリーミング方式でデータセットを読み込む"""
    def generator():
        with open(file_path, 'r', encoding='utf-8') as f:
            buffer = []
            for line in f:
                if line.strip():
                    tokens = [int(token) for token in line.strip().split()]
                    buffer.extend(tokens)
                    
                    while len(buffer) >= block_size:
                        yield {"input_ids": buffer[:block_size]}
                        buffer = buffer[block_size:]
            
            # 残りのバッファも使用
            if buffer:
                yield {"input_ids": buffer}
    
    return Dataset.from_generator(generator)

# 保存したデータセットを読み込む
print("Loading tokenized datasets...")
train_dataset = get_streaming_dataset(train_file, block_size=1024)
val_dataset = get_streaming_dataset(val_file, block_size=1024)

print(f"Loaded {len(train_dataset)} training examples and {len(val_dataset)} validation examples")

# データコラレーターの設定
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

config = load_config_from_json(config_file)

# モデルの初期化
model = MistralForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None, 
    config=config, 
    state_dict=OrderedDict(),
    # attn_implementation="flash_attention_2"
)
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

model_size = sum(t.numel() for t in model.parameters())
print(f"size: {model_size/1000**2:.1f}M parameters")

training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=1, 
    per_device_train_batch_size=4,  # 要調整
    learning_rate=5e-4,
    lr_scheduler_type="cosine",  # コサインスケジュールを
    warmup_steps=10000,  # ウォームアップステップ
    save_steps=20000,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=20000,  # 評価頻度も調整
    load_best_model_at_end=True,  # 最良モデルをロード
    # bf16=True,
    fp16=True,
    torch_compile=True,
    gradient_accumulation_steps=8,  # 増やして実効バッチサイズを維持
    gradient_checkpointing=True,
    optim="adamw_torch",
    # 訓練再開
    resume_from_checkpoint=True if os.path.exists("./model_output/checkpoint-*") else None,
)

# Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # deepspeed="ds_config.json"  # DeepSpeed設定ファイル
)

gc.collect()

# Training
print("[INFO] Start training...")
train_result = trainer.train()
trainer.save_model()

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
metrics["eval_samples"] = len(val_dataset)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluation
print("[INFO] Start evaluation...")
metrics = trainer.evaluate()

metrics["eval_samples"] = len(val_dataset)
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# トレーニング済みモデルの保存
model.save_pretrained("./model_output")
