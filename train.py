from collections import OrderedDict
from datasets import Dataset
from transformers import LlamaTokenizer, MistralForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, MistralConfig
import torch
import json
from tqdm import tqdm
import math
import gc

# ファイルパスを指定
tokenizer_file = "./tokenizer"  # トークナイザーファイルのパス
config_file = "./config.json"
train_file = "./corpus/train_tokens.jsonl"
val_file = "./corpus/val_tokens.jsonl"

# トークナイザーの読み込み
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_file)
tokenizer.pad_token = tokenizer.eos_token

def load_tokenized_dataset_from_file(file_path):
    """保存したトークンデータを読み込む"""
    token_sequences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {file_path}"):
            if line.strip():
                tokens = json.loads(line.strip())
                token_sequences.append({"input_ids": tokens})
    
    return Dataset.from_list(token_sequences)

# 保存したデータセットを読み込む
print("Loading tokenized datasets...")
train_dataset = load_tokenized_dataset_from_file(train_file)
val_dataset = load_tokenized_dataset_from_file(val_file)

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

# トレーニング設定の定義
training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=5e-4,
    save_steps=100_000,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=5000,
    # bf16=True,
    fp16=True,
    torch_compile=True,
    # メモリ効率化のためのオプション
    gradient_accumulation_steps=4,  # 勾配蓄積によるバッチサイズ実質増加
    gradient_checkpointing=True,    # メモリ効率を優先（計算速度は犠牲に）
    optim="adamw_torch"             # メモリ効率の良いオプティマイザ
)

# Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# メモリ節約のためにデータセットをキャッシュ
# 必要なデータがメモリにロードされたら、参照を削除
gc.collect()

# Training
print("[INFO] Start training...")
train_result = trainer.train()
trainer.save_model()

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)

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
