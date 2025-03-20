
from collections import OrderedDict
from datasets import load_dataset, Dataset
from transformers import LlamaTokenizer, MistralForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer, MistralConfig
import torch
import json
from tqdm.notebook import tqdm
import math

def load_text_from_file(file_path, tokenizer, max_length=1024):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # トークン化して分割
    tokenized_text = tokenizer(text)["input_ids"]
    chunks = [tokenized_text[i:i + max_length] for i in range(0, len(tokenized_text), max_length)]
    
    # 既にトークン化されたデータを返す
    return Dataset.from_dict({"input_ids": chunks})

# ファイルパスを指定
file_path = "./corpus/corpus.txt"  # コーパスファイルのパス
tokenizer_file = "./tokenizer"  # トークナイザーファイルのパス
config_file = "./config.json"

# 2. カスタムトークナイザーの読み込み（事前に保存してあるもの）
# tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
tokenizer = LlamaTokenizer.from_pretrained(tokenizer_file)
# EOSトークンをPADトークンに設定（PADトークンあるけどね）
tokenizer.pad_token = tokenizer.eos_token

# データセットの作成
dataset = load_text_from_file(file_path, tokenizer)

# 訓練用と検証用に分割（例：80%:20%の割合）
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = {
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
}

# 3. テキストをトークン化する関数の定義
def tokenize_function(examples, max_length=128):
    return tokenizer(examples["text"], truncation=True, max_length=max_length)

# 4. データセットにトークナイズを適用
tokenized_datasets = {
    "train": dataset["train"],
    "validation": dataset["validation"]
}

# 5. データコラレーターの設定
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

def load_config_from_json(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        config = MistralConfig.from_dict(config)
    return config

config = load_config_from_json(config_file)

# 6. モデルの初期化
model = MistralForCausalLM.from_pretrained(
    pretrained_model_name_or_path=None, 
    config=config, 
    state_dict=OrderedDict(),
    # attn_implementation="flash_attention_2",
    # torch_dtype=torch.float16
)
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))

model_size = sum(t.numel() for t in model.parameters())
print(f"size: {model_size/1000**2:.1f}M parameters")

# 7. トレーニング設定の定義
training_args = TrainingArguments(
    output_dir="./model_output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    learning_rate=5e-4,
    save_steps=100_000,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=5000,
    bf16=True,
    # torch_compile=True
)

# 8. Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Training
print("[INFO] Strat training...")
train_result = trainer.train()
trainer.save_model()

metrics = train_result.metrics

metrics["train_samples"] = len(tokenized_datasets["train"])

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# Evaluation
print("[INFO] Start evaluation...")
metrics = trainer.evaluate()

metrics["eval_samples"] = len(tokenized_datasets["validation"])
try:
    perplexity = math.exp(metrics["eval_loss"])
except OverflowError:
    perplexity = float("inf")
metrics["perplexity"] = perplexity

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# 10. トレーニング済みモデルの保存
model.save_pretrained("./model_output")
