from datasets import Dataset
from transformers import LlamaTokenizer
import json
import gc
from tqdm import tqdm
import os

def load_text_from_file(file_path, tokenizer, max_length=1024, chunk_size=1000000):
    """ファイルを少しずつ読み込んでデータセットを作成する"""
    all_chunks = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            # 一度に一定量のテキストだけを読み込む
            text_chunk = f.read(chunk_size)
            if not text_chunk:
                break
                
            # 読み込んだチャンクをトークン化
            tokenized_chunk = tokenizer(text_chunk)["input_ids"]
            # トークン列を指定の長さに分割
            chunks = [tokenized_chunk[i:i + max_length] for i in range(0, len(tokenized_chunk), max_length)]
            all_chunks.extend(chunks)
            
            print(f"Processed chunk with {len(chunks)} sequences")
    
    return Dataset.from_dict({"input_ids": all_chunks})

def save_tokens_to_file(tokens, output_file):
    """トークン化されたデータをファイルに保存"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for token_seq in tqdm(tokens, desc=f"Saving to {output_file}"):
            # トークンIDをJSON形式で保存（復元しやすくするため）
            f.write(json.dumps(token_seq) + "\n")
    print(f"Saved {len(tokens)} sequences to {output_file}")

if __name__ == "__main__":
    # ファイルパスを指定
    file_path = "./corpus/corpus.txt"  # コーパスファイルのパス
    tokenizer_file = "./tokenizer"  # トークナイザーファイルのパス
    
    # 出力先
    train_file = "./corpus/train_tokens.jsonl"
    val_file = "./corpus/val_tokens.jsonl"
    
    # トークナイザーの読み込み
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_file)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading and tokenizing corpus...")
    dataset = load_text_from_file(file_path, tokenizer)
    
    print("Splitting dataset...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print("Saving train dataset...")
    save_tokens_to_file(split_dataset["train"]["input_ids"], train_file)
    
    print("Saving validation dataset...")
    save_tokens_to_file(split_dataset["test"]["input_ids"], val_file)
    
    # メモリ解放
    del dataset
    del split_dataset
    gc.collect()
    
    print("Dataset preparation completed!")
    