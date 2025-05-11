from datasets import Dataset
from transformers import LlamaTokenizer
import json
import gc
from tqdm import tqdm
import os
import numpy as np

def process_and_save_chunks_text(file_path, tokenizer, output_train_file, output_val_file, max_length=1024, chunk_size=1000000, val_ratio=0.1, seed=42):
    """ファイルを少しずつ読み込んでテキスト形式（txt）として保存する"""
    # 初期設定
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_file), exist_ok=True)
    
    # 乱数生成器
    rng = np.random.RandomState(seed)
    
    train_count = 0
    val_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f, \
         open(output_train_file, 'w', encoding='utf-8') as train_f, \
         open(output_val_file, 'w', encoding='utf-8') as val_f:
        
        chunk_counter = 0
        while True:
            # 一度に一定量のテキストだけを読み込む
            text_chunk = f.read(chunk_size)
            if not text_chunk:
                break
                
            # 読み込んだチャンクをトークン化
            tokenized_chunk = tokenizer(text_chunk)["input_ids"]
            # トークン列を指定の長さに分割
            chunks = [tokenized_chunk[i:i + max_length] for i in range(0, len(tokenized_chunk), max_length)]
            
            # バリデーション用と訓練用に分割して保存
            for seq in tqdm(chunks, desc=f"Processing chunk {chunk_counter}"):
                # 短すぎるシーケンスはスキップ
                if len(seq) < max_length // 2:
                    continue
                
                # スペース区切りでテキストに変換
                seq_text = ' '.join(map(str, seq))
                
                # バリデーションとトレーニングに分割
                if rng.random() < val_ratio:
                    # バリデーションデータ
                    val_f.write(seq_text + '\n')
                    val_count += 1
                else:
                    # トレーニングデータ
                    train_f.write(seq_text + '\n')
                    train_count += 1
            
            print(f"Processed chunk {chunk_counter} with {len(chunks)} sequences")
            chunk_counter += 1
            
            # メモリ解放
            del tokenized_chunk
            del chunks
            gc.collect()
    
    print(f"Saved {train_count} sequences to {output_train_file}")
    print(f"Saved {val_count} sequences to {output_val_file}")
    print(f"Text file size (train): {os.path.getsize(output_train_file) / (1024*1024):.2f} MB")
    print(f"Text file size (val): {os.path.getsize(output_val_file) / (1024*1024):.2f} MB")
    
    return train_count, val_count

def load_text_tokens(file_path):
    """テキスト形式からトークンデータを読み込む"""
    tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                # スペース区切りの文字列を整数のリストに変換
                tokens.append([int(token) for token in line.strip().split()])
    return tokens

# メインプログラムの変更例
if __name__ == "__main__":
    # ファイルパスを指定
    file_path = "/media/toda/ESD-EJ_R/corpus.txt"  # コーパスファイルのパス
    tokenizer_file = "./tokenizer"  # トークナイザーファイルのパス
    
    # 出力先
    train_file_txt = "./corpus/train_tokens.txt"
    val_file_txt = "./corpus/val_tokens.txt"
    
    # トークナイザーの読み込み
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_file)
    tokenizer.pad_token = tokenizer.eos_token
    
    # テキスト形式で保存
    print("Loading, tokenizing, and saving corpus to text files...")
    train_count, val_count = process_and_save_chunks_text(
        file_path, 
        tokenizer, 
        train_file_txt, 
        val_file_txt,
        val_ratio=0.05, 
        seed=42
    )
    
    print(f"Text dataset preparation completed! Train: {train_count} sequences, Val: {val_count} sequences")
    
