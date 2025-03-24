from datasets import Dataset
from transformers import LlamaTokenizer
import json
import gc
from tqdm import tqdm
import os
import numpy as np

def process_and_save_chunks(file_path, tokenizer, output_train_bin, output_val_bin, 
                           max_length=1024, chunk_size=1000000, val_ratio=0.1, seed=42):
    """ファイルを少しずつ読み込んで直接保存する"""
    # 初期設定
    os.makedirs(os.path.dirname(output_train_bin), exist_ok=True)
    os.makedirs(os.path.dirname(output_val_bin), exist_ok=True)
    
    # 一時ファイルの作成（後で結合するため）
    temp_train_file = output_train_bin + ".temp"
    temp_val_file = output_val_bin + ".temp"
    
    # 長さ情報を格納するリスト
    train_lengths = []
    val_lengths = []
    train_count = 0
    val_count = 0
    
    # 乱数生成器
    rng = np.random.RandomState(seed)
    
    with open(file_path, 'r', encoding='utf-8') as f, \
         open(temp_train_file, 'wb') as train_f, \
         open(temp_val_file, 'wb') as val_f:
        
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
                
                seq_array = np.array(seq, dtype=np.int32)
                
                # バリデーションとトレーニングに分割
                if rng.random() < val_ratio:
                    # バリデーションデータ
                    seq_array.tofile(val_f)
                    val_lengths.append(len(seq))
                    val_count += 1
                else:
                    # トレーニングデータ
                    seq_array.tofile(train_f)
                    train_lengths.append(len(seq))
                    train_count += 1
            
            print(f"Processed chunk {chunk_counter} with {len(chunks)} sequences")
            chunk_counter += 1
            
            # メモリ解放
            del tokenized_chunk
            del chunks
            gc.collect()
    
    # ヘッダー情報を含めた最終ファイルを作成
    finalize_binary_file(temp_train_file, output_train_bin, train_lengths, train_count)
    finalize_binary_file(temp_val_file, output_val_bin, val_lengths, val_count)
    
    # 一時ファイルを削除
    os.remove(temp_train_file)
    os.remove(temp_val_file)
    
    return train_count, val_count

def finalize_binary_file(temp_file, output_file, lengths, count):
    """一時バイナリファイルをヘッダー情報付きの最終ファイルに変換"""
    lengths_array = np.array(lengths, dtype=np.int32)
    
    with open(output_file, 'wb') as f_out:
        # ヘッダー情報：シーケンス総数
        np.array([count], dtype=np.int32).tofile(f_out)
        # 各シーケンスの長さ
        lengths_array.tofile(f_out)
        
        # データ本体をコピー
        with open(temp_file, 'rb') as f_in:
            while True:
                buffer = f_in.read(8192)  # 8KBずつ読み込み
                if not buffer:
                    break
                f_out.write(buffer)
    
    print(f"Saved {count} sequences to {output_file} (binary format)")
    print(f"Binary file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

def load_binary_tokens(file_path):
    """バイナリ形式のトークンデータを読み込む（検証用）"""
    with open(file_path, 'rb') as f:
        # ヘッダー情報を読み込む
        seq_count = np.fromfile(f, dtype=np.int32, count=1)[0]
        # 各シーケンスの長さを読み込む
        lengths = np.fromfile(f, dtype=np.int32, count=seq_count)
        # トークンを読み込む
        all_tokens = []
        for length in lengths:
            tokens = np.fromfile(f, dtype=np.int32, count=length).tolist()
            all_tokens.append(tokens)
        
    return all_tokens

# 以下の関数は参照用に残しておく
def save_tokens_to_file(tokens, output_file):
    """トークン化されたデータをファイルに保存"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for token_seq in tqdm(tokens, desc=f"Saving to {output_file}"):
            # トークンIDをJSON形式で保存（復元しやすくするため）
            f.write(json.dumps(token_seq) + "\n")
    print(f"Saved {len(tokens)} sequences to {output_file}")

def save_tokens_binary(tokens, output_file):
    """トークン化されたデータをバイナリ形式で保存"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 各シーケンスの長さを記録
    lengths = np.array([len(seq) for seq in tokens], dtype=np.int32)
    
    # すべてのトークンを一次元配列に結合
    all_tokens = []
    for seq in tokens:
        all_tokens.extend(seq)
    all_tokens = np.array(all_tokens, dtype=np.int32)
    
    # データを保存
    with open(output_file, 'wb') as f:
        # ヘッダー情報：シーケンス総数
        np.array([len(tokens)], dtype=np.int32).tofile(f)
        # 各シーケンスの長さ
        lengths.tofile(f)
        # すべてのトークンデータ
        all_tokens.tofile(f)
        
    print(f"Saved {len(tokens)} sequences to {output_file} (binary format)")
    print(f"Binary file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # ファイルパスを指定
    file_path = "./corpus/corpus.txt"  # コーパスファイルのパス
    tokenizer_file = "./tokenizer"  # トークナイザーファイルのパス
    
    # 出力先
    train_file_bin = "./corpus/train_tokens.bin"
    val_file_bin = "./corpus/val_tokens.bin"
    
    # トークナイザーの読み込み
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_file)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading, tokenizing, and saving corpus in chunks...")
    train_count, val_count = process_and_save_chunks(
        file_path, 
        tokenizer, 
        train_file_bin, 
        val_file_bin,
        val_ratio=0.1, 
        seed=42
    )
    
    print(f"Dataset preparation completed! Train: {train_count} sequences, Val: {val_count} sequences")
    
    # オプション：生成されたファイルが正しいか確認
    # print("Verifying saved files...")
    # train_tokens = load_binary_tokens(train_file_bin)
    # print(f"Loaded {len(train_tokens)} training sequences")
    # val_tokens = load_binary_tokens(val_file_bin)
    # print(f"Loaded {len(val_tokens)} validation sequences")
    