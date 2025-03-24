import sentencepiece as spm
from transformers import LlamaTokenizer
import gc
import os

MODEL_PREFIX = "Chisato_test"
OUTPUT_MODEL_DIR = "./tokenizer"

# トークナイザーのトレーニングとモデル変換を分離する
def train_tokenizer():
    spm.SentencePieceTrainer.train(
        input="./corpus/corpus.txt",
        model_type="unigram",
        model_prefix=MODEL_PREFIX,
        add_dummy_prefix=False,
        byte_fallback=True,
        vocab_size=31996,
        accept_language=["ja", "en"],
        character_coverage=0.9995,
        unk_piece="<unk>",
        pad_piece="<pad>",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        input_sentence_size=1000000,
        seed_sentencepiece_size=1000000,  # シードサイズの制限
        train_extremely_large_corpus=True  # 大規模コーパス用の設定
    )
    print(f"トークナイザーのトレーニング完了: {MODEL_PREFIX}.model")

def test_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PREFIX + ".model")

    # テストと検証
    print("=== トークナイザーのテスト ===")
    print(sp.encode_as_pieces("これは、テストです。"))
    print(sp.encode_as_ids("これは、テストです。"))
    print(sp.decode_pieces(['▁', 'これは', '、', 'テスト', 'です', '。']))
    print(sp.decode_ids([381, 260, 1662, 279, 261]))
    print(f"語彙サイズ: {sp.get_piece_size()}")
    
    # メモリ解放
    del sp
    gc.collect()

def convert_to_hf_tokenizer():
    tokenizer = LlamaTokenizer(
        vocab_file=MODEL_PREFIX + ".model",
        unk_token='<unk>',
        bos_token='<s>',
        eos_token='</s>',
        pad_token='<pad>',
        extra_ids=0,
        model_max_length=1024
    )
    
    # チャット用の特殊トークンを追加
    special_tokens_dict = {
        'sep_token': '<sep>',
        'additional_special_tokens': [
            '<|system|>',
            '<|user|>',
            '<|assistant|>'
        ]
    }
    
    # トークナイザーに特殊トークンを追加
    tokenizer.add_special_tokens(special_tokens_dict)
    
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"HuggingFace形式のトークナイザーを保存しました: {OUTPUT_MODEL_DIR}")
    print(f"追加された特殊トークン: {tokenizer.additional_special_tokens}")

if __name__ == "__main__":
    try:
        # すでにモデルファイルが存在する場合はトレーニングをスキップ
        if not os.path.exists(f"{MODEL_PREFIX}.model"):
            train_tokenizer()
            gc.collect()
        
        test_tokenizer()
        convert_to_hf_tokenizer()
        
        print("全ての処理が正常に完了しました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
