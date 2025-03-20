import sentencepiece as spm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, LlamaTokenizer
from tokenizers.implementations import SentencePieceUnigramTokenizer

MODEL_PREFIX = "Chisato_test"
OUTPUT_MODEL_DIR = "./tokenizer"

spm.SentencePieceTrainer.train(
    input="./corpus/corpus.txt",  # コーパスファイル
    model_type="unigram",  # デフォルト
    model_prefix=MODEL_PREFIX,  # 出力されるモデルのファイル名に使われる
    add_dummy_prefix=False,# rinna-3.6bに習って、文章の先頭にスペースが追加されないように
    byte_fallback=True,# 未知語をutf-8バイトに分解するために
    vocab_size=32000,  # 32k 64kのサイズがいい説もある
    character_coverage=0.9995,
    unk_piece="<unk>",
    pad_piece="<pad>",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    input_sentence_size=12000000
)

sp = spm.SentencePieceProcessor()
sp.Load(MODEL_PREFIX+".model")

def tokenize(raw_text):
    tokenized=sp.encode_as_pieces(raw_text)
    return tokenized

# encode: text => is
print(sp.encode_as_pieces("これは、テストです。"))
print(sp.encode_as_ids("これは、テストです。"))

# decode: id => text
print(sp.decode_pieces(['▁', 'これは', '、', 'テスト', 'です', '。']))
print(sp.decode_ids([381, 260, 1662, 279, 261]))

# check vocab size
print(sp.get_piece_size())

tokenizer = LlamaTokenizer(
    vocab_file=MODEL_PREFIX+".model",
    unk_token = '<unk>',
    bos_token = '<s>',
    eos_token = '</s>',
    pad_token = '<pad>',
    extra_ids=0,
    model_max_length=1000000000000000019884624838656,
)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR) 
