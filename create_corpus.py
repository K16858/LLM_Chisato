import json

# コーパスファイルのパス
input_file_path = './corpus/common_crawl_0.jsonl'
output_file_path = './corpus/corpus.txt'

# ストリーミング方式でファイルを処理
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as fin, \
         open(output_file_path, 'w', encoding='utf-8') as fout:
        count = 0
        for line in fin:
            if line.strip():  # 空行をスキップ
                try:
                    entry = json.loads(line)
                    fout.write(entry['content'] + "\n")
                    count += 1
                    if count % 10000 == 0:  # 進捗状況を表示
                        print(f"{count} 行処理しました")
                except json.JSONDecodeError:
                    print(f"JSONデコードエラー: {line[:50]}...")
                except KeyError:
                    print("'content' キーが見つかりません")
    
    print(f"合計 {count} 行処理し、ファイル '{output_file_path}' に保存しました。")

# ファイル処理を実行
process_file(input_file_path, output_file_path)
