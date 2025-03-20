import json

# コーパスファイルのパス
input_file_path = './corpus/common_crawl_0.jsonl'
output_file_path = './corpus/corpus.txt'

# コーパスファイルの内容をロードして、必要なテキスト部分を抽出
def extract_contents_from_file(input_file_path):
    contents = []
    with open(input_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 空行をスキップ
                # 各行を個別にJSONオブジェクトとして解析
                entry = json.loads(line)
                contents.append(entry['content'])
    return contents

# テキストを事前学習用にファイルに書き出す
def write_contents_to_file(contents, output_file_path):
    with open(output_file_path, 'w', encoding='utf-8') as f:
        # 内容を1行ごとに書き出す
        for content in contents:
            f.write(content + "\n")

# コーパスファイルから内容を抽出
contents = extract_contents_from_file(input_file_path)

# 学習用ファイルに書き出す
write_contents_to_file(contents, output_file_path)

print(f"ファイル '{output_file_path}' が作成されました。")
