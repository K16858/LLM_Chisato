echo "Start download"

> merged_common_crawl.jsonl

for i in {0..10}
do
  echo ${i}
  curl https://abeja-cc-ja.s3.ap-northeast-1.amazonaws.com/common_crawl_${i}.jsonl -O ./corpus/common_crawl_${i}.jsonl
  echo "Merging files..."
  cat common_crawl_${i}.jsonl >> merged_common_crawl.jsonl
  rm common_crawl_${i}.jsonl
done

echo "Download completed"

echo "Finish"
