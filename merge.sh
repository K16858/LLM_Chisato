echo "Start merge"

for i in {0..10}
do
  echo ${i}
  cat corpus/common_crawl_${i}.jsonl > corpus/merged_common_crawl.jsonl
  rm corpus/common_crawl_${i}.jsonl
done

echo "Finish"
