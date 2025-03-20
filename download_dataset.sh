echo "Start download"

for i in {0..18}
do
  echo ${i}
  curl https://abeja-cc-ja.s3.ap-northeast-1.amazonaws.com/common_crawl_${i}.jsonl -O ./corpus/common_crawl_${i}.jsonl
done

echo "Finish"
