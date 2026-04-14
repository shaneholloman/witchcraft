rm -rf output.txt

make warp-cli || exit 1

export RUN=./warp-cli

# assumes that "make nfcorpus" has already been run
$RUN querycsv $HOME/src/xtr-warp/beir/nfcorpus/questions.test.tsv output.txt &&\

echo scoring...
python score.py output.txt $HOME/src/xtr-warp/beir/nfcorpus/collection_map.json $HOME/src/xtr-warp/beir/nfcorpus/qrels.test.json
