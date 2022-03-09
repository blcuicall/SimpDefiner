#! /bin/bash
# Read arguments

set -e
export CUDA_VISIBLE_DEVICES=7
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --model_dir)
    MODEL_DIR="$2"
    shift 2
    ;;
  *)
    POSITIONAL+=("$1")
    shift
    ;;
  esac
done
set -- "${POSITIONAL[@]}"

DATA_DIR=data/oxford/processed
OALD_DATA_DIR=data/annotated-oxford-oald-test/processed
MODEL=$MODEL_DIR/checkpoint_best.pt

fairseq-generate $DATA_DIR \
  --path $MODEL \
  --user-dir mass \
  --task translation_mix \
  --model_lang_pairs src-tgt oald-oald \
  --lang-pairs src-tgt \
  --dae-styles oald \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --beam 5 \
  --lenpen 1.0 \
  --min-len 2 \
  --max-len-b 30 \
  --unkpen 3 \
  --no-repeat-ngram-size 3 \
  2>&1 | tee $MODEL_DIR/output_src_tgt.txt
#bash metrics/calc_metrics.sh $MODEL_DIR oxford $CUDA >$MODEL_DIR/log_oxford_metrics.txt

cp $OALD_DATA_DIR/test.src-oald.src.bin $DATA_DIR/test.src-oald.src.bin
cp $OALD_DATA_DIR/test.src-oald.src.idx $DATA_DIR/test.src-oald.src.idx
cp $OALD_DATA_DIR/test.src-oald.oald.bin $DATA_DIR/test.src-oald.oald.bin
cp $OALD_DATA_DIR/test.src-oald.oald.idx $DATA_DIR/test.src-oald.oald.idx
cp $OALD_DATA_DIR/dict.oald.txt $DATA_DIR/dict.oald.txt

fairseq-generate $DATA_DIR \
  --path $MODEL \
  --user-dir mass \
  --task translation_mix \
  --model_lang_pairs src-tgt oald-oald \
  --lang-pairs src-oald \
  --dae-styles oald \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --beam 5 \
  --lenpen 1.0 \
  --min-len 2 \
  --max-len-b 30 \
  --unkpen 3 \
  --no-repeat-ngram-size 3 \
  2>&1 | tee $MODEL_DIR/output_src_oald.txt
#bash metrics/calc_metrics.sh $MODEL_DIR oald $CUDA >$MODEL_DIR/log_oald_metrics.txt
