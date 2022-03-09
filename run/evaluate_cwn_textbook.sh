#! /bin/bash
# Read arguments
set -e
export CUDA_VISIBLE_DEVICES=0
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

DATA_DIR=data/cwn/processed
MODEL=$MODEL_DIR/checkpoint_best.pt

fairseq-generate $DATA_DIR \
  --path $MODEL \
  --user-dir mass \
  --task translation_mix \
  --model_lang_pairs src-tgt textbook-textbook \
  --lang-pairs src-tgt \
  --dae-styles textbook \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --beam 5 \
  --lenpen 1.0 \
  --min-len 2 \
  --max-len-b 30 \
  --unkpen 3 \
  --no-repeat-ngram-size 3 \
  2>&1 | tee $MODEL_DIR/output_src_tgt.txt

cp $DATA_DIR/test.src-tgt.src.bin $DATA_DIR/test.src-textbook.src.bin
cp $DATA_DIR/test.src-tgt.src.idx $DATA_DIR/test.src-textbook.src.idx
cp $DATA_DIR/test.src-tgt.tgt.bin $DATA_DIR/test.src-textbook.textbook.bin
cp $DATA_DIR/test.src-tgt.tgt.idx $DATA_DIR/test.src-textbook.textbook.idx
cp $DATA_DIR/dict.tgt.txt $DATA_DIR/dict.textbook.txt

fairseq-generate $DATA_DIR \
  --path $MODEL \
  --user-dir mass \
  --task translation_mix \
  --model_lang_pairs src-tgt textbook-textbook \
  --lang-pairs src-textbook \
  --dae-styles textbook \
  --batch-size 128 \
  --skip-invalid-size-inputs-valid-test \
  --beam 5 \
  --lenpen 1.0 \
  --min-len 2 \
  --max-len-b 30 \
  --unkpen 3 \
  --no-repeat-ngram-size 3 \
  2>&1 | tee $MODEL_DIR/output_src_textbook.txt
