#!/usr/bin/env bash
set -e
DATA_DIR=data/oxford/raw
OUT_DIR=data/oxford/processed

for SPLIT in train valid test; do
    python encode.py \
        --inputs $DATA_DIR/oxford.${SPLIT}.src \
        --outputs $DATA_DIR/oxford.${SPLIT}.bpe.src \
        --workers 30
    python encode.py \
        --inputs $DATA_DIR/oxford.${SPLIT}.tgt \
        --outputs $DATA_DIR/oxford.${SPLIT}.bpe.tgt \
        --workers 30
done

fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref $DATA_DIR/oxford.train.bpe \
    --validpref $DATA_DIR/oxford.valid.bpe \
    --testpref $DATA_DIR/oxford.test.bpe \
    --destdir $OUT_DIR \
    --srcdict pretrained_model/MASS-zh/dict.txt \
    --tgtdict pretrained_model/MASS-zh/dict.txt \
    --workers 30

DATA_DIR=data/annotated_oald_both/raw
OUT_DIR=data/annotated_oald_both/processed

python encode.py \
    --inputs $DATA_DIR/test.src \
    --outputs $DATA_DIR/test.bpe.src \
    --workers 30

python encode.py \
    --inputs $DATA_DIR/test.oxford \
    --outputs $DATA_DIR/test.bpe.oxford \
    --workers 30

python encode.py \
    --inputs $DATA_DIR/test.oald \
    --outputs $DATA_DIR/test.bpe.oald \
    --workers 30

fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang oxford \
    --testpref $DATA_DIR/test.bpe \
    --destdir $OUT_DIR \
    --srcdict pretrained_model/MASS/dict.txt \
    --tgtdict pretrained_model/MASS/dict.txt \
    --workers 20

fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang oald \
    --testpref $DATA_DIR/test.bpe \
    --destdir $OUT_DIR \
    --srcdict pretrained_model/MASS/dict.txt \
    --tgtdict pretrained_model/MASS/dict.txt \
    --workers 20

cp $DEST_DIR/dict.txt $DEST_DIR/dict.oxford.txt
cp $DEST_DIR/dict.txt $DEST_DIR/dict.oald.txt

DATA_DIR=data/oald/raw
DEST_DIR=data/oald/processed

for SPLIT in train valid test; do
    python encode.py \
        --inputs $DATA_DIR/oald.${SPLIT}.txt \
        --outputs $DATA_DIR/oald.${SPLIT}.bpe \
        --workers 30
done

fairseq-preprocess \
    --user-dir mass \
    --task translation_mix \
    --only-source \
    --trainpref ${DATA_DIR}/oald.train.bpe \
    --validpref ${DATA_DIR}/oald.valid.bpe \
    --testpref ${DATA_DIR}/oald.test.bpe \
    --destdir $DEST_DIR \
    --workers 20 \
    --srcdict pretrained_model/MASS/dict.txt

for split in train valid; do
    cp $DEST_DIR/$split.idx $DEST_DIR/$split.oald-None.oald.idx
    cp $DEST_DIR/$split.bin $DEST_DIR/$split.oald-None.oald.bin
done

cp $DEST_DIR/test.bin $DEST_DIR/test.noise-oald.oald.bin
cp $DEST_DIR/test.idx $DEST_DIR/test.noise-oald.oald.idx
cp $DEST_DIR/dict.txt $DEST_DIR/dict.noise.txt
cp $DEST_DIR/dict.txt $DEST_DIR/dict.oald.txt
