#!/bin/bash
set -e
DATA_DIR=data/cwn/raw
OUT_DIR=data/cwn/processed

for SPLIT in train valid test; do
    python encode-zh.py \
        --inputs $DATA_DIR/${SPLIT}.src \
        --outputs $DATA_DIR/${SPLIT}.bpe.src \
        --workers 30
    python encode-zh.py \
        --inputs $DATA_DIR/${SPLIT}.tgt \
        --outputs $DATA_DIR/${SPLIT}.bpe.tgt \
        --workers 30
done

fairseq-preprocess \
    --user-dir mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref $DATA_DIR/train.bpe \
    --validpref $DATA_DIR/valid.bpe \
    --testpref $DATA_DIR/test.bpe \
    --destdir $OUT_DIR \
    --srcdict pretrained_model/MASS-zh/dict.txt \
    --tgtdict pretrained_model/MASS-zh/dict.txt \
    --workers 20

DATA_DIR=data/textbook/raw
DEST_DIR=data/textbook/processed

for SPLIT in train valid test; do
    python encode-zh.py \
        --inputs $DATA_DIR/${SPLIT}.txt \
        --outputs $DATA_DIR/${SPLIT}.bpe \
        --workers 30
done

fairseq-preprocess \
    --user-dir mass \
    --task translation_mix \
    --only-source \
    --trainpref ${DATA_DIR}/train.bpe \
    --validpref ${DATA_DIR}/valid.bpe \
    --testpref ${DATA_DIR}/test.bpe \
    --destdir $DEST_DIR \
    --workers 20 \
    --srcdict pretrained_model/MASS-zh/dict.txt

for split in train valid; do
    cp $DEST_DIR/$split.idx $DEST_DIR/$split.prim-None.prim.idx
    cp $DEST_DIR/$split.bin $DEST_DIR/$split.prim-None.prim.bin
done

cp $DEST_DIR/test.bin $DEST_DIR/test.noise-prim.prim.bin
cp $DEST_DIR/test.idx $DEST_DIR/test.noise-prim.prim.idx
cp $DEST_DIR/dict.txt $DEST_DIR/dict.noise.txt
cp $DEST_DIR/dict.txt $DEST_DIR/dict.prim.txt
