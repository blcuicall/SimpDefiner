#! /bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3

SUMM_DIR=data/oxford_public/processed
DAE_DIR=data/oald/processed
PRETRAINED_MODEL_PATH=pretrained_model/MASS/mass-base-uncased.pt
SAVE_DIR=checkpoints/sdgf-811
mkdir -p $SAVE_DIR

python train.py \
  $SUMM_DIR:$DAE_DIR \
  --user-dir mass --task translation_mix --arch transformer_mix_base \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 3e-4 --min-lr 1e-09 \
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 500 \
  --weight-decay 0.0 \
  --seed 1111 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --update-freq 4 --max-tokens 3072 \
  --ddp-backend=no_c10d --max-epoch 20 \
  --max-source-positions 512 --max-target-positions 512 \
  --skip-invalid-size-inputs-valid-test \
  --dropout 0.2 \
  --load-from-pretrained-model $PRETRAINED_MODEL_PATH \
  --model_lang_pairs src-tgt oald-oald --lang-pairs src-tgt --dae-styles oald \
  --lambda-parallel-config 0.8 --lambda-denoising-config 0.1 --lambda-lm-config 0.1 \
  --max-word-shuffle-distance 5 \
  --word-dropout-prob 0.2 \
  --word-blanking-prob 0.2 \
  --divide-decoder-self-attn-norm True \
  --divide-decoder-final-norm True \
  --divide-decoder-encoder-attn-query True \
  --save-dir $SAVE_DIR \
  2>&1 | tee $SAVE_DIR/training.log
