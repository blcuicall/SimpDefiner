#! /bin/bash
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
    --style)
        STYLE="$2"
        shift 2
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITIONAL[@]}"

OXFORD_SRC=data/oxford/raw/oxford.test.src
OXFORD_TGT=data/oxford/raw/oxford.test.tgt
OALD_SRC=data/aligned-oxford-oald-text/raw/test.src
OALD_TGT=data/aligned-oxford-oald-text/raw/test.oald
OALD_TGT_COMPLEX=data/aligned-oxford-oald-text/raw/test.oxford

CWN_SRC=data/cwn/raw/test.src
CWN_TGT=data/cwn/raw/test.tgt

if [[ "${STYLE}" == "oxford" ]]; then
    grep ^H "${MODEL_DIR}/output_src_tgt.txt" |
        sed 's/^H-//' |
        sort -n -k 1 |
        cut -f 3 |
        sed "s/ ##//g" \
            >"${MODEL_DIR}/output_src_tgt.ordered.tgt"
    echo "Calculating BLEU Score"
    python metrics/calc_bleu.py "${OXFORD_SRC}" "${OXFORD_TGT}" "${MODEL_DIR}/output_src_tgt.ordered.tgt"
    echo "Calculating Semantic Score"
    CUDA_VISIBLE_DEVICES=${CUDA} python metrics/calc_sent_sim.py --out_path "${MODEL_DIR}/output_src_tgt.ordered.tgt" --tgt_path "${OXFORD_TGT}"
    python metrics/sari.py ${OALD_TGT_COMPLEX} ${MODEL_DIR}/output_src_tgt.ordered.tgt ${OALD_TGT}

elif [[ "${STYLE}" == "oald" ]]; then
    grep ^H "${MODEL_DIR}/output_src_oald.txt" |
        sed 's/^H-//' |
        sort -n -k 1 |
        cut -f 3 |
        sed "s/ ##//g" \
            >"${MODEL_DIR}/output_src_oald.ordered.tgt"
    echo "Calculating BLEU Score"
    python metrics/calc_bleu.py "${OALD_SRC}" "${OALD_TGT}" "${MODEL_DIR}/output_src_oald.ordered.tgt"
    echo "Calculating Semantic Score"
    CUDA_VISIBLE_DEVICES=${CUDA} python metrics/calc_sent_sim.py --out_path "${MODEL_DIR}/output_src_oald.ordered.tgt" --tgt_path "${OALD_TGT}"
    python metrics/sari.py ${OALD_TGT_COMPLEX} ${MODEL_DIR}/output_src_oald.ordered.tgt ${OALD_TGT}

elif [[ "${STYLE}" == "cwn" ]]; then
    grep ^H "${MODEL_DIR}/output_src_tgt.txt" |
        sed 's/^H-//' |
        sort -n -k 1 |
        cut -f 3 |
        sed "s/ ##//g" \
            >"${MODEL_DIR}/output_src_tgt.ordered.tgt"
    echo "Calculating BLEU Score"
    python metrics/calc_bleu_zh.py "${CWN_SRC}" "${CWN_TGT}" "${MODEL_DIR}/output_src_tgt.ordered.tgt"
    echo "Calculating Semantic Score"
    CUDA_VISIBLE_DEVICES=${CUDA} python metrics/calc_sent_sim_zh.py --out_path "${MODEL_DIR}/output_src_tgt.ordered.tgt" --tgt_path "${CWN_TGT}"
    python metrics/hsk_freq.py ${MODEL_DIR}/output_src_tgt.ordered.tgt

elif [[ "${STYLE}" == "textbook" ]]; then
    grep ^H "${MODEL_DIR}/output_src_textbook.txt" |
        sed 's/^H-//' |
        sort -n -k 1 |
        cut -f 3 |
        sed "s/ ##//g" \
            >"${MODEL_DIR}/output_src_textbook.ordered.tgt"
    echo "Calculating BLEU Score"
    python metrics/calc_bleu_zh.py "${CWN_SRC}" "${CWN_TGT}" "${MODEL_DIR}/output_src_textbook.ordered.tgt"
    echo "Calculating Semantic Score"
    CUDA_VISIBLE_DEVICES=${CUDA} python metrics/calc_sent_sim_zh.py --out_path "${MODEL_DIR}/output_src_textbook.ordered.tgt" --tgt_path "${CWN_TGT}"
    python metrics/hsk_freq.py ${MODEL_DIR}/output_src_textbook.ordered.tgt
fi
