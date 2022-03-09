# -*- coding: utf-8 -*-
import sys
import argparse
import scipy
import jieba
from sentence_transformers import SentenceTransformer


def main(args):
    embedder = SentenceTransformer('distiluse-base-multilingual-cased')

    hyp_data = []
    tgt_data = []
    with open(args.out_path) as fr_out, open(args.tgt_path) as fr_tgt:
        for hyp, tgt in zip(fr_out, fr_tgt):
            hyp_data.append(' '.join(jieba.lcut(hyp.replace(' ', ''))))
            tgt_data.append(' '.join(jieba.lcut(tgt.replace(' ', ''))))

    assert len(hyp_data) == len(tgt_data)
    total_sim = 0
    for hyp, tgt in zip(hyp_data, tgt_data):
        hyp_embedding = embedder.encode([hyp])
        tgt_embedding = embedder.encode([tgt])
        sim = 1 - scipy.spatial.distance.cdist(hyp_embedding, tgt_embedding, 'cosine')[0][0]
        total_sim += sim

    avg_sim = total_sim / len(hyp_data)
    print(f"Semantic Score: {avg_sim}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', type=str, help='path of reference file')
    parser.add_argument('--tgt_path', type=str, help='path of target file')
    args = parser.parse_args()
    sys.exit(main(args))
