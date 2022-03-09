# -*- coding: utf-8 -*-
import os
import sys
import random
import string
from nltk.translate import bleu_score
from subprocess import Popen, PIPE
from collections import defaultdict
import jieba


def bleu(hyp, raw_data, bleu_path="metrics/sentence-bleu", nltk="cpp"):
    assert nltk in ['cpp', 'corpus', 'sentence'], \
        "nltk param should be cpp/corpus/sentence"
    assert len(hyp) == len(raw_data), \
        "sentence num in hyp not equal to dataset"
    tmp_dir = "/tmp"
    suffix = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    hyp_path = os.path.join(tmp_dir, 'hyp-' + suffix)
    base_ref_path = os.path.join(tmp_dir, 'ref-' + suffix)
    to_be_deleted = set()
    to_be_deleted.add(hyp_path)

    ref_dict = defaultdict(list)
    for word, exp, sense in raw_data:
        ref_dict[word].append(sense)

    score = 0
    num_hyp = 0
    if nltk == 'corpus':
        refs = []
    with open(os.devnull, 'w') as devnull:
        for idx, desc in enumerate(hyp):
            word = raw_data[idx][0]
            if nltk == 'sentence':
                if len(desc) == 0:
                    auto_reweigh = False
                else:
                    auto_reweigh = True
                bleu = bleu_score.sentence_bleu(
                    [r.split(' ') for r in ref_dict[word]],
                    desc,
                    smoothing_function=bleu_score.SmoothingFunction().method2,
                    auto_reweigh=auto_reweigh)
                score += bleu
                num_hyp += 1

            elif nltk == 'corpus':
                refs.append([r.split(' ') for r in ref_dict[word]])

            elif nltk == 'cpp':
                ref_paths = []
                for i, ref in enumerate(ref_dict[word][:30]):
                    ref_path = base_ref_path + str(i)
                    with open(ref_path, 'w') as f:
                        f.write(ref + '\n')
                        ref_paths.append(ref_path)
                        to_be_deleted.add(ref_path)

                with open(hyp_path, 'w') as f:
                    f.write(' '.join(desc) + '\n')

                rp = Popen(['cat', hyp_path], stdout=PIPE)
                bp = Popen([bleu_path] + ref_paths, stdin=rp.stdout, stdout=PIPE, stderr=devnull)
                out, err = bp.communicate()
                bleu = float(out.strip())
                score += bleu
                num_hyp += 1

            else:
                raise ValueError("nltk must be sentence/corpus/cpp")
    if nltk == 'cpp':
        for f in to_be_deleted:
            if os.path.exists(f):
                os.remove(f)
    if nltk == 'corpus':
        bleu = bleu_score.corpus_bleu(refs, [h for h in hyp],
                                      smoothing_function=bleu_score.SmoothingFunction().method2)
        ret_bleu = bleu
    else:
        ret_bleu = score / num_hyp

    return ret_bleu


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
        assert len(argv) == 3
    gold_src = argv[0]
    gold_tgt = argv[1]
    hyp_file = argv[2]

    hyp_data = []
    raw_data = []
    with open(gold_src) as fr_src, \
            open(gold_tgt) as fr_tgt, \
            open(hyp_file) as fr_hyp:
        src_content = fr_src.readlines()
        tgt_content = fr_tgt.readlines()
        hyp_content = fr_hyp.readlines()
    assert len(src_content) == len(tgt_content) == len(hyp_content)
    for src, tgt, hyp in zip(src_content, tgt_content, hyp_content):
        word, exp = src.strip().split(' [SEP] ')
        exp = ' '.join(jieba.lcut(exp.replace(' ', '')))
        hyp = jieba.lcut(hyp.strip().replace(' ', ''))
        tgt = ' '.join(jieba.lcut(tgt.strip().replace(' ', '')))
        hyp_data.append(hyp)
        raw_data.append((word, exp, tgt))

    bleu_score = bleu(hyp_data, raw_data, nltk='cpp')
    print(f"BLEU Score: {bleu_score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
