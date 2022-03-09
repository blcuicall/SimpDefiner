# -*- coding -*-
import sys
from easse.sari import corpus_sari


def read_file(file):
    sents = []
    with open(file) as fr:
        for line in fr:
            sents.append(line.strip())
    return sents


def main(*argv):
    if not argv:
        argv = sys.argv[1:]
        assert len(argv) == 3
    orig_file = argv[0]
    sys_file = argv[1]
    refs_file = argv[2]
    orig_sents = read_file(orig_file)
    sys_sents = read_file(sys_file)
    refs_sents = read_file(refs_file)
    assert len(orig_sents) == len(sys_sents) == len(refs_sents)
    sari = corpus_sari(orig_sents=orig_sents,
                       sys_sents=sys_sents,
                       refs_sents=[refs_sents])
    print(sari)


if __name__ == '__main__':
    sys.exit(main())
