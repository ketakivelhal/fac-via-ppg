import os
import argparse
from pathlib import Path
import random
from itertools import chain


def partition(lst, n): 
    random.shuffle(lst)
    division = len(lst) / float(n) 
    lsts = [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]
    val = lsts[0]
    train = chain.from_iterable(x for x in lsts[1:])
    return train, val

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to partition dataset to train and val set.')

    parser.add_argument('feature_root', type=Path, default='',
                        help='location of the data corpus')
    parser.add_argument('--n_part', type=int, default=10,
                        help='number of partitions, the first partition becomes val set, the others become train set')
    args = parser.parse_args()

    speakers = os.listdir(args.feature_root)

    for speaker in speakers:
        speaker_dir = os.path.join(args.feature_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        print(speaker_dir)
        with open(os.path.join(speaker_dir, '_sources.txt'), 'r') as f:
            speaker_files = f.readlines()
        train, val = partition(speaker_files, args.n_part)
        with open(os.path.join(speaker_dir, '_sources_train.txt'), 'w') as f:
            f.writelines('%s' % line for line in train)
        with open(os.path.join(speaker_dir, '_sources_val.txt'), 'w') as f:
            f.writelines('%s' % line for line in val)
