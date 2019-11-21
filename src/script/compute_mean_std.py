import argparse
from pathlib import Path
from common.data_utils import Speaker
import numpy as np

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return np.log(np.clip(x, a_max=None, a_min=clip_val) * C)

parser = argparse.ArgumentParser(description='compute mean std')
parser.add_argument('dataset_dir', type=Path)
parser.add_argument('output_path_mean', type=Path, help='output path to the mean')
parser.add_argument('output_path_std', type=Path, help='output path to the std')
args = parser.parse_args()

speaker_dirs = [f for f in args.dataset_dir.glob("*") if f.is_dir()]
if len(speaker_dirs) == 0:
    raise Exception("No speakers found. Make sure you are pointing to the directory "
                    "containing all preprocessed speaker directories.")
speakers = [Speaker(speaker_dir) for speaker_dir in speaker_dirs]

sources = []
for speaker in speakers:
    sources.extend(speaker.sources)

sumx = np.zeros(80, dtype=np.float32)
sumx2 = np.zeros(80, dtype=np.float32)
count = 0
n = len(sources)
for i, source in enumerate(sources):
    print('{} / {}'.format(i, n))
    feature = np.load(source[0].joinpath(source[1]))
    feature = dynamic_range_compression(feature)
    sumx += feature.sum(axis=0)
    sumx2 += (feature * feature).sum(axis=0)
    count += feature.shape[0]

mean = sumx / count
std = np.sqrt(sumx2 / count - mean * mean)

mean = mean.astype(np.float32)
std = std.astype(np.float32)

np.save(args.output_path_mean, mean)
np.save(args.output_path_std, std)