import os
import argparse
import pathlib
from datasets import load_dataset

DATA_DIR = pathlib.Path(__file__).parents[2] / 'data'

def main(destination, revision):
    os.makedirs(DATA_DIR / destination, exist_ok=True)
    dataset = load_dataset(
        'EsportsBench/EsportsBench',
        revision=revision
    )
    dataset.set_format('pandas')
    for split in dataset:
        dataset[split].to_csv(DATA_DIR / destination / f'{split}.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dest', required=False, default='hf_data')
    parser.add_argument('-r', '--revision', required=False, default='1.0')
    args = parser.parse_args()
    dest = args.dest + f'_{int(float(args.revision))}'
    main(dest, args.revision)