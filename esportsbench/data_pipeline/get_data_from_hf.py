import os
import argparse
import pathlib
from datasets import load_dataset

DATA_DIR = pathlib.Path(__file__).parents[2] / 'data'

def main(destination, file_format, revision):
    os.makedirs(DATA_DIR / destination / file_format, exist_ok=True)
    dataset = load_dataset(
        'EsportsBench/EsportsBench',
        revision=revision
    )
    for split in dataset:
        df = dataset[split].to_polars()
        out_path = DATA_DIR / destination / file_format / f'{split}.{file_format}'
        if file_format == 'csv':
            df.write_csv(out_path)
        elif file_format == 'parquet':
            df.write_parquet(out_path)
        else:
            raise ValueError('only csv and parquet supported')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dest', required=False, default='hf_data')
    parser.add_argument('-r', '--revision', required=False, default='1.0')
    parser.add_argument('-f', '--file_format', required=False, default='parquet')
    args = parser.parse_args()
    dest = args.dest + f'/v{args.revision.replace(".", "_")}'
    main(dest, args.file_format, args.revision)