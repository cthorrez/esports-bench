import argparse
from datasets import load_dataset

def main(va, vb):
    dataset_a = load_dataset(
        "EsportsBench/EsportsBench",
        revision=va
    )
    dataset_b = load_dataset(
        "EsportsBench/EsportsBench",
        revision=vb
    )
    total = 0
    for split in dataset_a:
        df_a = dataset_a[split].to_polars()
        if (split == 'fifa') and ('fifa' not in dataset_b):
            split = 'ea_sports_fc'
        df_b = dataset_b[split].to_polars()
        diff = len(df_b) - len(df_a)
        total += diff
        print(f'{split}: {diff} new rows')
    print(f'total new rows: {total}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-va', default='4.0')
    parser.add_argument('-vb', default='5.0')
    args = parser.parse_args()
    main(args.va, args.vb)