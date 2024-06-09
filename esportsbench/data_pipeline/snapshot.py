import os
import glob
import pathlib
from datetime import datetime
from argparse import ArgumentParser
import polars as pl

DATA_DIR = pathlib.Path(__file__).parents[2] / 'data'
SNAPSHOT_DIR = DATA_DIR / 'snapshots'

def make_snapshot_for_file(file_path):
    game_name = file_path.split('/')[-1].removesuffix('.csv')
    df = pl.read_csv(file_path)
    game_snapshot = df.select(
        pl.lit(game_name).alias('game_name'),
        pl.count().cast(pl.Int32).alias('num_matches'),
        pl.concat([pl.col('competitor_1'), pl.col('competitor_2')]).unique().count().cast(pl.Int32).alias('num_competitors'),
        pl.col('date').min().alias('first_date'),
        pl.col('date').max().alias('last_date'),
        pl.col('outcome').mean().alias('mean_outcome'),
    )
    return game_snapshot

def compare_snapshots():
    snapshots = sorted(glob.glob(f'{SNAPSHOT_DIR}/*.parquet'))
    prev = pl.read_parquet(snapshots[-2])
    cur = pl.read_parquet(snapshots[-1])
    comparison = prev.join(cur, on='game_name', how='inner', suffix="_2")
    summary = comparison.select([
        pl.col('game_name'),
        (pl.col('num_matches_2').cast(pl.Int32) - pl.col('num_matches').cast(pl.Int32)).alias('diff_num_matches'),
        (pl.col('num_competitors_2').cast(pl.Int32) - pl.col('num_competitors').cast(pl.Int32)).alias('diff_num_competitors'),
        pl.format("{} -> {}", pl.col('first_date').str.slice(0,10), pl.col('first_date_2').str.slice(0,10)).alias('change_first_date'),
        pl.format("{} -> {}", pl.col('last_date').str.slice(0,10), pl.col('last_date_2').str.slice(0,10)).alias('change_last_date'),
        (pl.col('mean_outcome_2') - pl.col('mean_outcome')).alias('diff_mean_outcome'),
    ])
    print(summary)

def make_snapshot():
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    game_snapshots = []
    for file in glob.glob(f'{DATA_DIR}/full_data/*.csv'):
        game_snapshot = make_snapshot_for_file(file)
        game_snapshots.append(game_snapshot)
    snapshot = pl.concat(game_snapshots)
    snapshot.write_parquet(SNAPSHOT_DIR / f'{name}.parquet')

if __name__ == '__main__':
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--make', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    pl.Config.set_tbl_rows(200)  # Adjust the number of rows to display
    pl.Config.set_fmt_str_lengths(100)
    if args.make:
        make_snapshot()
    if args.compare:
        compare_snapshots()
