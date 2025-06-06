import os
import glob
import pathlib
from datetime import datetime
from argparse import ArgumentParser
import polars as pl

DATA_DIR = pathlib.Path(__file__).parents[2] / 'data'
SNAPSHOT_DIR = DATA_DIR / 'snapshots'

def make_snapshot_for_file(file_path):
    game_name = file_path.split('/')[-1].removesuffix('.parquet')
    df = pl.read_parquet(file_path)
    game_snapshot = df.select(
        pl.lit(game_name).alias('game_name'),
        pl.len().cast(pl.Int32).alias('num_matches'),
        pl.concat([pl.col('competitor_1'), pl.col('competitor_2')]).unique().count().cast(pl.Int32).alias('num_competitors'),
        pl.col('date').min().alias('first_date'),
        pl.col('date').max().alias('last_date'),
        pl.col('outcome').mean().alias('mean_outcome'),
    )
    return game_snapshot

def make_snapshot(write=False):
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    game_snapshots = []
    for file in glob.glob(f'{DATA_DIR}/full_data/parquet/*.parquet'):
        game_snapshot = make_snapshot_for_file(file)
        game_snapshots.append(game_snapshot)
    snapshot = pl.concat(game_snapshots)
    if write:
        snapshot.write_parquet(SNAPSHOT_DIR / f'{name}.parquet')
    return snapshot


def compare_file_snapshots(prev_idx=-2, cur_idx=-1):
    snapshots = sorted(glob.glob(f'{SNAPSHOT_DIR}/*.parquet'))
    prev = pl.read_parquet(snapshots[prev_idx])
    cur = pl.read_parquet(snapshots[cur_idx])
    compare_snapshots(prev, cur)


def compare_snapshots(prev, cur):
    prev = prev.with_columns(
        pl.when(pl.col("game_name") == "fifa").then(pl.lit("ea_sports_fc")).otherwise(pl.col("game_name")).alias("game_name")
    )
    comparison = prev.join(cur, on='game_name', how='inner', suffix="_2")
    summary = comparison.select([
        pl.col('game_name'),
        (pl.col('num_matches_2').cast(pl.Int64) - pl.col('num_matches').cast(pl.Int64)).alias('diff_num_matches'),
        (pl.col('num_competitors_2').cast(pl.Int64) - pl.col('num_competitors').cast(pl.Int64)).alias('diff_num_competitors'),
        pl.when(pl.col('first_date') != pl.col('first_date_2')).then(
           pl.format("{} -> {}", pl.col('first_date').str.slice(0,10), pl.col('first_date_2').str.slice(0,10)) 
        ).otherwise(pl.lit('unchanged')).alias('first_date'),
        pl.when(pl.col('last_date') != pl.col('last_date_2')).then(
             pl.format("{} -> {}", pl.col('last_date').str.slice(0,10), pl.col('last_date_2').str.slice(0,10))
        ).otherwise(pl.lit('unchanged')).alias('last_date'),
        (pl.col('mean_outcome_2') - pl.col('mean_outcome')).alias('diff_mean_outcome'),
    ])
    # Create a new row
    total = pl.DataFrame({
        'game_name': ['total'],
        'diff_num_matches': [summary['diff_num_matches'].sum()],
        'diff_num_competitors': [summary['diff_num_competitors'].sum()],
        'first_date': [None],
        'last_date': [None],
        'diff_mean_outcome': [None],
    })

    # Append the new row to the summary
    summary = summary.vstack(total)
    print(summary)

def make_and_compare():
    snapshots = sorted(glob.glob(f'{SNAPSHOT_DIR}/*.parquet'))
    cur = make_snapshot().sort('num_matches', descending=True)
    print(cur)
    prev =  pl.read_parquet(snapshots[-1])
    compare_snapshots(prev, cur)


if __name__ == '__main__':
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    parser = ArgumentParser()
    parser.add_argument('--make', action='store_true')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    pl.Config.set_tbl_rows(200)  # Adjust the number of rows to display
    pl.Config.set_fmt_str_lengths(100)
    if args.make:
        make_snapshot(write=True)
    elif args.compare:
        compare_file_snapshots()
    else:
        make_and_compare()
