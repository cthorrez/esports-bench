from datetime import datetime
import polars as pl
import pathlib
from esportsbench.constants import GAME_NAME_MAP

DATA_DIR = pathlib.Path(__file__).resolve().parents[2] / 'data'


def summary(game):
    print('======================================================')
    print(f'summary for {game}:')
    df = pl.read_csv(DATA_DIR / f'{game}.csv')
    print(f'num matches: {len(df)}')
    df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
    df = df.with_columns(pl.col('date').dt.year().alias('year'))
    future_df = df.filter(pl.col('date') >= datetime.now())
    print('num future matches:', len(future_df))
    year_counts = df.group_by('year').count().sort('year')
    print(year_counts)
    print('======================================================')


if __name__ == '__main__':
    for game in GAME_NAME_MAP:
        summary(game)
