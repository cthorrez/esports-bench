"""module for managing esports datasets for rating system experiments"""
import pathlib
import numpy as np
import polars as pl
from riix.utils.data_utils import TimedPairDataset
from esportsbench.constants import GAME_NAME_MAP

BASE_DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / 'data' 


def load_dataset(
    game,
    rating_period='7D',
    drop_draws=False,
    max_rows=None,
    train_end_date='2023-03-31',
    test_end_date='2024-03-31',
    data_dir = 'final_data',
):
    # map short name to full name if short name is provided
    if game in GAME_NAME_MAP:
        game = GAME_NAME_MAP[game]
    df = pl.read_parquet(BASE_DATA_DIR / data_dir / f'parquet/{game}.parquet')
    if drop_draws:
        df = df.filter(pl.col('outcome') != 0.5)
    if max_rows:
        df = df.head(max_rows)
    train_mask = df['date'].cast(pl.Utf8) <= train_end_date
    test_mask = (df['date'].cast(pl.Utf8) > train_end_date) & (df['date'].cast(pl.Utf8) <= test_end_date)
    train_rows = int(train_mask.sum())
    test_rows = int(test_mask.sum())
    dataset = TimedPairDataset(
        df=df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period=rating_period,
        verbose=True
    )
    dataset = dataset[:train_rows + test_rows]
    print(f'dataset is split into {train_rows} train rows and {test_rows} test rows')
    final_test_mask = np.arange(train_rows + test_rows) >= train_rows
    return dataset, final_test_mask
