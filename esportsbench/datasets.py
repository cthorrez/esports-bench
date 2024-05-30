"""module for managing esports datasets for rating system experiments"""
import pathlib
import pandas as pd
from riix.utils.data_utils import MatchupDataset
from esportsbench.constants import GAME_NAME_MAP

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'final_data'


def load_dataset(
    game,
    rating_period='7D',
    drop_draws=False,
    max_rows=None,
    train_end_date='2023-03-31',
    test_end_date='2024-03-31',
):
    # map short name to full name if short name is provided
    if game in GAME_NAME_MAP:
        game = GAME_NAME_MAP[game]
    df = pd.read_csv(DATA_DIR / f'{game}.csv')
    if drop_draws:
        df = df[df['outcome'] != 0.5].reset_index()
    if max_rows:
        df = df.head(max_rows).reset_index()
    test_mask = (df['date'] > train_end_date) & (df['date'] <= test_end_date)
    dataset = MatchupDataset(
        df=df,
        competitor_cols=['competitor_1', 'competitor_2'],
        outcome_col='outcome',
        datetime_col='date',
        rating_period=rating_period,
    )
    test_rows = int(test_mask.sum())
    train_rows = len(dataset) - test_rows
    print(f'dataset is split into {train_rows} train rows and {test_rows} test rows')
    return dataset, test_mask
