"""module for managing esports datasets for rating system experiments"""
import re
import json
import numpy as np
import polars as pl
from riix.utils.data_utils import TimedPairDataset
from esportsbench.constants import GAME_NAME_MAP
from esportsbench.paths import DATA_DIR

BASE_DATA_DIR = DATA_DIR
RELEASE_INFO_FILE = 'release.json'


def release_versions():
    """versions with a final_data_v{version} directory, ascending"""
    versions = [int(match.group(1)) for path in DATA_DIR.glob('final_data_v*') if (match := re.fullmatch(r'final_data_v(\d+)', path.name))]
    return sorted(versions)


def read_release_info(version):
    """the release.json written for final_data_v{version}, None if there isn't one"""
    path = DATA_DIR / f'final_data_v{version}' / RELEASE_INFO_FILE
    return json.load(open(path)) if path.exists() else None


def latest_release_info():
    """release.json of the newest release which has one"""
    for version in reversed(release_versions()):
        info = read_release_info(version)
        if info is not None:
            return info
    raise FileNotFoundError(
        f'no {DATA_DIR}/final_data_v*/{RELEASE_INFO_FILE} found, make a release with '
        'python -m esportsbench.release or pass data_dir and the train/test end dates explicitly'
    )


def resolve_data_args(data_dir=None, train_end_date=None, test_end_date=None):
    """fill in data_dir and the train/test end dates from the latest release where not given"""
    if data_dir in (None, 'latest') or train_end_date is None or test_end_date is None:
        info = latest_release_info() if data_dir in (None, 'latest') else read_release_info(data_dir.removeprefix('final_data_v'))
        if info is None:
            raise FileNotFoundError(f'{data_dir} has no {RELEASE_INFO_FILE}, pass train_end_date and test_end_date')
        data_dir = info['data_dir']
        train_end_date = train_end_date or info['train_end_date']
        test_end_date = test_end_date or info['test_end_date']
    return data_dir, train_end_date, test_end_date


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
