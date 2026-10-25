import argparse
import tempfile
import polars as pl
from datasets import load_dataset, Features, Value
import huggingface_hub
from esportsbench.constants import GAME_NAMES
from esportsbench.paths import DATA_DIR

TEST_REPO = 'cthorrez/EsportsBenchTest'
PROD_REPO = 'EsportsBench/EsportsBench'
FEATURES = Features({
    'date': Value('date32'),
    'competitor_1': Value('string'),
    'competitor_2': Value('string'),
    'outcome': Value('float64'),
    'match_id': Value('string'),
    'page': Value('string'),
})


def release_tag(version):
    return f'{int(version)}.0'


def load_release_dataset(version):
    """the final_data_v{version} parquet files as a DatasetDict with one split per game"""
    return load_dataset(
        path=str(DATA_DIR / f'final_data_v{version}' / 'parquet'),
        data_files={game_name: f'{game_name}.parquet' for game_name in GAME_NAMES},
        features=FEATURES,
    )


def push_release(version, repo_id):
    dataset = load_release_dataset(version)
    print('Dataset loaded.')
    dataset.save_to_disk(DATA_DIR / f'EsportsBench{version}')
    print('Dataset saved to disk.')
    dataset.push_to_hub(repo_id, commit_message=f'update to {release_tag(version)}')
    print(f'Dataset pushed to {repo_id}.')
    huggingface_hub.create_tag(repo_id=repo_id, tag=release_tag(version), repo_type='dataset')
    print(f'Tag {release_tag(version)} applied')


def verify_release(version, repo_id):
    """re-download the pushed tag and check every split matches the local release, returns True if they all do"""
    with tempfile.TemporaryDirectory() as cache_dir:
        remote = load_dataset(repo_id, revision=release_tag(version), cache_dir=cache_dir)
        all_match = True
        for game in GAME_NAMES:
            remote_df = pl.from_arrow(remote[game].data.table).with_columns(pl.col('date').cast(pl.Utf8))
            local_df = pl.read_parquet(DATA_DIR / f'final_data_v{version}' / 'parquet' / f'{game}.parquet')
            # match_id is an integer for some games locally, FEATURES stores it as a string
            local_df = local_df.with_columns(pl.col('match_id').cast(pl.Utf8))
            matches = remote_df.equals(local_df)
            all_match &= matches
            print(f'{game:18s} {len(remote_df):8d} rows {"ok" if matches else "MISMATCH"}')
    print(f'{repo_id}@{release_tag(version)}: {"all splits match" if all_match else "SOME SPLITS DO NOT MATCH"}')
    return all_match


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=int, required=True)
    parser.add_argument('--prod', action='store_true')
    args = parser.parse_args()
    repo = PROD_REPO if args.prod else TEST_REPO
    push_release(args.version, repo)
    verify_release(args.version, repo)
