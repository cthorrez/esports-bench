"""make a new EsportsBench release with one command

    python -m esportsbench.release --as-of 2026-12-31                # refresh all data, build and check the next final_data_v{N}
    python -m esportsbench.release --as-of 2026-12-31 --push test    # ... then push it to the test repo and verify the upload
    python -m esportsbench.release --as-of 2026-12-31 --push prod    # ... test repo first, then prod

the test set is the year ending at --as-of (inclusive) and the train set is everything before it.
a release.json is written next to the data, bench.py and the sweeps default to the newest release.
"""
import argparse
import datetime
import json
import subprocess
import polars as pl
from esportsbench.paths import DATA_DIR
from esportsbench.constants import GAME_NAMES, GAME_SHORT_NAMES
from esportsbench.datasets import release_versions, RELEASE_INFO_FILE

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(200)
pl.Config.set_tbl_hide_dataframe_shape(True)
pl.Config.set_tbl_hide_column_data_types(True)

# a game losing more than this fraction of the rows the previous release had fails validation
MAX_SHRINK_ERROR = 0.05
MAX_SHRINK_WARNING = 0.005
# a game whose newest match is older than this many days before --as-of gets a warning
STALE_DAYS = 45


def one_year_before(date_str):
    date = datetime.date.fromisoformat(date_str)
    try:
        return date.replace(year=date.year - 1).isoformat()
    except ValueError:  # Feb 29
        return date.replace(year=date.year - 1, day=28).isoformat()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, cwd=DATA_DIR.parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_release_info(version, as_of, train_end_date, test_end_date, created_at=None):
    release_dir = DATA_DIR / f'final_data_v{version}'
    games = {}
    for game in GAME_NAMES:
        df = pl.read_parquet(release_dir / 'parquet' / f'{game}.parquet')
        games[game] = {
            'rows': len(df),
            'train_rows': df.filter(pl.col('date') <= train_end_date).height,
            'test_rows': df.filter(pl.col('date') > train_end_date).height,
            'first_date': df['date'].min(),
            'last_date': df['date'].max(),
        }
    info = {
        'version': version,
        'data_dir': release_dir.name,
        'as_of': as_of,
        'train_end_date': train_end_date,
        'test_end_date': test_end_date,
        'created_at': created_at or datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': git_commit(),
        'games': games,
    }
    with open(release_dir / RELEASE_INFO_FILE, 'w') as out_file:
        json.dump(info, out_file, indent=2)
    return info


def validate_release(version, previous_version, as_of):
    """sanity check a release against the previous one, prints a table and returns the list of errors"""
    release_dir = DATA_DIR / f'final_data_v{version}' / 'parquet'
    previous_dir = DATA_DIR / f'final_data_v{previous_version}' / 'parquet' if previous_version else None
    stale_date = (datetime.date.fromisoformat(as_of) - datetime.timedelta(days=STALE_DAYS)).isoformat()
    errors, warnings, rows = [], [], []
    for game in GAME_NAMES:
        path = release_dir / f'{game}.parquet'
        if not path.exists():
            errors.append(f'{game}: missing from the release')
            continue
        df = pl.read_parquet(path)
        row = {'game': game, 'rows': len(df), 'test_rows': df.filter(pl.col('date') > one_year_before(as_of)).height}
        if len(df) == 0:
            errors.append(f'{game}: no rows')
            continue
        nulls = df.select('date', 'competitor_1', 'competitor_2', 'outcome').null_count().sum_horizontal().item()
        if nulls:
            errors.append(f'{game}: {nulls} null values in date/competitors/outcome')
        bad_outcomes = df.filter(~pl.col('outcome').is_in([0.0, 0.5, 1.0])).height
        if bad_outcomes:
            errors.append(f'{game}: {bad_outcomes} outcomes not in (0, 0.5, 1)')
        row['newest'] = df['date'].max()
        if row['newest'] < stale_date:
            warnings.append(f'{game}: newest match is {row["newest"]}, more than {STALE_DAYS} days before {as_of}')
        row['dup_match_ids'] = df['match_id'].is_duplicated().sum()

        if previous_dir is not None and (previous_dir / f'{game}.parquet').exists():
            previous = pl.read_parquet(previous_dir / f'{game}.parquet')
            # rows the previous release covered should mostly still be there, losing many means data went missing
            overlap = df.filter(pl.col('date') <= previous['date'].max()).height
            row['prev_rows'] = len(previous)
            row['change_in_prev_range'] = overlap - len(previous)
            shrink = (len(previous) - overlap) / len(previous)
            if shrink > MAX_SHRINK_ERROR:
                errors.append(f'{game}: {len(previous) - overlap} fewer rows than v{previous_version} over the dates it covered ({shrink:.1%})')
            elif shrink > MAX_SHRINK_WARNING:
                warnings.append(f'{game}: {len(previous) - overlap} fewer rows than v{previous_version} over the dates it covered ({shrink:.1%})')
        rows.append(row)

    print(pl.DataFrame(rows))
    for warning in warnings:
        print(f'WARNING {warning}')
    for error in errors:
        print(f'ERROR {error}')
    return errors


def main(as_of, version=None, push=None, skip_ingest=False, full=False, overlap_days=90, overwrite=False, force=False):
    existing = release_versions()
    version = version or (max(existing) + 1 if existing else 1)
    previous_version = max((v for v in existing if v < version), default=None)
    if version in existing and not overwrite:
        raise SystemExit(f'final_data_v{version} already exists, pass --overwrite to rebuild it')
    if as_of > datetime.date.today().isoformat():
        raise SystemExit(f'--as-of {as_of} is in the future')
    train_end_date, test_end_date = one_year_before(as_of), as_of
    print(f'building v{version}: train <= {train_end_date} < test <= {test_end_date}, comparing against v{previous_version}')

    # imported here so the release metadata helpers above don't need api keys
    from esportsbench.data_pipeline.get_data_from_source import run_pipeline
    from esportsbench.data_pipeline.postprocess import postprocess
    from esportsbench.data_pipeline.push_to_hf_hub import push_release, verify_release, TEST_REPO, PROD_REPO

    run_pipeline(GAME_SHORT_NAMES, action='process' if skip_ingest else 'all', full=full, overlap_days=overlap_days)
    postprocess(train_end_date, test_end_date, min_rows_year=100, version=version)
    write_release_info(version, as_of, train_end_date, test_end_date)
    errors = validate_release(version, previous_version, as_of)
    if errors and not force:
        raise SystemExit(f'v{version} failed validation, fix the errors above or rerun with --force')
    print(f'v{version} written to {DATA_DIR / f"final_data_v{version}"}')

    if push in ('test', 'prod'):
        push_release(version, TEST_REPO)
        if not verify_release(version, TEST_REPO):
            raise SystemExit(f'the {TEST_REPO} upload does not match, not pushing further')
    if push == 'prod':
        push_release(version, PROD_REPO)
        if not verify_release(version, PROD_REPO):
            raise SystemExit(f'the {PROD_REPO} upload does not match')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--as-of', dest='as_of', required=True, help='last date of the test set, YYYY-MM-DD')
    parser.add_argument('--version', type=int, help='default: one more than the newest final_data_v*')
    parser.add_argument('--push', choices=['test', 'prod'], help='push to the HF test repo, or test then prod')
    parser.add_argument('--skip-ingest', dest='skip_ingest', action='store_true', help="reprocess the stored raw data without fetching")
    parser.add_argument('--full', action='store_true', help='re-fetch all history instead of refreshing recent matches (slow)')
    parser.add_argument('--overlap-days', dest='overlap_days', type=int, default=90)
    parser.add_argument('--overwrite', action='store_true', help='rebuild an existing version')
    parser.add_argument('--force', action='store_true', help='continue even if validation fails')
    main(**vars(parser.parse_args()))
