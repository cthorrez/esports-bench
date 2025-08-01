import os
import argparse
from datetime import datetime
import polars as pl
import pathlib
from esportsbench.constants import GAME_NAME_MAP

pl.Config.set_tbl_rows(100)


def postprocess(train_end_date, test_end_date, min_rows_year, version):
    data_dir = pathlib.Path(__file__).resolve().parents[2] / 'data'
    input_file_paths = sorted(data_dir.glob('full_data/parquet/*'))
    print('')
    for input_file_path in input_file_paths:
        game = input_file_path.stem
        print(f'summary for {game}:')
        df = pl.read_parquet(input_file_path).drop('competitor_1_score', 'competitor_2_score')
        if 'game' in df.columns:
            df = df.drop('game')
        print(f'num total matches: {len(df)}')
        first_date = df.select('date').min().item()[:10]
        last_date = df.select('date').max().item()[:10]
        print(f'input date range: {first_date} to {last_date}')

        comp_1_error = df.select('competitor_1').filter(pl.col('competitor_1').str.contains('<div class="error">')).count().item()
        comp_2_error = df.select('competitor_2').filter(pl.col('competitor_2').str.contains('<div class="error">')).count().item()

        print(f'comp_1 error count: {comp_1_error}')
        print(f'comp_2 error count: {comp_2_error}')

        df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
        df = df.with_columns(pl.col('date').dt.year().alias('year'))
        year_counts = df.group_by('year').len().sort('year')
        first_valid_year = year_counts.filter(
            pl.col('len') >= min_rows_year
        ).select('year').min().item()
        print(f'first year with at least {min_rows_year} rows: {first_valid_year}')
        df = df.with_columns(
            pl.col('date').dt.date().cast(pl.Utf8).alias('date')
        )
        future_df = df.filter(pl.col('date') > datetime.today().strftime("%Y-%m-%d"))
        print('num future matches:', len(future_df))

        df = df.filter(
            (pl.col('year') >= first_valid_year)
            & (pl.col('date') <= test_end_date)
        )
        print(df.columns)
        print(f'filtered row count: {len(df)}')
        unique_competitors = pl.concat([
            df["competitor_1"],
            df["competitor_2"]
        ]).unique()
        print(f'num unique competitors: {unique_competitors.count()}')

        draw_rate = df.select(pl.col('outcome')== 0.5).mean().item()
        print(f'Draw rate: {draw_rate}')
        print(f'mean outcome: {df.select("outcome").mean().item()}')
        first_date = df.select('date').min().item()[:10]
        last_date = df.select('date').max().item()[:10]
        print(f'output date range: {first_date} to {last_date}')

        train_df = df.filter(
            (pl.col('date') <= train_end_date)
        )
        test_df = df.filter(
            (pl.col('date') > train_end_date)
            & (pl.col('date') <= test_end_date)
        )

        print(f'num train rows: {len(train_df)}')
        print(f'num test rows: {len(test_df)}')

        output_dir = data_dir / f'final_data_v{version}'
        os.makedirs(output_dir / 'csv', exist_ok=True)
        os.makedirs(output_dir / 'parquet', exist_ok=True)
        output_csv_path = output_dir / 'csv' / f'{game}.csv'
        output_parquet_path = output_dir / 'parquet' / f'{game}.parquet'
        df = df.drop('year')
        df.write_csv(output_csv_path)
        df.write_parquet(output_parquet_path)
        print('======================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_end_date', type=str, default='2024-06-30', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2025-06-30', help='inclusive end date for test set')
    parser.add_argument('--min_rows_year', type=int, default=100, help='minmum number of rows in a year to begin including data')
    parser.add_argument('--version', '-v', type=str, default='6', help='which version of the dataset')
    args = vars(parser.parse_args())
    postprocess(**args)
