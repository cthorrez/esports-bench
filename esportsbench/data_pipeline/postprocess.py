import os
import argparse
from datetime import datetime
import polars as pl
import pathlib
from esportsbench.constants import GAME_NAME_MAP

pl.Config.set_tbl_rows(100)


def main(end_date, min_rows_year):
    data_dir = pathlib.Path(__file__).resolve().parents[2] / 'data'
    input_file_paths = data_dir.glob('full_data/*')
    for input_file_path in input_file_paths:
        game = input_file_path.name
        print(f'summary for {game}:')
        df = pl.read_csv(input_file_path)
        print(f'num total matches: {len(df)}')
        df = df.with_columns(pl.col('date').str.to_datetime().alias('date'))
        df = df.with_columns(pl.col('date').dt.year().alias('year'))
        future_df = df.filter(pl.col('date') >= datetime.now())
        print('num future matches:', len(future_df))
        year_counts = df.group_by('year').count().sort('year')
        first_valid_year = year_counts.filter(
            pl.col('count') >= min_rows_year
        ).select('year').min().item()
        print(f'{first_valid_year=}')
        # print(year_counts)
        df = df.with_columns(
            pl.col('date').dt.date().cast(pl.Utf8).alias('date')
        )
        df = df.filter(
            (pl.col('year') >= first_valid_year)
            & (pl.col('date') <= end_date)
        )
        print(f'filtered row count: {len(df)}')

        output_file_path = data_dir / 'final_data' / input_file_path.name
        df.write_csv(output_file_path)

        print('======================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_date', type=str, default='2024-03-31', help='inclusive end date for final data')
    parser.add_argument('--min_rows_year', type=int, default=100, help='minmum number of rows in a year to begin including data')
    args = vars(parser.parse_args())
    main(**args)
