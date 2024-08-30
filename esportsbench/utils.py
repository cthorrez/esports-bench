"""utility functions for dealing with match data"""
from typing import Tuple, Optional
import polars as pl


def outcome_from_scores(score_1, score_2):
    if score_1 > score_2:
        return 1.0
    if score_1 < score_2:
        return 0.0
    return 0.5

def bestof_from_scores(score_1, score_2):
    if score_1 != score_2:
        return int((max(score_1, score_2) * 2) - 1)
    elif (int(score_1) != -1) and (int(score_2) != -1):
        return int(max(score_1, score_2) * 2)
    else:
        return 1


def float_nullable(value):
    if value is None:
        return None
    return value


def is_null_or_empty(col):
    """returns a polars expression for is the column is null, empty string(or string False for some reason 😑"""
    if isinstance(col, str):
        col = pl.col(col)
    return col.is_null() | (col.cast(pl.Utf8) == '') | (col.cast(pl.Utf8) == 'False')


def delimited_list(string_list, delimiter=','):
    return string_list.split(delimiter)


# lots of ways a date can be bad lol
invalid_date_expr = (
    pl.col('date').is_null()
    | (pl.col('date') == '')
    | (pl.col('date').str.to_datetime().dt.year() == 0)
    | (pl.col('date').str.to_datetime().dt.year() == 1000)
    | (pl.col('date').str.to_datetime().dt.year() == 1970)
)

def debug_ndjson(file_path: str) -> Tuple[Optional[int], Optional[str]]:
    def attempt_read(n: int) -> bool:
        try:
            # pl.read_ndjson(file_path, n_rows=n)
            pl.scan_ndjson(file_path, n_rows=n, infer_schema_length=n).collect()

            return True
        except Exception as e:
            print(e)
            return False

    with open(file_path, 'r') as f:
        total_lines = sum(1 for _ in f)

    left, right = 1, total_lines
    last_failing = None

    while left <= right:
        mid = (left + right) // 2
        if attempt_read(mid):
            left = mid + 1
        else:
            last_failing = mid
            right = mid - 1

    if last_failing is None:
        return None, "No errors found in the file."

    # Find the exact failing line
    for i in range(max(1, last_failing - 10), last_failing + 1):
        if not attempt_read(i):
            # Read the problematic line
            with open(file_path, 'r') as f:
                for _ in range(i - 1):
                    next(f)
                problematic_line = next(f).strip()
            return i, f"Error on line {i}: {problematic_line}"

    return None, "Could not pinpoint the exact error location."
    