"""utility functions for dealing with match data"""
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
    """returns a polars expression for is the column is null or empty string"""
    if isinstance(col, str):
        col = pl.col(col)
    return col.is_null() | (col.cast(pl.Utf8) == '')


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
