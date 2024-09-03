import polars as pl
from datasets import load_dataset
from datetime import timedelta

def find_diff(v2: pl.DataFrame, cur: pl.DataFrame) -> pl.DataFrame:
    # Convert date columns to datetime
    v2 = v2.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))
    cur = cur.with_columns(pl.col("date").str.to_datetime("%Y-%m-%d"))

    # Create a date range for joining
    v2 = v2.with_columns([
        (pl.col("date") - pl.duration(days=1)).alias("date_lower"),
        (pl.col("date") + pl.duration(days=1)).alias("date_upper")
    ])

    # Perform a cross join
    joined = v2.join(
        cur,
        how="cross"
    )

    # Filter for potential matches
    matches = joined.filter(
        (pl.col("competitor_1") == pl.col("competitor_1_right")) &
        (pl.col("competitor_2") == pl.col("competitor_2_right")) &
        (pl.col("outcome") == pl.col("outcome_right")) &
        (pl.col("date_right") >= pl.col("date_lower")) &
        (pl.col("date_right") <= pl.col("date_upper"))
    )

    # Group by the original v2 rows and check if there are any matches
    v2_with_match_flag = v2.join(
        matches.group_by(v2.columns).agg(
            pl.count().alias("match_count")
        ),
        how="left",
        on=v2.columns
    ).with_columns(
        pl.col("match_count").fill_null(0)
    )

    # Filter for rows with no matches
    diff = v2_with_match_flag.filter(pl.col("match_count") == 0)

    # Select only the original columns from v2 and convert date back to string
    diff = diff.select(
        v2.columns
    ).with_columns(
        pl.col("date").dt.strftime("%Y-%m-%d")
    )

    return diff

def main():
    v2 = load_dataset('EsportsBench/EsportsBench', split='starcraft1').to_polars().lazy()
    cur = pl.scan_csv('../../data/full_data/starcraft1.csv').filter(pl.col('date') <= '2024-06-30')
    cur = cur.with_columns(
        pl.col('date').str.slice(0, 10).alias('date')
    ).drop('competitor_1_score', 'competitor_2_score')

    # print(len(v2))
    # print(len(cur))

    diff = find_diff(v2, cur).collect()
    # print(len(diff))
    diff.write_csv("diff_output.csv")

if __name__ == '__main__':
    main()