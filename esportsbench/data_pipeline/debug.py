import polars as pl
from datasets import load_dataset

def main():
    v2 = load_dataset('EsportsBench/EsportsBench', split='starcraft1').to_polars()
    cur = pl.read_csv('../../data/full_data/starcraft1.csv').filter(pl.col('date') <= '2024-06-30')
    cur = cur.with_columns(
        pl.col('date').str.slice(0, 10).alias('date')
    ).drop('competitor_1_score', 'competitor_2_score')

    print(len(v2))
    print(len(cur))

    diff = v2.join(cur, on=v2.columns, how="anti")

    print(len(diff))
    diff.write_csv("diff_output.csv")

if __name__ == '__main__':
    main()