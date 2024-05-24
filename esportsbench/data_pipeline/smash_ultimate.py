import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import invalid_date_expr


class SmashUltimateDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing raw ssbu data from LPDB"""

    game = 'smash_ultimate'
    version = 'v1'
    request_params_groups = {
        'smash_ultimate.jsonl': {
            'wiki': 'smash',
            'query': 'date, opponent1, opponent2, opponent1score, opponent2score, winner, matchid, pagename',
            'conditions': '[[walkover::!1]] AND [[walkover::!2]] AND [[mode::singles]] AND [[game::ultimate]] AND [[opponent1::!Bye]] AND [[opponent2::!Bye]]',
            'order': 'date ASC, matchid ASC',
        }
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / 'smash_ultimate.jsonl', infer_schema_length=1).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        df = df.with_columns(
            pl.when(pl.col('winner') == pl.col('opponent1'))
            .then(1.0)
            .when(pl.col('winner') == pl.col('opponent2'))
            .then(0.0)
            .otherwise(None)
            .alias('outcome')
        )
        null_outcome_expr = pl.col('outcome').is_null()
        df = self.filter_invalid(df, null_outcome_expr, 'null_outcome')

        played_self_expr = pl.col('opponent1') == pl.col('opponent2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        df = (
            df.select(
                'date',
                pl.col('opponent1').alias('competitor_1'),
                pl.col('opponent2').alias('competitor_2'),
                pl.col('opponent1score').cast(pl.Float64).alias('competitor_1_score'),
                pl.col('opponent2score').cast(pl.Float64).alias('competitor_2_score'),
                'outcome',
                pl.col('matchid').alias('match_id'),
                pl.col('pagename').alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )

        print(f'final row count: {df.shape[0]}')
        df.write_csv(self.final_data_path)
