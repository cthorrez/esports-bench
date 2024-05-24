import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr


class OverwatchDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing overwatch data from LPDB"""

    game = 'overwatch'
    version = 'v1'
    request_params_groups = {
        'overwatch.jsonl': {
            'wiki': 'overwatch',
            'query': 'date, opponent1, opponent2, opponent1score, opponent2score, winner, game, status, mode, resulttype, walkover, matchid, pagename',
            'conditions': '[[liquipediatier::!-1]] AND [[finished::1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[opponent1::!Bye]] AND [[opponent2::!Bye]] AND [[opponent1::!TBD]] AND [[opponent2::!TBD]]',
            'order': 'date ASC, matchid ASC',
        }
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / 'overwatch.jsonl', infer_schema_length=1).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        df = df.with_columns(
            pl.col('opponent1score').cast(pl.Float64).alias('team_1_score'),
            pl.col('opponent2score').cast(pl.Float64).alias('team_2_score'),
        )

        missing_team_expr = is_null_or_empty('opponent1') | is_null_or_empty('opponent2')
        df = self.filter_invalid(df, missing_team_expr, 'missing_team')

        did_not_play_expr = (pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & is_null_or_empty('winner')
        df = self.filter_invalid(df, did_not_play_expr, 'did_not_play')

        df = df.with_columns(
            pl.when(pl.col('team_1_score') > pl.col('team_2_score'))
            .then(1.0)
            .when(pl.col('team_1_score') < pl.col('team_2_score'))
            .then(0.0)
            .when(
                ~is_null_or_empty('team_1_score')
                & ~is_null_or_empty('team_2_score')
                & (pl.col('team_1_score') == pl.col('team_2_score'))
            )
            .then(0.5)
            .otherwise(None)
            .alias('score_outcome')
        )

        df = df.with_columns(
            pl.when(pl.col('winner') == '1')
            .then(1.0)
            .when(pl.col('winner') == '2')
            .then(0.0)
            .when(pl.col('winner') == '0')
            .then(0.5)
            .when(~is_null_or_empty('score_outcome'))
            .then(pl.col('score_outcome'))
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
                pl.col('team_1_score').alias('competitor_1_score'),
                pl.col('team_2_score').alias('competitor_2_score'),
                'outcome',
                pl.col('matchid').alias('match_id'),
                pl.col('pagename').alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )

        print(f'final row count: {df.shape[0]}')
        df.write_csv(self.final_data_path)
