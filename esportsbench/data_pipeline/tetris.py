import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr


class TetrisDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing tetris data from LPDB"""

    game = 'tetris'
    version = 'v3'
    request_params_groups = {
        'tetris.jsonl': {
            'wiki': 'tetris',
            'query': 'date, match2opponents, winner, resulttype, extradata, finished, bestof, match2id',
            'conditions': '[[game::classic]] AND [[mode::individual]] AND [[finished::1]] AND [[walkover::!1]] AND [[walkover::!2]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    placeholder_player_names = {'bye', 'tbd'}

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / 'tetris.jsonl', infer_schema_length=1).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        # filter out matches without exactly 2 players
        not_two_players_expr = pl.col('match2opponents').list.len() != 2
        df = self.filter_invalid(df, not_two_players_expr, 'not_two_players')

        # extract team names and scores
        df = df.with_columns(
            pl.col('match2opponents').list.get(0).alias('player_1_struct'),
            pl.col('match2opponents').list.get(1).alias('player_2_struct'),
        )

        df = df.with_columns(
            pl.col('player_1_struct').struct.field('name').alias('player_1_name'),
            pl.col('player_1_struct').struct.field('template').alias('player_1_template'),
            pl.col('player_1_struct').struct.field('score').cast(pl.Float64).alias('player_1_score'),
            pl.col('player_1_struct').struct.field('status').alias('player_1_status'),
            pl.col('player_2_struct').struct.field('name').alias('player_2_name'),
            pl.col('player_2_struct').struct.field('template').alias('player_2_template'),
            pl.col('player_2_struct').struct.field('score').cast(pl.Float64).alias('player_2_score'),
            pl.col('player_2_struct').struct.field('status').alias('player_2_status'),
        )

        did_not_play_expr = (
            (pl.col('player_1_score') == 0) & (pl.col('player_2_score') == 0) & is_null_or_empty('winner')
        )
        df = self.filter_invalid(df, did_not_play_expr, 'did_not_play')

        df = df.with_columns(
            pl.when(pl.col('player_1_score') > pl.col('player_2_score'))
            .then(1.0)
            .when(pl.col('player_1_score') < pl.col('player_2_score'))
            .then(0.0)
            .when(
                ~is_null_or_empty('player_1_score')
                & ~is_null_or_empty('player_2_score')
                & (pl.col('player_1_score') == pl.col('player_2_score'))
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

        # use name if it is not null, use template otherwise
        df = df.with_columns(
            pl.when(~is_null_or_empty('player_1_name'))
            .then(pl.col('player_1_name').str.to_lowercase())
            .when(~is_null_or_empty('player_1_template'))
            .then(pl.col('player_1_template').str.to_lowercase())
            .otherwise(None)
            .alias('player_1'),
            pl.when(~is_null_or_empty('player_2_name'))
            .then(pl.col('player_2_name').str.to_lowercase())
            .when(~is_null_or_empty('player_2_template'))
            .then(pl.col('player_2_template').str.to_lowercase())
            .otherwise(None)
            .alias('player_2'),
        )

        placeholder_expr = pl.col('player_1').str.to_lowercase().is_in(self.placeholder_player_names) | pl.col(
            'player_2'
        ).str.to_lowercase().is_in(self.placeholder_player_names)
        df = self.filter_invalid(df, placeholder_expr, 'placeholder')

        played_self_expr = pl.col('player_1') == pl.col('player_2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        # select final columns and write to csv
        df = (
            df.select(
                'date',
                pl.col('player_1').alias('competitor_1'),
                pl.col('player_2').alias('competitor_2'),
                pl.col('player_1_score').alias('competitor_1_score'),
                pl.col('player_2_score').alias('competitor_2_score'),
                'outcome',
                pl.col('match2id').alias('match_id'),
                (pl.lit(self.page_prefix) + pl.col('pagename')).alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )
        return df
