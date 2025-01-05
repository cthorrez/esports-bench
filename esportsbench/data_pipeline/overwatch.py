import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr


class OverwatchDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing overwatch data from LPDB"""

    game = 'overwatch'
    version = 'v3'
    request_params_groups = {
        'overwatch.jsonl': {
            'wiki': 'overwatch',
            'query': 'date, match2opponents, winner, game, mode, resulttype, walkover, match2id, pagename',
            'conditions': '[[liquipediatier::!-1]] AND [[finished::1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]]',
            'order': 'date ASC, match2id ASC',
        }
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, to_lowercase=False, **kwargs):
        self.to_lowercase = to_lowercase
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / 'overwatch.jsonl', infer_schema_length=100000).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        # filter out matches without exactly 2 teams
        not_two_teams_expr = pl.col('match2opponents').list.len() != 2
        df = self.filter_invalid(df, not_two_teams_expr, 'not_two_teams', drop_cols=['match2opponents'])

        # extract team names and scores
        df = df.with_columns(
            pl.col('match2opponents').list.get(0).alias('team_1_struct'),
            pl.col('match2opponents').list.get(1).alias('team_2_struct'),
        ).drop('match2opponents')
        df = df.with_columns(
            pl.col('team_1_struct').struct.field('name').alias('team_1_name'),
            pl.col('team_1_struct').struct.field('template').alias('team_1_template'),
            pl.col('team_1_struct').struct.field('teamtemplate').struct.field('name').alias('team_1_template_name'),
            pl.col('team_1_struct').struct.field('teamtemplate').struct.field('page').alias('team_1_template_page'),
            pl.col('team_1_struct').struct.field('score').cast(pl.Float64).alias('team_1_score'),
            pl.col('team_2_struct').struct.field('name').alias('team_2_name'),
            pl.col('team_2_struct').struct.field('template').alias('team_2_template'),
            pl.col('team_2_struct').struct.field('teamtemplate').struct.field('name').alias('team_2_template_name'),
            pl.col('team_2_struct').struct.field('teamtemplate').struct.field('page').alias('team_2_template_page'),
            pl.col('team_2_struct').struct.field('score').cast(pl.Float64).alias('team_2_score'),
        ).drop('team_1_struct', 'team_2_struct')

        # use name if it is not null, use template otherwise
        df = df.with_columns(
            pl.when(~is_null_or_empty('team_1_template_name'))
            .then(pl.col('team_1_template_name'))
            .when(is_null_or_empty('team_1_name') & ~is_null_or_empty('team_1_template'))
            .then(pl.col('team_1_template'))
            .otherwise(pl.col('team_1_name'))
            .alias('team_1'),
            pl.when(~is_null_or_empty('team_2_template_name'))
            .then(pl.col('team_2_template_name'))
            .when(is_null_or_empty('team_2_name') & ~is_null_or_empty('team_2_template'))
            .then(pl.col('team_2_template'))
            .otherwise(pl.col('team_2_name'))
            .alias('team_2'),
        )

        if self.to_lowercase:
            df = df.with_columns(
                pl.col('team_1').str.to_lowercase().alias('team_1'),
                pl.col('team_2').str.to_lowercase().alias('team_2'),
            )

        invalid_competitor_expr = (
            pl.col('team_1').str.to_lowercase().is_in(self.invalid_competitor_names) 
            | pl.col('team_2').str.to_lowercase().is_in(self.invalid_competitor_names)
        )
        df = self.filter_invalid(df, invalid_competitor_expr, 'invalid_competitor')


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

        played_self_expr = pl.col('team_1') == pl.col('team_2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        df = (
            df.select(
                'date',
                pl.col('team_1').alias('competitor_1'),
                pl.col('team_2').alias('competitor_2'),
                pl.col('team_1_score').alias('competitor_1_score'),
                pl.col('team_2_score').alias('competitor_2_score'),
                'outcome',
                pl.col('match2id').alias('match_id'),
                (pl.lit(self.page_prefix) + pl.col('pagename')).alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )
        return df
