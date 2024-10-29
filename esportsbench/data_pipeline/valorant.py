import json
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr


class ValorantDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing valorant data from LPDB"""

    game = 'valorant'
    version = 'v3'
    schema_overrides = {'extradata': json.dumps}
    request_params_groups = {
        'valorant.jsonl': {
            'wiki': 'valorant',
            'query': 'date, match2opponents, winner, resulttype, extradata, finished, bestof',
            'conditions': '[[mode::team]] AND [[winner::!-1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]] AND [[section::!Showmatch]] AND [[liquipediatiertype::!Showmatch]]',
            'order': 'date ASC, match2id ASC',
        }
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def process_data(self):
        df = pl.scan_ndjson(
            self.raw_data_dir / 'valorant.jsonl', infer_schema_length=75369, ignore_errors=True
        ).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        # filter out matches without exactly 2 teams
        not_two_teams_expr = pl.col('match2opponents').list.len() != 2
        df = self.filter_invalid(df, not_two_teams_expr, 'not_two_teams')

        # extract team names and scores
        df = df.with_columns(
            pl.col('match2opponents').list.get(0).alias('team_1_struct'),
            pl.col('match2opponents').list.get(1).alias('team_2_struct'),
        )
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
        )

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

        # filter out matches with placeholder teams or without any results recorded
        placeholder_expr = (pl.col('team_1') == 'TBD') & (pl.col('team_2') == 'TBD')
        df = self.filter_invalid(df, placeholder_expr, 'placeholder')

        missing_team_expr = is_null_or_empty('team_1') | is_null_or_empty('team_2')
        df = self.filter_invalid(df, missing_team_expr, 'missing_team')

        missing_results_expr = (
            ((pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1) & (is_null_or_empty('winner') | (pl.col('winner') == '0')))
            | ((pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & (is_null_or_empty('winner') | (pl.col('winner') == '0')))
        )
        df = self.filter_invalid(df, missing_results_expr, 'missing_results')

        # logic for outcome column
        df = df.with_columns(
            pl.when(pl.col('winner') == '1')
            .then(1.0)
            .when(pl.col('winner') == '2')
            .then(0.0)
            .when((pl.col('winner') == '0') & (pl.col('resulttype') == 'draw'))
            .then(0.5)
            .otherwise(None)
            .alias('outcome')
        )

        # best of 1 logic, score is initially in rounds so set it to be game score
        # team_1_score is the same as outcome, team_2_score is 1 - outcome
        team_1_score_expr = (
            pl.when(pl.col('bestof') == 1)
            .then(pl.col('outcome'))
            .otherwise(pl.col('team_1_score'))
            .alias('competitor_1_score')
        )
        team_2_score_expr = (
            pl.when((pl.col('bestof') == 1))
            .then(1.0 - pl.col('outcome'))
            .otherwise(pl.col('team_2_score'))
            .alias('competitor_2_score')
        )

        played_self_expr = pl.col('team_1') == pl.col('team_2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        # select final columns and write to csv
        df = (
            df.select(
                'date',
                pl.col('team_1').alias('competitor_1'),
                pl.col('team_2').alias('competitor_2'),
                team_1_score_expr,
                team_2_score_expr,
                'outcome',
                pl.col('match2id').alias('match_id'),
                (pl.lit(self.page_prefix) + pl.col('pagename')).alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )

        print(f'valid row count: {df.shape[0]}')
        df.write_csv(self.full_data_path)
