import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr


class CounterStrikeDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing counterstrike data from LPDB"""

    game = 'counterstrike'
    version = 'v3'
    request_params_groups = {
        'counterstrike.jsonl': {
            'wiki': 'counterstrike',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof',
            'conditions': '[[mode::team]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]] AND [[section::!Showmatch]] AND [[liquipediatiertype::!Showmatch]]',
            'order': 'date ASC, match2id ASC',
        }
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def process_data(self):
        """read data from raw jsonl, filter invalid data, process and write to final data  location"""

        df = pl.scan_ndjson(
            self.raw_data_dir / 'counterstrike.jsonl',
            infer_schema_length=100,
            low_memory=True,
            ignore_errors=True,
        )
        # print(f'initial row count: {df.select(pl.len()).collect(streaming=True).item()}')

        print('filtering invalid dates')
        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date', drop_cols=['match2opponents'])

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
            .then(pl.col('team_1_template_name').str.to_lowercase())
            .when(is_null_or_empty('team_1_name') & ~is_null_or_empty('team_1_template'))
            .then(pl.col('team_1_template').str.to_lowercase())
            .otherwise(pl.col('team_1_name').str.to_lowercase())
            .alias('team_1'),
            pl.when(~is_null_or_empty('team_2_template_name'))
            .then(pl.col('team_2_template_name').str.to_lowercase())
            .when(is_null_or_empty('team_2_name') & ~is_null_or_empty('team_2_template'))
            .then(pl.col('team_2_template').str.to_lowercase())
            .otherwise(pl.col('team_2_name').str.to_lowercase())
            .alias('team_2'),
        )

        # filter out matches with placeholder teams or without any results recorded
        print('filtering tbd')
        placeholder_expr = (pl.col('team_1') == 'tbd') | (pl.col('team_2') == 'tbd')
        df = self.filter_invalid(df, placeholder_expr, 'placeholder')

        missing_team_expr = is_null_or_empty('team_1') | is_null_or_empty('team_2')
        df = self.filter_invalid(df, missing_team_expr, 'missing_team')

        played_self_expr = pl.col('team_1') == pl.col('team_2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        missing_results_expr = (
            (pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1) & (pl.col('winner') == '')
        ) | ((pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & (pl.col('winner') == ''))
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
                pl.col('pagename').alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
            .collect(streaming=True)
        )

        print(f'valid row count: {df.shape[0]}')
        df.write_csv(self.final_data_path)
