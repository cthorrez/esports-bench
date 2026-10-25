import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBTeamDataPipeline


class HaloDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing halo data from LPDB"""

    game = 'halo'
    schema_overrides = None
    request_params_groups = {
        'halo.jsonl': {
            'wiki': 'halo',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof',
            'conditions': '[[mode::team]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[walkover::!dq]] AND [[resulttype::!default]] AND [[finished::1]] AND [[section::!Showmatch]] AND [[liquipediatiertype::!Showmatch]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    read_kwargs = {'infer_schema_length': 50_000}
    lowercase_names = True
    score_outcome_fallback = True

    def filter_raw(self, df):
        return self.filter_invalid(df, pl.col('pagename').str.to_lowercase().str.contains('2v2'), 'two_v_two')

    def missing_results_expr(self):
        no_winner = pl.col('winner') == ''
        return (
            ((pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1) & no_winner)
            | ((pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & no_winner)
        )

    def filter_teams(self, df):
        # map W/L results to a score of 1 to 0
        status_1, status_2 = pl.col('team_1_status').str.to_lowercase(), pl.col('team_2_status').str.to_lowercase()
        team_1_won = (status_1 == 'w') & (status_2 == 'l')
        team_2_won = (status_1 == 'l') & (status_2 == 'w')
        return df.with_columns(
            pl.when(team_1_won).then(1.0).when(team_2_won).then(0.0).otherwise(pl.col('team_1_score')).alias('team_1_score'),
            pl.when(team_1_won).then(0.0).when(team_2_won).then(1.0).otherwise(pl.col('team_2_score')).alias('team_2_score'),
        )

    def filter_outcomes(self, df):
        return self.filter_invalid(df, pl.col('outcome') != pl.col('score_outcome'), 'outcome_disagree')
