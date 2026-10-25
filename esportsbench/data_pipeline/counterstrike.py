import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBTeamDataPipeline
from esportsbench.utils import is_null_or_empty


class CounterStrikeDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing counterstrike data from LPDB"""

    game = 'counterstrike'
    request_params_groups = {
        'counterstrike.jsonl': {
            'wiki': 'counterstrike',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof',
            'conditions': '[[mode::team]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]] AND [[section::!Showmatch]] AND [[liquipediatiertype::!Showmatch]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    read_kwargs = {'infer_schema_length': 100_000, 'low_memory': True, 'ignore_errors': True}
    lowercase_names = True
    require_two_teams = False
    bo1_scores_from_outcome = True

    def missing_results_expr(self):
        return (
            ((pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1) & (pl.col('winner') == '-1'))
            | ((pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1) & (pl.col('winner') == ''))
            | ((pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & (pl.col('winner') == ''))
            | is_null_or_empty('team_1_score')
            | is_null_or_empty('team_2_score')
        )
