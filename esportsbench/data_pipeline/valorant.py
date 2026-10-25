import json
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline, LPDBTeamDataPipeline
from esportsbench.utils import is_null_or_empty


class ValorantDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing valorant data from LPDB"""

    game = 'valorant'
    schema_overrides = {
        'extradata': json.dumps,
        'match2opponents': LPDBDataPipeline.drop_opponent_extradata
    }
    request_params_groups = {
        'valorant.jsonl': {
            'wiki': 'valorant',
            'query': 'date, match2opponents, winner, resulttype, extradata, finished, bestof',
            'conditions': '[[mode::team]] AND [[winner::!-1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]] AND [[section::!Showmatch]] AND [[liquipediatiertype::!Showmatch]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    read_kwargs = {'infer_schema_length': 100_000, 'ignore_errors': True}
    bo1_scores_from_outcome = True

    def missing_results_expr(self):
        no_winner = is_null_or_empty('winner') | (pl.col('winner') == '0')
        return (
            ((pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1) & no_winner)
            | ((pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & no_winner)
        )
