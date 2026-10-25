"""classes and functions for ingesting and processing rocket league data"""
import json
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBTeamDataPipeline


class RocketLeagueDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing rocket league data from LPDB"""

    game = 'rocket_league'
    schema_overrides = {'extradata': json.dumps}
    request_params_groups = {
        'rocket_league.jsonl': {
            'wiki': 'rocketleague',
            'query': 'date, match2opponents, winner, resulttype, extradata, finished, bestof',
            'conditions': '[[mode::3v3]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    read_kwargs = {'infer_schema_length': 120_000, 'ignore_errors': True}
    name_priority = 'name'
    placeholder_names = {'bye', 'tbd'}
    filter_dq = True

    def missing_results_expr(self):
        return (pl.col('winner') == '0') & (pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1)
