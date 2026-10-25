"""classes and functions for ingesting and processing rainbow six data"""
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBTeamDataPipeline


class RainbowSixDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing rainbow six data from LPDB"""

    game = 'rainbow_six'
    request_params_groups = {
        'rainbow_six.jsonl': {
            'wiki': 'rainbowsix',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof',
            'conditions': '[[mode::team]] AND ([[game::siege]] OR [[game::siegex]]) AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    read_kwargs = {'infer_schema_length': 89_000}
    name_priority = 'name'
    placeholder_names = {'bye', 'tbd'}
    filter_dq = True

    def missing_results_expr(self):
        return (pl.col('winner') == '0') & (pl.col('team_1_score') == -1) & (pl.col('team_2_score') == -1)
