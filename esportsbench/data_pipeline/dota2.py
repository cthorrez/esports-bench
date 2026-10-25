"""Module for ingesting dota2 data from liquipedia"""
import json
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline, LPDBTeamDataPipeline
from esportsbench.utils import is_null_or_empty


class Dota2DataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing dota2 data from LPDB"""

    game = 'dota2'
    schema_overrides = {
        'match2games': json.dumps,
        'match2opponents': LPDBDataPipeline.drop_opponent_extradata
    }
    request_params_groups = {
        'dota2.jsonl': {
            'wiki': 'dota2',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof, match2id',
            'conditions': '[[game::dota2]] AND [[mode::team]] AND [[finished::1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[resulttype::!default]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    read_kwargs = {'infer_schema_length': 200_000}
    name_priority = 'name'
    lowercase_names = True
    filter_missing_teams = False
    draw_requires_resulttype = False
    score_outcome_fallback = True

    def filter_raw(self, df):
        return self.filter_invalid(df, pl.col('date') < '2011-01-01', 'too_early')

    def missing_results_expr(self):
        # did not play
        return (pl.col('team_1_score') == 0) & (pl.col('team_2_score') == 0) & is_null_or_empty('winner')
