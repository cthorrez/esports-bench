import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBTeamDataPipeline
from esportsbench.utils import is_null_or_empty


class CallOfDutyDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing  cod data from LPDB"""

    game = 'call_of_duty'
    request_params_groups = {
        'call_of_duty.jsonl': {
            'wiki': 'callofduty',
            'query': 'date, match2opponents, winner, game, mode, resulttype, walkover, match2id, pagename',
            'conditions': '[[mode::team]] AND [[game::!mobile]] AND [[game::!codm]] AND [[game::!Call of Duty: Mobile]] AND [[finished::1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[section::!Showmatch]] AND [[liquipediatiertype::!Showmatch]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    filter_missing_teams = False
    draw_requires_resulttype = False
    score_outcome_fallback = True

    def filter_raw(self, df):
        return self.filter_invalid(df, is_null_or_empty(pl.col('game')), 'missing_game')

    def filter_teams(self, df):
        two_v_two_expr = (
            pl.col('pagename').str.to_lowercase().str.contains('2v2')
            | pl.col('team_1').str.contains(r'\s&\s')
            | pl.col('team_2').str.contains(r'\s&\s')
        )
        return self.filter_invalid(df, two_v_two_expr, 'two_v_two')
