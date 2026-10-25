from esportsbench.data_pipeline.data_pipeline import LPDBTeamDataPipeline


class OverwatchDataPipeline(LPDBTeamDataPipeline):
    """class for ingesting and processing overwatch data from LPDB"""

    game = 'overwatch'
    request_params_groups = {
        'overwatch.jsonl': {
            'wiki': 'overwatch',
            'query': 'date, match2opponents, winner, game, mode, resulttype, walkover, match2id, pagename',
            'conditions': '[[liquipediatier::!-1]] AND [[finished::1]] AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    filter_missing_teams = False
    draw_requires_resulttype = False
    score_outcome_fallback = True
