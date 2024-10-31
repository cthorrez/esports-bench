import json
import polars as pl
import requests
from esportsbench.data_pipeline.data_pipeline import DataPipeline


class LeaugeOfLegendsDataPipeline(DataPipeline):
    """class to ingest and process data from leaguepedia"""

    game = 'league_of_legends'
    base_url = 'https://lol.fandom.com/api.php?'
    request_params_groups = {
        'league_of_legends.jsonl': {
            'action': 'cargoquery',
            'format': 'json',
            'tables': 'MatchSchedule=MS, TeamRedirects=TRA, TeamRedirects=TRB',
            'fields': 'DateTime_UTC, Team1, Team2, TRA._pageName=Team1Redirect, TRB._pageName=Team2Redirect, Team1Final, Team2Final, Team1Advantage, Team2Advantage, IsNullified, Team1Score, Team2Score, Winner, MatchId, Player1, Player2, OverviewPage',
            'where': '(DateTime_UTC IS NOT NULL) AND (FF IS NULL) AND (Winner IS NOT NULL) AND (Team1 != "TBD") AND (Team2 != "TBD") AND (Player1 IS NULL) AND (Player2 IS NULL)',
            'join_on': 'MS.Team1=TRA.AllName, MS.Team2=TRB.AllName',
            'order_by': 'DateTime_UTC, MatchId',
        }
    }

    def __init__(self, rows_per_request=500, timeout=2.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)
        self.rows_per_request = rows_per_request

    def get_request_iterator(self, request_params):
        request_params['limit'] = self.rows_per_request

        def request_iterator():
            offset = 0
            while True:
                request_params['offset'] = offset
                offset += self.rows_per_request
                request = requests.Request(method='GET', url=self.base_url, params=request_params)
                prepared_request = request.prepare()
                yield prepared_request

        return iter(request_iterator())

    def process_response(self, response):
        response_text = response.text
        response_json = json.loads(response_text)
        results = []
        if 'cargoquery' not in response_json:
            print(response_json)
        for result in response_json['cargoquery']:
            results.append(result['title'])
        is_done = len(results) < self.rows_per_request
        return results, is_done

    def process_data(self):
        """function to process the raw data"""
        df = pl.scan_ndjson(self.raw_data_dir / 'league_of_legends.jsonl', infer_schema_length=1).collect()
        print(f'initial row count: {df.shape[0]}')

        # if the team name redirects, replace the original name with the redirect
        df = df.with_columns(
            pl.when(pl.col('Team1Redirect').is_not_null())
            .then(pl.col('Team1Redirect'))
            .otherwise(pl.col('Team1'))
            .alias('Team1'),
            pl.when(pl.col('Team2Redirect').is_not_null())
            .then(pl.col('Team2Redirect'))
            .otherwise(pl.col('Team2'))
            .alias('Team2'),
        )

        played_self_expr = pl.col('Team1') == pl.col('Team2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        df = df.with_columns(
            pl.when(pl.col('Winner') == '1')
            .then(1.0)
            .when(pl.col('Winner') == '2')
            .then(0.0)
            .when(pl.col('Winner') == '0')
            .then(0.5)
            .otherwise(None)
            .alias('outcome')
        )
        null_outcome_expr = pl.col('outcome').is_null()
        df = self.filter_invalid(df, null_outcome_expr, 'null_outcome')

        df = (
            df.select(
                pl.col('DateTime UTC').alias('date'),
                pl.col('Team1').alias('competitor_1'),
                pl.col('Team2').alias('competitor_2'),
                pl.col('Team1Score').cast(pl.Float64).alias('competitor_1_score'),
                pl.col('Team2Score').cast(pl.Float64).alias('competitor_2_score'),
                'outcome',
                pl.col('MatchId').alias('match_id'),
                (pl.lit('https://lol.fandom.com/wiki/') + pl.col('OverviewPage').str.replace_all('\s', '_')).alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )
        return df
