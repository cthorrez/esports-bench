import os
import json
import requests
from dotenv import load_dotenv
import polars as pl
from esportsbench.data_pipeline.data_pipeline import DataPipeline


class Starcraft2DataPipeline(DataPipeline):
    """class for ingesting and processing starcraft 2 data from aligulac"""
    load_dotenv()
    aligulac_api_key = os.getenv('ALIGULAC_API_KEY')
    if aligulac_api_key is None:
        raise EnvironmentError('ALIGULAC_API_KEY is not set')
    game = 'starcraft2'
    base_url = 'http://aligulac.com/api/v1/match/'
    request_params_groups = {
        'starcraft2.jsonl': {
            'format': 'json',
            'order_by': 'date',  # should this be id?
            'apikey': aligulac_api_key,
        }
    }

    def __init__(self, rows_per_request=500, timeout=10.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)
        self.rows_per_request = rows_per_request

    def get_request_iterator(self, request_params):
        def request_iterator():
            offset = 0
            is_done = False
            request_params['limit'] = self.rows_per_request
            while not is_done:
                request_params['offset'] = offset
                offset += self.rows_per_request
                request = requests.Request(method='GET', url=self.base_url, params=request_params)
                if offset >= self.max_rows:
                    is_done = True
                prepared_request = request.prepare()
                yield prepared_request

        return iter(request_iterator())

    def process_response(self, response):
        response_text = response.text
        response_json = json.loads(response_text)
        objects = response_json['objects']
        is_done = len(objects) < self.rows_per_request
        return objects, is_done

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / 'starcraft2.jsonl', infer_schema_length=1).collect()
        print(f'initial row count: {df.shape[0]}')

        df = df.with_columns(
            pl.concat_str(
                pl.col('pla').struct.field('tag'),
                pl.col('pla').struct.field('id'),
                separator='_',
            ).alias('player_1'),
            pl.concat_str(
                pl.col('plb').struct.field('tag'),
                pl.col('plb').struct.field('id'),
                separator='_',
            ).alias('player_2'),
            pl.col('sca').cast(pl.Float64).alias('player_1_score'),
            pl.col('scb').cast(pl.Float64).alias('player_2_score'),
            pl.col('eventobj').struct.field('fullname').str.replace_all('\s', '-').str.replace_all('\/', '').alias('event_name'),
            pl.col('eventobj').struct.field('id').cast(pl.Utf8).alias('event_id'),

        )
        df = df.with_columns(
            (pl.lit('http://aligulac.com/results/events/') + pl.col('event_id') + pl.lit('-') + pl.col('event_name')).alias('page')
        )
        df = df.with_columns(
            pl.when(pl.col('player_1_score') > pl.col('player_2_score'))
            .then(1.0)
            .when(pl.col('player_1_score') < pl.col('player_2_score'))
            .then(0.0)
            .when(pl.col('player_1_score') == pl.col('player_2_score'))
            .then(0.5)
            .otherwise(None)
            .alias('outcome')
        )
        null_outcome_expr = pl.col('outcome').is_null()
        df = self.filter_invalid(df, null_outcome_expr, 'null_outcome')

        played_self_expr = pl.col('player_1') == pl.col('player_2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        df = (
            df.select(
                'date',
                pl.col('player_1').alias('competitor_1'),
                pl.col('player_2').alias('competitor_2'),
                pl.col('player_1_score').alias('competitor_1_score'),
                pl.col('player_2_score').alias('competitor_2_score'),
                'outcome',
                pl.col('id').alias('match_id'),
                'page',
            )
            .unique()
            .sort('date', 'match_id')
        )
        return df
