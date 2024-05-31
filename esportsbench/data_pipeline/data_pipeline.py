"""
Classes for data pipelines
Each data pipeline must implement ingest_data and process_data methods
"""

from abc import ABC
from typing import List, Dict
import time
import json
from pathlib import Path
import os
import polars as pl
from requests import Request, Response
from requests_cache import CachedSession
from dotenv import load_dotenv


def print_request(request):
    method = request.method
    url = request.url
    if method == 'GET':
        print(f'GET: {url}')
    elif method == 'POST':
        print(f'POST: {url}')
        print(f'body: {request.body}')


class Waiter:
    """class to handle waiting appropriate times between making requests"""

    def __init__(self, timeout):
        self.timeout = timeout
        self.prev_time = time.time()
        self.first_time = True  # we do not need to wait on the first call of a run

    def wait(self):
        if self.first_time:
            self.first_time = False
        else:
            time_since_prev = time.time() - self.prev_time
            wait_time = max(0, self.timeout - time_since_prev)
            if wait_time > 0:
                print(f'waiting {wait_time} seconds before issuing the request')
                time.sleep(wait_time)
            self.prev_time = time.time()


class DataPipeline(ABC):
    """base class for ingesting and processing data from various APIs"""

    game: str = None
    schema_overrides: Dict = None
    request_params_groups: Dict = None

    def __init__(
        self,
        rows_per_request,
        timeout,
        keys_to_refresh=None,
        max_rows=float('inf'),
        **kwargs,
    ):
        self.rows_per_request = rows_per_request
        self.waiter = Waiter(timeout=timeout)
        self.keys_to_refresh = []
        if keys_to_refresh is not None:
            self.keys_to_refresh = keys_to_refresh
        self.max_rows = max_rows

        data_dir = Path(__file__).resolve().parents[2] / 'data'
        self.raw_data_dir = data_dir / 'raw_data'
        self.invalid_data_dir = data_dir / 'invalid_data' / self.game
        self.full_data_dir = data_dir / 'full_data'
        self.full_data_path = self.full_data_dir / f'{self.game}.csv'
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.invalid_data_dir, exist_ok=True)
        os.makedirs(self.full_data_dir, exist_ok=True)

        cache_dir = data_dir / 'requests_cache'
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = cache_dir / self.game


    def process_response(self, response: Response) -> List[dict]:
        raise NotImplementedError

    def ingest_data(self):
        for filename, request_params in self.request_params_groups.items():
            request_iterator = self.get_request_iterator(request_params)
            self.ingest_data_for_group(filename, request_iterator)

    def ingest_data_for_group(self, filename, request_iterator):
        """main ingestion method. Makes requests from the iterate_requests methods, processes the
        results and writes the raw data to disk.
        """
        sess = CachedSession(self.cache_path, backend='sqlite', allowable_methods=('GET', 'POST'))
        output_path = self.raw_data_dir / filename
        with open(output_path, 'w', encoding='utf8') as out_file:
            num_rows = 0
            is_done = False
            req_idx = 0
            while not is_done:
                request = next(request_iterator)
                print('current request:')
                print_request(request)
                request_key = sess.cache.create_key(request)
                cache_has_key = sess.cache.contains(request_key)
                force_refresh = (not cache_has_key) or (request_key in self.keys_to_refresh)
                if force_refresh:
                    self.waiter.wait()
                response = sess.send(request, force_refresh=force_refresh)
                processed_rows, is_done = self.process_response(response)

                if cache_has_key and is_done:
                    print(f'response had less than {self.rows_per_request} rows, retrying with force_refresh=True')
                    self.waiter.wait()
                    response = sess.send(request, force_refresh=True)
                    processed_rows, is_done = self.process_response(response)

                num_rows += len(processed_rows)
                req_idx += 1

                for row in processed_rows:
                    row['request_key'] = request_key
                    if self.schema_overrides is not None:
                        for key, value in self.schema_overrides.items():
                            if key in row:
                                row[key] = value(row[key])
                    out_file.write(json.dumps(row) + '\n')
                out_file.flush()
                if num_rows >= self.max_rows:
                    is_done = True
                if num_rows == 0:
                    is_done = True
                    sess.cache.delete(keys=request_key)
        sess.close()
        print(f'wrote {num_rows} to {output_path}')

    def process_data(self):
        raise NotImplementedError

    def filter_invalid(self, df: pl.DataFrame, invalid_expr: pl.Expr, invalid_file_name: str, drop_cols: list = None):
        """given a dataframe and an expression:
        filter rows from a dataframe, write the filtered rows to disk, return the filtered df

        Usage:
        invalid_expr = pl.col('some_col') == 'some_invalid_value'
        df = self.filter_invalid(df, invalid_expr, 'some_col_has_invalid_value')
        """
        invalid_rows = df.filter(invalid_expr)
        if drop_cols is not None:
            invalid_rows = invalid_rows.drop(drop_cols)
        if isinstance(invalid_rows, pl.LazyFrame):
            invalid_rows = invalid_rows.collect(streaming=True)
        else:
            invalid_rows = invalid_rows.rechunk()  # pretty sure I only need this because of a polars bug...
        invalid_rows.write_ndjson(self.invalid_data_dir / f'{invalid_file_name}.jsonl')
        invalid_rows = invalid_rows.clear()  # probably does nothing
        del invalid_rows
        df = df.filter(~invalid_expr)
        return df


class LPDBDataPipeline(DataPipeline):
    """class for ingesting and processing data from LPDB"""

    version = None

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)
        load_dotenv()
        lpdb_api_key = os.getenv('LPDB_API_KEY')
        if lpdb_api_key is None:
            raise EnvironmentError('LPDB_API_KEY is not set')
        self.base_url = f'https://api.liquipedia.net/api/{self.version}/match'
        self.base_request_params = {'limit': rows_per_request}
        self.headers = {
            'authorization': f'Apikey {lpdb_api_key}',
            'accept': 'application/json',
        }
        if self.version == 'v1':
            self.base_request_params['apikey'] = lpdb_api_key
            self.headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'accept-encoding': 'gzip',
            }

    def get_request_iterator(self, request_params):
        def request_iterator():
            offset = 0
            is_done = False
            request_params.update(self.base_request_params)
            while not is_done:
                request_params['offset'] = offset
                offset += self.rows_per_request
                if self.version == 'v3':
                    request = Request(
                        method='GET',
                        url=self.base_url,
                        params=request_params,
                        headers=self.headers,
                    )
                elif self.version == 'v1':
                    request = Request(
                        method='POST',
                        url=self.base_url,
                        data=request_params,
                        headers=self.headers,
                    )
                else:
                    raise ValueError('version must be either v1 or v3')
                prepared_request = request.prepare()
                yield prepared_request

        return iter(request_iterator())

    def process_response(self, response):
        response_text = response.text
        if (response.status_code == 200) and (response_text == ''):
            print('🤔Something is REAL 🐠 🐟 🎣 🐟🐠 going on')
            print(response.headers)
            print(response.raw)
            print(response.raw.data)
            exit(1)
        else:
            response_json = json.loads(response_text)
            results = response_json['result']
            is_done = len(results) < self.rows_per_request
        return results, is_done
