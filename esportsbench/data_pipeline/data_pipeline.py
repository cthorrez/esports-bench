"""
Classes for data pipelines
Each data pipeline must implement ingest_data and process_data methods
"""

from abc import ABC
from typing import List, Dict, Tuple
import time
import json
import datetime
from pathlib import Path
import os
import polars as pl
import requests
from requests import Request, Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from esportsbench.utils import is_null_or_empty, outcome_from_scores, invalid_date_expr


def make_session():
    """requests session which retries transient failures (dropped connections, rate limiting, server errors)"""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=10, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None)
    session.mount('https://', HTTPAdapter(max_retries=retry))
    session.mount('http://', HTTPAdapter(max_retries=retry))
    return session


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
                time.sleep(wait_time)
        self.prev_time = time.time()


class DataPipeline(ABC):
    """base class for ingesting and processing data from various APIs

    raw data is stored as one jsonl file per request group, with one row per match keyed by key_fields.
    ingest_data refreshes it incrementally: it re-fetches only matches dated within overlap_days of the
    newest stored match and upserts them by key, so it costs a few requests per source and picks up
    late result corrections. ingest_data(full=True) rebuilds the file from scratch.
    """

    game: str = None
    schema_overrides: Dict = None
    request_params_groups: Dict = None
    # raw fields which together uniquely identify a match
    key_fields = ('match2id',)
    # raw field holding the match date
    date_field = 'date'
    # a refresh which returns fewer rows than this fraction of the stored matches in its window is treated as
    # a broken response (e.g. an expired api key returning an empty result) and nothing is written
    min_fetched_fraction = 0.5

    def __init__(
        self,
        rows_per_request,
        timeout,
        max_rows=float('inf'),
        **kwargs,
    ):
        self.rows_per_request = rows_per_request
        self.waiter = Waiter(timeout=timeout)
        self.max_rows = max_rows
        self.session = make_session()

        data_dir = Path(__file__).resolve().parents[2] / 'data'
        self.raw_data_dir = data_dir / 'raw_data'
        self.invalid_data_dir = data_dir / 'invalid_data' / self.game
        self.full_data_dir = data_dir / 'full_data'
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.invalid_data_dir, exist_ok=True)
        os.makedirs(self.full_data_dir / 'csv', exist_ok=True)
        os.makedirs(self.full_data_dir / 'parquet', exist_ok=True)

    @property
    def rate_limit_group(self):
        """pipelines in the same group share a rate limit so must not fetch at the same time"""
        return type(self).__name__

    def get_request_iterator(self, request_params):
        raise NotImplementedError

    def process_response(self, response: Response) -> Tuple[List[dict], bool]:
        """returns the rows from a response and whether it was the last page"""
        raise NotImplementedError

    def with_date_filter(self, request_params: dict, start_date: str) -> dict:
        """returns request_params restricted to matches after start_date (YYYY-MM-DD)"""
        raise NotImplementedError

    def row_key(self, row: dict) -> str:
        return json.dumps([row.get(field) for field in self.key_fields])

    def row_date(self, row: dict) -> str:
        return str(row.get(self.date_field) or '')

    def clean_row(self, row: dict) -> dict:
        if self.schema_overrides is not None:
            for key, value in self.schema_overrides.items():
                if key in row:
                    row[key] = value(row[key])
        # HACKY HACKY HACK
        if 'match2opponents' in row:
            for opp in row['match2opponents']:
                if ('extradata' in opp) and (opp['extradata'] == []):
                    opp['extradata'] = None
        return row

    def fetch_rows(self, request_params: dict, label: str) -> List[dict]:
        """fetch every page for request_params, waiting between requests to respect the rate limit"""
        rows = []
        for request in self.get_request_iterator(dict(request_params)):
            self.waiter.wait()
            response = self.session.send(request, timeout=300)
            page_rows, is_done = self.process_response(response)
            rows.extend(self.clean_row(row) for row in page_rows)
            print(f'{label}: fetched {len(rows)} rows', flush=True)
            if is_done or len(rows) >= self.max_rows:
                return rows
        return rows

    def read_raw_rows(self, path: Path) -> Dict[str, Tuple[str, str]]:
        """map of key -> (date, raw json line) for a stored raw file, the last copy of a key wins"""
        rows = {}
        with open(path, encoding='utf8') as in_file:
            for line in in_file:
                row = json.loads(line)
                if 'request_key' in row:  # left over from the old requests_cache ingestion
                    del row['request_key']
                    line = json.dumps(row) + '\n'
                if not line.endswith('\n'):
                    line += '\n'
                rows[self.row_key(row)] = (self.row_date(row), line)
        return rows

    def ingest_data(self, full=False, overlap_days=90):
        for filename, request_params in self.request_params_groups.items():
            self.ingest_data_for_group(filename, request_params, full=full, overlap_days=overlap_days)

    def ingest_data_for_group(self, filename, request_params, full=False, overlap_days=90):
        """refresh one raw data file, see the class docstring"""
        path = self.raw_data_dir / filename
        rows = {} if (full or not path.exists()) else self.read_raw_rows(path)
        start_date = None
        if rows:
            newest = min(max(date for date, _ in rows.values())[:10], datetime.date.today().isoformat())
            start_date = (datetime.date.fromisoformat(newest) - datetime.timedelta(days=overlap_days)).isoformat()
            request_params = self.with_date_filter(dict(request_params), start_date)
        label = f'{self.game} {filename} ' + (f'since {start_date}' if start_date else '(full)')

        fetched = {self.row_key(row): (self.row_date(row), json.dumps(row) + '\n') for row in self.fetch_rows(request_params, label)}

        removed = []
        if start_date is not None:
            # stored matches inside the window which the source no longer returns were deleted, moved
            # or stopped matching the conditions. Skip the boundary day since APIs differ on > vs >=
            cutoff = (datetime.date.fromisoformat(start_date) + datetime.timedelta(days=1)).isoformat()
            in_window = [key for key, (date, _) in rows.items() if date[:10] >= cutoff]
            if len(fetched) < self.min_fetched_fraction * len(in_window):
                raise RuntimeError(
                    f'{label}: only {len(fetched)} matches returned but {len(in_window)} are stored for the window, '
                    'not writing, check the API response'
                )
            removed = [key for key in in_window if key not in fetched]
            for key in removed:
                del rows[key]
        num_updated = sum(key in rows for key in fetched)
        rows.update(fetched)

        tmp_path = path.with_suffix('.jsonl.tmp')
        with open(tmp_path, 'w', encoding='utf8') as out_file:
            for key in sorted(rows, key=lambda key: (rows[key][0], key)):
                out_file.write(rows[key][1])
        os.replace(tmp_path, path)
        print(
            f'{label}: {len(fetched)} fetched, {len(fetched) - num_updated} new, {num_updated} updated, '
            f'{len(removed)} removed, {len(rows)} total in {path}', flush=True
        )

    def process_and_write(self):
        df = self.process_data()
        print(f'{self.game} final row count: {df.shape[0]}')
        df.write_csv(self.full_data_dir / 'csv' / f'{self.game}.csv')
        df.write_parquet(self.full_data_dir / 'parquet' / f'{self.game}.parquet')

    def process_data(self) -> pl.DataFrame:
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


def unpack_players(players):
    out_players = []
    if isinstance(players, list):
        for player in players:
            if isinstance(player, dict):
                out_players.append(player['player'])
    if isinstance(players, dict):
        for player in players.values():
            if isinstance(player, dict):
                out_players.append(player['player'])
    return out_players


class LPDBDataPipeline(DataPipeline):
    """class for ingesting and processing data from LPDB"""

    version = None
    # these names appear when a match did not occur, has not occured yet, or data was never entered
    invalid_competitor_names = {'bye', 'tba', 'tbd'}

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
        self.wiki = next(iter(self.request_params_groups.values()))['wiki']
        self.page_prefix = f'https://liquipedia.net/{self.wiki}/'

    @property
    def rate_limit_group(self):
        # the LPDB rate limit is per wiki
        return f'liquipedia/{self.wiki}'

    def with_date_filter(self, request_params, start_date):
        request_params['conditions'] = f"({request_params['conditions']}) AND [[date::>{start_date}]]"
        return request_params

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
        if response.status_code != 200:
            # error responses still contain 'result': [] which would otherwise look like the last page
            raise RuntimeError(f'LPDB request failed with status {response.status_code}: {response_text[:500]}')
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

    
    @staticmethod
    def unpack_team_match(raw_games):
        if raw_games == '[]':
            return None
        games = json.loads(raw_games)
        outputs = []
        for idx, game in enumerate(games):
            if game['mode'] == '2v2':
                continue
            if len(game['opponents']) != 2:
                continue
            opponents = game['opponents']

            opp_1_players = []
            if 'players' in opponents[0]:
                opp_1_players = unpack_players(opponents[0]['players'])
            opp_2_players = []
            if 'players' in opponents[1]:
                opp_2_players = unpack_players(opponents[1]['players'])

            if (len(opp_1_players) != 1) or (len(opp_2_players) != 1):
                continue

            player_1 = opp_1_players[0]
            player_2 = opp_2_players[0]
            try:
                player_1_score = float(opponents[0]['score'])
                player_2_score = float(opponents[1]['score'])
            except:
                continue
            outcome = outcome_from_scores(player_1_score, player_2_score)
            outputs.append(
                {
                    'player_1': player_1,
                    'player_2': player_2,
                    'player_1_score': float(player_1_score),
                    'player_2_score': float(player_2_score),
                    'outcome': outcome,
                    'game_idx': idx,
                    'bestof': 1,
                }
            )
        return outputs

    def unpack_team_matches(self, team_df):

        team_match_struct = pl.Struct([
            pl.Field("player_1", pl.Utf8),
            pl.Field("player_2", pl.Utf8),
            pl.Field("player_1_score", pl.Float64),
            pl.Field("player_2_score", pl.Float64),
            pl.Field("outcome", pl.Float64),
            pl.Field("game_idx", pl.Int64),
            pl.Field("bestof", pl.Int64)
        ])

        team_df = team_df.with_columns(
            pl.col('match2games').map_elements(
                function=self.unpack_team_match,
                skip_nulls=True,
                return_dtype=pl.List(team_match_struct)
            ).alias('games')
        )

        games_df = team_df.explode('games').unnest('games')
        bad_team_game_expr = (
            is_null_or_empty(pl.col('player_1'))
            | is_null_or_empty(pl.col('player_2'))
            | is_null_or_empty(pl.col('player_1_score'))
            | is_null_or_empty(pl.col('player_2_score'))
            | (pl.col('player_1_score') == -1)
            | (pl.col('player_2_score') == -1)
        )
        games_df = self.filter_invalid(games_df, bad_team_game_expr, 'bad_team_game')

        games_df = games_df.with_columns(
            pl.col('player_1_score').cast(pl.Float64).alias('player_1_score'),
            pl.col('player_2_score').cast(pl.Float64).alias('player_2_score'),
            pl.concat_str([pl.col('match2id'), pl.col('game_idx').cast(pl.Utf8)], separator='_').alias('match2id'),
        )

        return games_df
    
    @staticmethod
    def drop_opponent_extradata(opps):
        for opp in opps:
            if 'match2players' in opp:
                for player in opp['match2players']:
                    if 'extradata' in player:
                        del player['extradata']
        return opps


def score_outcome_expr():
    """1.0 / 0.0 / 0.5 from comparing team_1_score and team_2_score, null if either is missing"""
    return (
        pl.when(pl.col('team_1_score') > pl.col('team_2_score'))
        .then(1.0)
        .when(pl.col('team_1_score') < pl.col('team_2_score'))
        .then(0.0)
        .when(
            ~is_null_or_empty('team_1_score')
            & ~is_null_or_empty('team_2_score')
            & (pl.col('team_1_score') == pl.col('team_2_score'))
        )
        .then(0.5)
        .otherwise(None)
    )


class LPDBTeamDataPipeline(LPDBDataPipeline):
    """shared processing for LPDB team games, where each match2 row has exactly 2 opponents

    subclasses set request_params_groups plus the class attributes below,
    and can override the hooks (filter_raw, missing_results_expr, filter_teams, filter_outcomes)
    """

    version = 'v3'
    schema_overrides = {'match2opponents': LPDBDataPipeline.drop_opponent_extradata}
    # kwargs for pl.scan_ndjson on the raw data
    read_kwargs = {'infer_schema_length': 100_000}
    # 'template': prefer the team template name, 'name': prefer the name entered on the match
    name_priority = 'template'
    lowercase_names = False
    placeholder_names = LPDBDataPipeline.invalid_competitor_names
    require_two_teams = True
    filter_missing_teams = True
    filter_dq = False
    # if False, winner == '0' is always a draw, otherwise only when resulttype == 'draw'
    draw_requires_resulttype = True
    # when the winner field doesn't decide the match, fall back to comparing scores
    score_outcome_fallback = False
    # scores of best of 1 matches are rounds, if True replace them with the outcome
    bo1_scores_from_outcome = False

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    def filter_raw(self, df):
        """hook for game specific filters on the raw columns"""
        return df

    def missing_results_expr(self):
        """hook for an expression marking matches without results, None for no filter"""
        return None

    def filter_teams(self, df):
        """hook for game specific filters/transforms once team_1, team_2 and their scores exist"""
        return df

    def filter_outcomes(self, df):
        """hook for game specific filters once outcome exists"""
        return df

    def team_name_expr(self, team):
        name, template, template_name = (pl.col(f'team_{team}_{field}') for field in ('name', 'template', 'template_name'))
        if self.name_priority == 'template':
            expr = (
                pl.when(~is_null_or_empty(template_name))
                .then(template_name)
                .when(is_null_or_empty(name) & ~is_null_or_empty(template))
                .then(template)
                .otherwise(name)
            )
        else:
            expr = (
                pl.when(~is_null_or_empty(name))
                .then(name)
                .when(~is_null_or_empty(template_name))
                .then(template_name)
                .when(~is_null_or_empty(template))
                .then(template)
                .otherwise(None)
            )
        if self.lowercase_names:
            expr = expr.str.to_lowercase()
        return expr.alias(f'team_{team}')

    def outcome_expr(self):
        is_draw = pl.col('winner') == '0'
        if self.draw_requires_resulttype:
            is_draw = is_draw & (pl.col('resulttype') == 'draw')
        expr = pl.when(pl.col('winner') == '1').then(1.0).when(pl.col('winner') == '2').then(0.0).when(is_draw).then(0.5)
        if self.score_outcome_fallback:
            expr = expr.when(~is_null_or_empty('score_outcome')).then(pl.col('score_outcome'))
        return expr.otherwise(None).alias('outcome')

    def score_exprs(self):
        if self.bo1_scores_from_outcome:
            # score is in rounds for best of 1, use the game score instead
            is_bo1 = pl.col('bestof') == 1
            return (
                pl.when(is_bo1).then(pl.col('outcome')).otherwise(pl.col('team_1_score')).alias('competitor_1_score'),
                pl.when(is_bo1).then(1.0 - pl.col('outcome')).otherwise(pl.col('team_2_score')).alias('competitor_2_score'),
            )
        return pl.col('team_1_score').alias('competitor_1_score'), pl.col('team_2_score').alias('competitor_2_score')

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / f'{self.game}.jsonl', **self.read_kwargs).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')
        if self.require_two_teams:
            df = self.filter_invalid(df, pl.col('match2opponents').list.len() != 2, 'not_two_teams')
        df = self.filter_raw(df)

        # extract team names, scores and statuses
        df = df.with_columns(
            pl.col('match2opponents').list.get(0).alias('team_1_struct'),
            pl.col('match2opponents').list.get(1).alias('team_2_struct'),
        ).drop('match2opponents')
        df = df.with_columns(
            expr
            for team in (1, 2)
            for expr in (
                pl.col(f'team_{team}_struct').struct.field('name').alias(f'team_{team}_name'),
                pl.col(f'team_{team}_struct').struct.field('template').alias(f'team_{team}_template'),
                pl.col(f'team_{team}_struct').struct.field('teamtemplate').struct.field('name').alias(f'team_{team}_template_name'),
                pl.col(f'team_{team}_struct').struct.field('score').cast(pl.Float64).alias(f'team_{team}_score'),
                pl.col(f'team_{team}_struct').struct.field('status').alias(f'team_{team}_status'),
            )
        ).drop('team_1_struct', 'team_2_struct')
        df = df.with_columns(self.team_name_expr(1), self.team_name_expr(2))

        placeholder_expr = (
            pl.col('team_1').str.to_lowercase().is_in(self.placeholder_names)
            | pl.col('team_2').str.to_lowercase().is_in(self.placeholder_names)
        )
        df = self.filter_invalid(df, placeholder_expr, 'invalid_competitor')
        if self.filter_missing_teams:
            df = self.filter_invalid(df, is_null_or_empty('team_1') | is_null_or_empty('team_2'), 'missing_team')
        if self.filter_dq:
            dq_expr = (pl.col('team_1_status').str.to_lowercase() == 'dq') | (pl.col('team_2_status').str.to_lowercase() == 'dq')
            df = self.filter_invalid(df, dq_expr, 'dq')
        missing_results_expr = self.missing_results_expr()
        if missing_results_expr is not None:
            df = self.filter_invalid(df, missing_results_expr, 'missing_results')
        df = self.filter_teams(df)

        df = df.with_columns(score_outcome_expr().alias('score_outcome'))
        df = df.with_columns(self.outcome_expr())
        df = self.filter_invalid(df, pl.col('outcome').is_null(), 'null_outcome')
        df = self.filter_outcomes(df)
        df = self.filter_invalid(df, pl.col('team_1') == pl.col('team_2'), 'played_self')

        df = df.select(
            'date',
            pl.col('team_1').alias('competitor_1'),
            pl.col('team_2').alias('competitor_2'),
            *self.score_exprs(),
            'outcome',
            pl.col('match2id').alias('match_id'),
            (pl.lit(self.page_prefix) + pl.col('pagename')).alias('page'),
        ).unique()
        # sort on every column so rows sharing a (date, match_id) come out in the same order every run
        return df.sort('date', 'match_id', *[col for col in df.columns if col not in ('date', 'match_id')], nulls_last=True)
