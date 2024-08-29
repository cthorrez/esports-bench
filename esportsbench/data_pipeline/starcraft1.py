import json
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr, outcome_from_scores


class Starcraft1DataPipeline(LPDBDataPipeline):
    """class for ingesting and processing starcraft 1 data from LPDB"""

    game = 'starcraft1'
    version = 'v3'
    schema_overrides = {'match2games': json.dumps}
    request_params_groups = {
        'starcraft1_1v1.jsonl': {
            'wiki': 'starcraft',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof, match2id',
            'conditions': '([[mode::1_1]] OR [[mode::solo]] or [[mode::1v1]]) AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]]',
            'order': 'date ASC, match2id ASC',
        },
        'starcraft1_team.jsonl': {
            'wiki': 'starcraft',
            'query': 'date, match2games, mode, resulttype, match2id',
            'conditions': '([[mode::team_team]] OR [[mode::team]] OR [[mode::mixed]]) AND [[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]] AND [[resulttype::!default]]',
            'order': 'date ASC, match2id ASC',
        },
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)

    @staticmethod
    def unpack_team_match(raw_games):
        if raw_games == '[]':
            return None
        games = json.loads(raw_games)
        outputs = []
        for idx, game in enumerate(games):
            if game['mode'] == '2v2':
                continue
            if game['participants'] == []:
                continue
            if len(game['participants']) != 2:
                continue
            if len(game['scores']) != 2:
                continue
            players = [participant['player'] for participant in game['participants'].values()]
            player_1, player_2 = players
            player_1_score, player_2_score = map(float, game['scores'])
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

    def process_data(self):
        df = pl.scan_ndjson(
            self.raw_data_dir / 'starcraft1_1v1.jsonl', infer_schema_length=100, ignore_errors=True
        ).collect()

        print(f'initial 1v1 row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        # filter out matches without exactly 2 teams
        not_two_players_expr = pl.col('match2opponents').list.len() != 2
        df = self.filter_invalid(df, not_two_players_expr, 'not_two_players')

        # extract player names and scores
        df = df.with_columns(
            pl.col('match2opponents').list.get(0).alias('player_1_struct'),
            pl.col('match2opponents').list.get(1).alias('player_2_struct'),
        )
        df = df.with_columns(
            pl.col('player_1_struct').struct.field('name').alias('player_1_name'),
            pl.col('player_1_struct').struct.field('template').alias('player_1_template'),
            pl.col('player_1_struct').struct.field('score').cast(pl.Float64).alias('player_1_score'),
            pl.col('player_1_struct').struct.field('status').alias('player_1_status'),
            pl.col('player_2_struct').struct.field('name').alias('player_2_name'),
            pl.col('player_2_struct').struct.field('template').alias('player_2_template'),
            pl.col('player_2_struct').struct.field('score').cast(pl.Float64).alias('player_2_score'),
            pl.col('player_2_struct').struct.field('status').alias('player_2_status'),
        )

        # use name if it is not null, use template otherwise
        df = df.with_columns(
            pl.when(~is_null_or_empty('player_1_name'))
            .then(pl.col('player_1_name'))
            .when(~is_null_or_empty('player_1_template'))
            .then(pl.col('player_1_template'))
            .otherwise(None)
            .alias('player_1'),
            pl.when(~is_null_or_empty('player_2_name'))
            .then(pl.col('player_2_name'))
            .when(~is_null_or_empty('player_2_template'))
            .then(pl.col('player_2_template'))
            .otherwise(None)
            .alias('player_2'),
        )

        invalid_competitor_expr = (
            pl.col('player_1').str.to_lowercase().is_in(self.invalid_competitor_names) 
            | pl.col('player_2').str.to_lowercase().is_in(self.invalid_competitor_names)
        )
        df = self.filter_invalid(df, invalid_competitor_expr, 'invalid_competitor')

        unknown_expr = (pl.col('player_1').str.to_lowercase() == 'unknown') | (pl.col('player_2').str.to_lowercase() == 'unknown')
        df = self.filter_invalid(df, unknown_expr, 'unknown_player')

        missing_player_expr = is_null_or_empty('player_1') | is_null_or_empty('player_2')
        df = self.filter_invalid(df, missing_player_expr, 'missing_team')

        dq_expr = (pl.col('player_1_status').str.to_lowercase() == 'dq') | (
            pl.col('player_2_status').str.to_lowercase() == 'dq'
        )
        df = self.filter_invalid(df, dq_expr, 'dq')

        missing_results_expr = (
            (pl.col('winner') == '0') & (pl.col('player_1_score') == -1) & (pl.col('player_2_score') == -1)
        )
        df = self.filter_invalid(df, missing_results_expr, 'missing_results')

        df = df.with_columns(
            pl.when(pl.col('winner') == '1')
            .then(1.0)
            .when(pl.col('winner') == '2')
            .then(0.0)
            .when((pl.col('winner') == '0') & (pl.col('resulttype') == 'draw'))
            .then(0.5)
            .otherwise(None)
            .alias('outcome')
        )
        null_outcome_expr = pl.col('outcome').is_null()
        df = self.filter_invalid(df, null_outcome_expr, 'null_outcome')

        team_matches = pl.scan_ndjson(self.raw_data_dir / 'starcraft1_team.jsonl', infer_schema_length=100).collect()
        team_matches = self.filter_invalid(team_matches, invalid_date_expr, 'invalid_date_team')
        print(f'initial team match row count: {team_matches.shape[0]}')

        team_games = self.unpack_team_matches(team_matches)
        print(f'1v1 matches from team matches: {len(team_games)}')

        df = pl.concat([df, team_games], how='diagonal')
        print(f'matches after merging: {len(df)}')

        played_self_expr = pl.col('player_1') == pl.col('player_2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        # select final columns and write to csv
        df = (
            df.select(
                'date',
                pl.col('player_1').alias('competitor_1'),
                pl.col('player_2').alias('competitor_2'),
                pl.col('player_1_score').alias('competitor_1_score'),
                pl.col('player_2_score').alias('competitor_2_score'),
                'outcome',
                pl.col('match2id').alias('match_id'),
                (pl.lit(self.page_prefix) + pl.col('pagename')).alias('page'),
            )
            .unique()
            .unique()
            .sort('date', 'match_id')
        )

        print(f'final row count: {df.shape[0]}')
        df.write_csv(self.full_data_path)
