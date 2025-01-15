import json
import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import is_null_or_empty, invalid_date_expr, outcome_from_scores


class EAFCDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing EA Sports FC data from LPDB"""

    game = 'ea_sports_fc'
    version = 'v3'
    schema_overrides = {'match2games': json.dumps}
    request_params_groups = {
        'ea_sports_fc.jsonl': {
            'wiki': 'easportsfc',
            'query': 'date, match2opponents, winner, resulttype, finished, bestof, match2id, mode, match2games, extradata',
            'conditions': '[[walkover::!1]] AND [[walkover::!2]] AND [[walkover::!ff]] AND [[finished::1]]',
            'order': 'date ASC, match2id ASC',
        }
    }
    placeholder_player_names = {'bye', 'tbd'}

    def __init__(self, rows_per_request=1000, timeout=60.0, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)


    @staticmethod
    def parse_team(raw_team):
        players = {}
        for player in raw_team['match2players']:
            players[player['id']] = player['name']
        return players

    @staticmethod
    def parse_opponents(raw_opponents):
        team_1 = EAFCDataPipeline.parse_team(raw_opponents[0])
        team_2 = EAFCDataPipeline.parse_team(raw_opponents[1])
        return [team_1, team_2]


    @staticmethod
    def unpack_team_match(raw_data):
        raw_games = raw_data['match2games']
        raw_opponents = raw_data['match2opponents']
        opponents = EAFCDataPipeline.parse_opponents(raw_opponents)
        if opponents[0] == {} or opponents[1] == {}:
            return None
        games = json.loads(raw_games)
        outputs = []
        for idx, game in enumerate(games):
            if game['participants'] == []:
                continue
            if len(game['participants']) != 2:
                continue
            if len(game['scores']) != 2:
                continue
            participant_keys = sorted(list(game['participants'].keys()))
            participant_ids = list(map(lambda x: int(x[2]), participant_keys))
            participant_names = []
            for pk in participant_keys:
                parts = game['participants']
                part = parts[pk]
                if isinstance(part, dict):
                    participant_names.append(part.get('name'))

            if 'tbd' in participant_names:
                continue
            if len(participant_ids) != 2:
                continue
            if len(participant_names) != 2:
                continue
            player_1_score, player_2_score = map(int, game['scores'])
            outcome = outcome_from_scores(player_1_score, player_2_score)
            player_1 = participant_names[0]
            if player_1 is None:
                if isinstance(opponents[0], dict):
                    player_1 = opponents[0].get(participant_ids[0])
                else:
                    print(opponents[0], participant_ids[0])
            player_2 = participant_names[1]
            if player_2 is None:
                if isinstance(opponents[1], dict):
                    player_2 = opponents[1].get(participant_ids[1])
                else:
                    print(opponents[1], participant_ids[1])
            if (player_1 is None) or (player_2 is None):
                continue
            outputs.append(
                {
                    'player_1': player_1,
                    'player_2': player_2,
                    'player_1_score': float(player_1_score),
                    'player_2_score': float(player_2_score),
                    'outcome': outcome,
                    'game_idx': idx,
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
        ])

        # first combine match2games and match2games into a single col
        team_df = team_df.with_columns(
             pl.struct(['match2opponents', 'match2games']).alias("match2data")
        )

        team_df = team_df.with_columns(
            pl.col('match2data').map_elements(
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
            self.raw_data_dir / 'ea_sports_fc.jsonl', infer_schema_length=100, ignore_errors=True
        ).collect()

        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        # filter out matches without exactly 2 teams
        not_two_players_expr = pl.col('match2opponents').list.len() != 2
        df = self.filter_invalid(df, not_two_players_expr, 'not_two_players')

        team_expr = (
            (pl.col('match2opponents').list.get(0).struct.field('type') == 'team') 
            | (pl.col('match2opponents').list.get(0).struct.field('type') == 'team')
        )
        team_df = df.filter(team_expr)
        print(f'num initial team matches: {len(team_df)}')

        df = df.filter(~team_expr)
        print(f'num initial 1v1 matches: {len(df)}')


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

        placeholder_expr = pl.col('player_1').str.to_lowercase().is_in(self.placeholder_player_names) | pl.col(
            'player_2'
        ).str.to_lowercase().is_in(self.placeholder_player_names)
        df = self.filter_invalid(df, placeholder_expr, 'placeholder')

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

        # team_games = self.unpack_team_matches(team_df)
        # print(f'1v1 matches from team matches: {len(team_games)}')
        # df = pl.concat([df, team_games], how='diagonal')
        # print(f'matches after merging: {len(df)}')

        # team_df = self.filter_invalid(team_matches, invalid_date_expr, 'invalid_date_team')
        # print(f'initial team match row count: {team_matches.shape[0]}')

        team_matches = self.unpack_team_matches(team_df)
        print(f'1v1 matches from team matches: {len(team_matches)}')

        df = pl.concat([df, team_matches], how='diagonal')
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
        return df
