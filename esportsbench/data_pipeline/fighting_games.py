import polars as pl
from esportsbench.data_pipeline.data_pipeline import LPDBDataPipeline
from esportsbench.utils import invalid_date_expr

# info from https://liquipedia.net/fighters/Module:Info alphabetical order
GAME_CONFIG = {
    'street_fighter': {
        'hsfii',
        'SFxT',
        'sf6',
        'sfa2',
        'sfa3',
        'sfii',
        'sfiii',
        'sfiiis',
        'sfiii3s',
        'sfiit',
        'sfiv',
        'sfv',
        'sfvce',
        'sfxt',
        'ssfiit',
        'ssfiithdr',
        'ssfiv',
        'ssfivae',
        'usfiv',
        'sfvae'
    },
    'tekken': {
        't4',
        't5',
        't5dr',
        't6',
        't7',
        't8',
        'ttt',
        'ttt2',
        'tvc',
        'SFxT'
    },
    'guilty_gear': {
        'gg',
        'ggx2r',
        'GGXrd',
        'ggst',
        'ggxrd',
        'ggxrdr',
        'ggxrdr2',
        'ggxx',
        'ggxxs',
        'ggxxacp',
        'ggxxacpr'
    },
    'king_of_fighters': {
        'KoFXV',
        'kof2002',
        'kof2002um',
        'kof2003',
        'kof98',
        'kof98um',
        'kofneowave',
        'kofxiii',
        'kofxiv',
        'kofxv',
    },
}


class FightingGamesDataPipeline(LPDBDataPipeline):
    """class for ingesting and processing raw fighting game data from LPDB"""

    game = 'fighting_games'
    version = 'v1'
    request_params_groups = {
        f'{game}.jsonl': {
            'wiki': 'fighters',
            'query': 'date, opponent1, opponent2, opponent1score, opponent2score, winner, game, matchid, pagename, objectname, extradata',
            'conditions': '[[walkover::!1]] AND [[walkover::!2]] AND [[mode::singles]] AND [[opponent1::!Bye]] AND [[opponent2::!Bye]]',
            'order': 'date ASC, objectname ASC',
        }
    }

    def __init__(self, rows_per_request=1000, timeout=60.0, game=None, **kwargs):
        super().__init__(rows_per_request=rows_per_request, timeout=timeout, **kwargs)
        self.game = game

    def process_data(self):
        df = pl.scan_ndjson(self.raw_data_dir / 'fighting_games.jsonl', infer_schema_length=200000).collect()
        print(f'initial row count: {df.shape[0]}')

        df = self.filter_invalid(df, invalid_date_expr, 'invalid_date')

        df = df.with_columns(
            pl.when(pl.col('winner') == pl.col('opponent1'))
            .then(1.0)
            .when(pl.col('winner') == pl.col('opponent2'))
            .then(0.0)
            .otherwise(None)
            .alias('outcome')
        )
        null_outcome_expr = pl.col('outcome').is_null()
        df = self.filter_invalid(df, null_outcome_expr, 'null_outcome')

        played_self_expr = pl.col('opponent1') == pl.col('opponent2')
        df = self.filter_invalid(df, played_self_expr, 'played_self')

        df = (
            df.select(
                'date',
                pl.col('opponent1').alias('competitor_1'),
                pl.col('opponent2').alias('competitor_2'),
                pl.col('opponent1score').cast(pl.Float64).alias('competitor_1_score'),
                pl.col('opponent2score').cast(pl.Float64).alias('competitor_2_score'),
                'outcome',
                'game',
                pl.col('objectname').str.replace('\n', '').alias('match_id'),
                (pl.lit(self.page_prefix) + pl.col('pagename')).alias('page'),
            )
            .unique()
            .sort('date', 'match_id')
        )
        return df.filter(pl.col('game').is_in(GAME_CONFIG[self.game]))
