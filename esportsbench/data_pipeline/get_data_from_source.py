"""main script for ingesting data from various sources"""
import argparse
from functools import partial
from multiprocessing import Pool
from operator import methodcaller
from esportsbench.data_pipeline.starcraft2 import Starcraft2DataPipeline
from esportsbench.data_pipeline.starcraft1 import Starcraft1DataPipeline
from esportsbench.data_pipeline.rocket_league import RocketLeagueDataPipeline
from esportsbench.data_pipeline.counterstrike import CounterStrikeDataPipeline
from esportsbench.data_pipeline.smash_melee import SmashMeleeDataPipeline
from esportsbench.data_pipeline.smash_ultimate import SmashUltimateDataPipeline
from esportsbench.data_pipeline.league_of_legends import LeaugeOfLegendsDataPipeline
from esportsbench.data_pipeline.dota2 import Dota2DataPipeline
from esportsbench.data_pipeline.valorant import ValorantDataPipeline
from esportsbench.data_pipeline.overwatch import OverwatchDataPipeline
from esportsbench.data_pipeline.warcraft3 import Warcraft3DataPipeline
from esportsbench.data_pipeline.rainbox_six import RainbowSixDataPipeline
from esportsbench.data_pipeline.halo import HaloDataPipeline
from esportsbench.data_pipeline.call_of_duty import CallOfDutyDataPipeline
from esportsbench.data_pipeline.tetris import TetrisDataPipeline
from esportsbench.data_pipeline.fighting_games import FightingGamesDataPipeline
from esportsbench.data_pipeline.fifa import FIFADataPipeline
from esportsbench.data_pipeline.postprocess import postprocess
from esportsbench.utils import delimited_list
from esportsbench.arg_parsers import get_games_argparser

GAME_CLASS_MAP = {
    'sc1': Starcraft1DataPipeline,
    'sc2': Starcraft2DataPipeline,
    'rl': RocketLeagueDataPipeline,
    'cs': CounterStrikeDataPipeline,
    'ssbm': SmashMeleeDataPipeline,
    'ssbu': SmashUltimateDataPipeline,
    'lol': LeaugeOfLegendsDataPipeline,
    'dota2': Dota2DataPipeline,
    'val': ValorantDataPipeline,
    'ow': OverwatchDataPipeline,
    'wc3': Warcraft3DataPipeline,
    'r6': RainbowSixDataPipeline,
    'halo': HaloDataPipeline,
    'cod': CallOfDutyDataPipeline,
    'tetris': TetrisDataPipeline,
    'sf': partial(FightingGamesDataPipeline, games=['street_fighter']),
    'tek': partial(FightingGamesDataPipeline, games=['tekken']),
    'kof': partial(FightingGamesDataPipeline, games=['king_of_fighters']),
    'gg': partial(FightingGamesDataPipeline, games=['guilty_gear']),
    'fifa' : FIFADataPipeline,
}


def run_pipeline(games, action, num_processes=1, **kwargs):
    """run ingenstion and/or processing for the specified games"""
    if num_processes > 1:
        pool = Pool(num_processes)
        map_fn = pool.map
    else:
        map_fn = map

    data_pipelines = [GAME_CLASS_MAP[game](**kwargs) for game in games]

    if action in {'ingest', 'all'}:
        list(map_fn(methodcaller('ingest_data'), data_pipelines))

    if num_processes > 1:
        pool.close()
        pool.join()

    if action in {'process', 'all'}:
        list(map(methodcaller('process_data'), data_pipelines))

    postprocess(
        args['train_end_date'],
        args['test_end_date'],
        args['min_rows_year']
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_games_argparser()])
    parser.add_argument(
        '--action',
        type=str,
        required=False,
        choices=['ingest', 'process', 'all'],
        default='all',
    )
    parser.add_argument('-t', '--timeout', type=float, required=False)
    parser.add_argument('-mr', '--max_rows', type=int, required=False)
    parser.add_argument('-kr', '--keys_to_refresh', type=delimited_list, required=False, default=[])
    parser.add_argument('-np', '--num_processes', type=int, required=False, default=1)
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('--min_rows_year', type=int, default=100, help='minmum number of rows in a year to begin including data')
    args = vars(parser.parse_args())
    args = {key: val for key, val in args.items() if val is not None}
    run_pipeline(**args)
