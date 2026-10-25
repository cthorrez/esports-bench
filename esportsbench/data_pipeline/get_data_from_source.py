"""main script for ingesting data from various sources"""
import argparse
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
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
from esportsbench.data_pipeline.rainbow_six import RainbowSixDataPipeline
from esportsbench.data_pipeline.halo import HaloDataPipeline
from esportsbench.data_pipeline.call_of_duty import CallOfDutyDataPipeline
from esportsbench.data_pipeline.tetris import TetrisDataPipeline
from esportsbench.data_pipeline.fighting_games import FightingGamesDataPipeline
from esportsbench.data_pipeline.eafc import EAFCDataPipeline
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
    'sf': partial(FightingGamesDataPipeline, game='street_fighter'),
    'tek': partial(FightingGamesDataPipeline, game='tekken'),
    'kof': partial(FightingGamesDataPipeline, game='king_of_fighters'),
    'gg': partial(FightingGamesDataPipeline, game='guilty_gear'),
    'eafc' : EAFCDataPipeline,
}


def ingest_rate_limit_group(data_pipelines, full=False, overlap_days=90):
    """ingest pipelines which share a rate limit one after another"""
    ingested_files = set()
    for data_pipeline in data_pipelines:
        raw_files = tuple(data_pipeline.request_params_groups)
        # e.g. the four fighting games all read the same raw file
        if raw_files not in ingested_files:
            data_pipeline.ingest_data(full=full, overlap_days=overlap_days)
            ingested_files.add(raw_files)


def run_pipeline(games, action, num_threads=None, full=False, overlap_days=90, **kwargs):
    """run ingestion and/or processing for the specified games

    each wiki/API has its own rate limit, so ingestion runs one thread per rate limit group
    """
    data_pipelines = [GAME_CLASS_MAP[game](**kwargs) for game in games]

    failed_groups = set()
    if action in {'ingest', 'all'}:
        groups = defaultdict(list)
        for data_pipeline in data_pipelines:
            groups[data_pipeline.rate_limit_group].append(data_pipeline)
        with ThreadPoolExecutor(num_threads or len(groups)) as executor:
            futures = {
                group: executor.submit(ingest_rate_limit_group, group_pipelines, full, overlap_days)
                for group, group_pipelines in groups.items()
            }
        for group, future in futures.items():
            if future.exception() is not None:
                failed_groups.add(group)
                print(f'ingestion failed for {group}:')
                traceback.print_exception(future.exception())

    if action in {'process', 'all'}:
        for data_pipeline in data_pipelines:
            if data_pipeline.rate_limit_group not in failed_groups:
                data_pipeline.process_and_write()

    if failed_groups:
        raise RuntimeError(f'ingestion failed for {sorted(failed_groups)}, their data was left unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_games_argparser()])
    parser.add_argument(
        '--action',
        type=str,
        required=False,
        choices=['ingest', 'process', 'all'],
        default='all',
    )
    parser.add_argument('--full', action='store_true', help='re-fetch everything instead of refreshing recent matches')
    parser.add_argument('--overlap_days', type=int, default=90, help='how far back from the newest stored match to re-fetch')
    parser.add_argument('-t', '--timeout', type=float, required=False)
    parser.add_argument('-mr', '--max_rows', type=int, required=False)
    parser.add_argument('-nt', '--num_threads', type=int, required=False, help='default: one per rate limit group')
    args = vars(parser.parse_args())
    args = {key: val for key, val in args.items() if val is not None}
    run_pipeline(**args)
