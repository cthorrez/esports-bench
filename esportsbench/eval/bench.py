"""module for runninng benchmarks"""
import os
import functools
import math
import json
from functools import partial
from collections import defaultdict
import numpy as np
import pandas as pd
from riix.eval import evaluate
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.glicko2 import Glicko2
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin import WengLin
from riix.models.melo import Melo
from riix.models.gen_elo import GenElo
from riix.models.constant_variance_glicko import ConstantVarianceGlicko
from riix.models.velo import vElo
from riix.models.online_disc_decomp import OnlineDiscDecomp
from riix.models.online_rao_kupper import OnlineRaoKupper
from riix.models.elo_davidson import EloDavidson
from riix.models.skf import VSKF
from riix.models.elomentum import EloMentum
from riix.models.yuksel_2024 import Yuksel2024
from esportsbench.arg_parsers import get_games_argparser, comma_separated
from esportsbench.datasets import load_dataset
from esportsbench.constants import GAME_NAME_MAP

RATING_SYSTEM_MAP = {
    'elo': Elo,
    'glicko': Glicko,
    'glicko2': Glicko2,
    'trueskill': TrueSkill,
    'wl_bt': partial(WengLin, model='bt', tau=0.0),
    'wl_tm': partial(WengLin, model='tm', tau=0.0),
    'melo': Melo,
    'genelo': GenElo,
    # 'cvglicko': ConstantVarianceGlicko,
    'velo': vElo,
    # 'im': IterativeMarkov, # these 2 are so bad it's not even worth comparing in most experiments
    # 'tm': TemporalMassey,
    'vskf_bt': partial(VSKF, model='bt'),
    'vskf_tm': partial(VSKF, model='tm'),
    # 'odd': OnlineDiscDecomp,
    # 'ork': OnlineRaoKupper,
    # 'elod': EloDavidson,
    # 'elom': EloMentum,
    # 'yuksel': Yuksel2024,
}


def add_mean_metrics(data_dict):
    """get overall mean for each rating system and metric at the game level"""
    # Initialize a structure to store sum and counts for calculating means
    rating_system_sums = defaultdict(lambda: defaultdict(float))
    rating_system_counts = defaultdict(lambda: defaultdict(int))

    # Collect sums and counts for each metric in each rating system
    for rating_systems in data_dict.values():
        for rating_system, metrics in rating_systems.items():
            for metric, value in metrics.items():
                rating_system_sums[rating_system][metric] += value
                rating_system_counts[rating_system][metric] += 1

    # Calculate mean metrics for each rating system
    mean_metrics = {
        sys: {metric: total / rating_system_counts[sys][metric] for metric, total in metrics.items()}
        for sys, metrics in rating_system_sums.items()
    }

    # Add the 'mean' key at the top level of the data_dict
    data_dict['mean'] = mean_metrics

    return data_dict





def run_benchmark(
    games,
    rating_systems,
    rating_period,
    train_end_date='2023-03-31',
    test_end_date='2024-03-31',
    drop_draws=False,
    max_rows=None,
    params_dir=None,
):
    """run a benchmark where all rating systems use default values"""
    results = defaultdict(dict)
    for game_short_name in games:
        game_name = GAME_NAME_MAP[game_short_name]
        print(f'evaluating {game_name}')
        dataset, test_mask = load_dataset(
            game=game_name,
            rating_period=rating_period,
            drop_draws=drop_draws,
            max_rows=max_rows,
            train_end_date=train_end_date,
            test_end_date=test_end_date
        )

        for rating_system_name in rating_systems:
            rating_system_class = RATING_SYSTEM_MAP[rating_system_name]
            name = rating_system_name
            params = {}
            if params_dir:
                params_path = f'{params_dir}/{game_short_name}/{rating_system_name}.json'
                if os.path.exists(params_path):
                    params = json.load(open(params_path))['best_params']
                    print(f'using params from {params_path}')
                    print(params)
                else:
                    print(f"couldn't find param config for {rating_system_name} on {game_name}, using default params")

            print(f'running {name}')
            rating_system = rating_system_class(competitors=dataset.competitors, **params)
            train_mask = np.logical_not(test_mask)
            # metrics = evaluate(rating_system, dataset, metrics_mask=train_mask)
            metrics = evaluate(rating_system, dataset, metrics_mask=test_mask)
            results[game_name][name] = metrics
    return results


def print_results(data_dict):
    """print results to stdout"""
    header_line = (
        f"{'Game':<18}{'Rating System':<30}{'Accuracy':>10}{'Log Loss':>10}{'Brier Score':>12}{'Duration (s)':>10}"
    )
    print(header_line)
    print('=' * len(header_line))

    for game, rating_systems in data_dict.items():
        acc_latex_line = f'{game} & accuracy'
        ll_latex_line =f'{game} & log loss'
        rating_system_names = ['elo', 'glicko', 'glicko2', 'trueskill', 'wl_bt', 'wl_tm', 'melo', 'genelo', 'vskf_bt', 'vskf_tm', 'velo']
        for rating_system_name in rating_system_names:
            metrics = rating_systems[rating_system_name]
            accuracy = f"{metrics.get('accuracy', 'N/A'):.4f}".rjust(10)
            accuracy_without_draws = f"{metrics.get('accuracy_without_draws', 'N/A'):.4f}".rjust(10)
            log_loss = f"{metrics.get('log_loss', 'N/A'):.4f}".rjust(10)
            brier_score = f"{metrics.get('brier_score', 'N/A'):.4f}".rjust(12)
            duration = f"{metrics.get('duration', 'N/A'):.4f}".rjust(10)
            # print(f'{game:<18}{rating_system_name:<30}{accuracy}{log_loss}{brier_score}{duration}')
            print(f'{game:<18}{rating_system_name:<30}{accuracy_without_draws}{log_loss}{brier_score}{duration}')

            acc_latex_line += f' & {accuracy_without_draws}'
            ll_latex_line += f' & {log_loss}'
        # print(acc_latex_line.replace("     ", " ") + ' \\\\')
        # print(ll_latex_line.replace("     ", " ") + ' \\\\')




def main(
    games,
    rating_systems,
    rating_period,
    train_end_date,
    test_end_date,
    drop_draws=False,
    max_rows=None,
    params_dir=None,
):
    """process arguments and launch the benchmark"""

    results = run_benchmark(
        games,
        rating_systems,
        rating_period,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
        drop_draws=drop_draws,
        max_rows=max_rows,
        params_dir=params_dir,
    )
    results = add_mean_metrics(results)
    print_results(results)


if __name__ == '__main__':
    all_rating_systems = list(RATING_SYSTEM_MAP.keys())
    parser = get_games_argparser()
    parser.add_argument(
        '-rs',
        '--rating_systems',
        type=comma_separated(all_rating_systems),
        default=all_rating_systems,
    )
    parser.add_argument('-mr', '--max_rows', type=int, required=False)
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    parser.add_argument('-rp', '--rating_period', type=str, required=False, default='7D')
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-pd', '--params_dir', type=str, required=False)
    args = parser.parse_args()
    main(
        games=args.games,
        rating_systems=args.rating_systems,
        drop_draws=args.drop_draws,
        max_rows=args.max_rows,
        rating_period=args.rating_period,
        train_end_date=args.train_end_date,
        test_end_date=args.test_end_date,
        params_dir=args.params_dir,
    )
