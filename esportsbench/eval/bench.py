"""module for runninng benchmarks"""
import os
import json
from functools import partial
import multiprocessing
from collections import defaultdict
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
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.models.baselines import BaselineRatingSystem
from esportsbench.arg_parsers import get_games_argparser, comma_separated
from esportsbench.datasets import load_dataset
from esportsbench.constants import GAME_NAME_MAP, ALL_RATING_SYSTEM_NAMES, RATING_SYSTEM_NAME_CLASS_MAP




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
    mean_metrics = {}
    for sys, metrics in rating_system_sums.items():
        mean_dict = {}
        for metric, total in metrics.items():
            mean_dict[metric] = total / rating_system_counts[sys][metric]
        mean_metrics[sys] = mean_dict

    # Add the 'mean' key at the top level of the data_dict
    data_dict['mean'] = mean_metrics

    return data_dict


def eval_func(input_tuple):
    game_name, rating_system_name, dataset, rating_system_class, params, test_mask = input_tuple
    rating_system = rating_system_class(competitors=dataset.competitors, **params)
    metrics = evaluate(rating_system, dataset, metrics_mask=test_mask)
    # rating_system.print_leaderboard(5)
    return (game_name, rating_system_name, metrics)

def run_benchmark(
    games,
    rating_period,
    train_end_date,
    test_end_date,
    data_dir,
    drop_draws=False,
    max_rows=None,
    rating_systems=ALL_RATING_SYSTEM_NAMES,
    hyperparameter_config='default',
    num_processes=8,
):
    """run a benchmark where all rating systems use default values"""
    results = defaultdict(dict)

    def eval_iterator():
        for game_short_name in games:
            game_name = GAME_NAME_MAP[game_short_name]
            print(game_name)
            dataset, test_mask = load_dataset(
                game=game_name,
                rating_period=rating_period,
                drop_draws=drop_draws,
                max_rows=max_rows,
                train_end_date=train_end_date,
                test_end_date=test_end_date,
                data_dir=data_dir,
            )

            if isinstance(hyperparameter_config, dict):
                rating_system_keys = [key for key in hyperparameter_config[game_short_name].keys()]
            else:
                rating_system_keys = rating_systems

            for rating_system_key in rating_system_keys:
                print(f'\nEvaluating {rating_system_key} on {game_short_name}')
                rating_system_name = hyperparameter_config[game_short_name][rating_system_key].get('model', rating_system_key)
                rating_system_class = RATING_SYSTEM_NAME_CLASS_MAP[rating_system_name]
                params = {}
                if hyperparameter_config == 'default':
                    print('No hyperparameter config specified, using class default hyperparameters')
                elif isinstance(hyperparameter_config, str):
                    params_path = f'{hyperparameter_config}/{game_short_name}/{rating_system_name}.json'
                    if os.path.exists(params_path):
                        params = json.load(open(params_path))['best_params']
                        print(f'Using hyperparameters from {params_path}')
                    else:
                        print(f"couldn't find param config for {rating_system_name} on {game_name}, Exiting.")
                        raise FileNotFoundError
                elif isinstance(hyperparameter_config, dict):
                    print('Using provided hyperparameters:')
                    params = hyperparameter_config[game_short_name][rating_system_key]
                    print(params)
                else:
                    print(hyperparameter_config)
                    raise ValueError('Expected config to be either a path or a dict')
                if 'model' in params: del params['model']
                yield (game_name, rating_system_key, dataset, rating_system_class, params, test_mask)
            
    pool = multiprocessing.Pool(processes=num_processes)
    eval_results = pool.imap(eval_func, eval_iterator())
    for game_name, rating_system_name, metrics in eval_results:
        results[game_name][rating_system_name] = metrics

    results = add_mean_metrics(results)
    return results


def print_results(data_dict):
    """print results to stdout"""
    header_line = (
        f"{'Game':<18}{'Rating System':<30}{'Accuracy':>10}{'Log Loss':>10}{'Brier Score':>12}{'Duration (s)':>10}"
    )
    print(header_line)
    print('=' * len(header_line))

    filler = '                                 '
    for game, rating_systems in data_dict.items():
        acc_latex_line = f'{game} & Accuracy'
        ll_latex_line =f'{filler} & Log Loss'
        for rating_system_name in rating_systems.keys():
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


if __name__ == '__main__':
    parser = get_games_argparser()
    parser.add_argument(
        '-rs',
        '--rating_systems',
        type=comma_separated(ALL_RATING_SYSTEM_NAMES),
        default=ALL_RATING_SYSTEM_NAMES,
    )
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    parser.add_argument('-rp', '--rating_period', type=str, required=False, default='7D')
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-d', '--data_dir', type=str, default='hf_data/v1_0')
    parser.add_argument('-c', '--hyperparameter_config', type=str, required=False, default='default')
    parser.add_argument('-np', '--num_processes', type=int, default=8)
    args = parser.parse_args()

    results = run_benchmark(
        args.games,
        args.rating_systems,
        args.rating_period,
        train_end_date=args.train_end_date,
        test_end_date=args.test_end_date,
        data_dir=args.data_dir,
        drop_draws=args.drop_draws,
        hyperparameter_config=args.hyperparameter_config,
        num_processes=args.num_processes,
    )
    print_results(results)
