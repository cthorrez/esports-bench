"""module for sweeping hyperparameters"""
import os
import pathlib
import json
import warnings
from collections import defaultdict
from contextlib import ExitStack
from functools import lru_cache
from multiprocessing import get_context
import numpy as np
from riix.eval import grid_search
from esportsbench.datasets import load_dataset, BASE_DATA_DIR
from esportsbench.constants import RATING_SYSTEM_NAME_CLASS_MAP, GAME_NAME_MAP

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)


def round_dict(dic, precision=4):
    out_dic = {}
    for key, val in dic.items():
        if isinstance(val, float):
            out_dic[key] = round(val, precision)
        else:
            out_dic[key] = val
    return out_dic

def construct_param_configurations(
    param_configs,
    num_samples: int,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    param_values = {}
    for param_name, param_config in param_configs.items():
        if param_name == 'model': continue
        
        if (hasattr(param_config, 'param_type')) and (getattr(param_config, 'param_type') == 'list'):
            idxs = rng.choice(len(param_config.options), size=num_samples)
            # doing list instead of array since this could be various dtypes
            sampled_values = [param_config.options[int(idx)] for idx in idxs]
        else:
            sampled_values = rng.uniform(low=param_config.min_value, high=param_config.max_value, size=num_samples)

        param_values[param_name] = sampled_values

    param_samples = []
    for sample_idx in range(num_samples):
        sample = {}
        for param_name in param_configs.keys():
            if param_name == 'model': continue
            sample[param_name] = param_values[param_name][sample_idx]
        param_samples.append(sample)
    return param_samples


@lru_cache(maxsize=2)
def load_train_dataset(game, **dataset_kwargs):
    """load the train split of a game, during sweeping the train data is the val data
    cached so a worker process loads each game once rather than once per rating system"""
    dataset, test_mask = load_dataset(game, **dataset_kwargs)
    train_rows = int(np.logical_not(test_mask).sum())
    return dataset[:train_rows]


def sweep_one(job):
    """run every param configuration for one (game, rating system) and return the best"""
    game, rating_system_key, rating_system_name, param_configurations, dataset_kwargs = job
    dataset = load_train_dataset(game, **dataset_kwargs)
    best_params, best_metrics = grid_search(
        rating_system_class=RATING_SYSTEM_NAME_CLASS_MAP[rating_system_name],
        dataset=dataset,
        metrics_mask=np.ones(len(dataset), dtype=np.bool_),
        param_configurations=param_configurations,
        metric='log_loss',
        minimize_metric=True,
        # riix traces hyperparameters so sequential fits share one compiled scan,
        # this is far faster than a pool per sweep where every worker recompiles
        num_processes=None,
    )
    return game, rating_system_key, best_params, best_metrics


def sweep(
    games,
    data_dir,
    granularity,
    sweep_config,
    train_end_date,
    test_end_date,
    rating_period='7D',
    drop_draws=False,
    num_samples=100,
    num_processes=8,
    rating_systems='all',
):
    """sweep hyperparameters for each (game, rating system) pair, running num_processes pairs in parallel"""
    sweep_results = defaultdict(dict)
    results_dir = pathlib.Path(__file__).parents[1] / 'experiments' / 'conf' / 'sweep_results' / f'{granularity}_sweep_{rating_period}_{num_samples}'
    dataset_kwargs = {
        'rating_period': rating_period,
        'drop_draws': drop_draws,
        'train_end_date': train_end_date,
        'test_end_date': test_end_date,
        'data_dir': data_dir,
    }

    jobs = []
    for game in games:
        game_sweep_config = sweep_config if granularity == 'broad' else sweep_config[game]
        rating_system_keys = list(game_sweep_config.keys())
        if rating_systems != 'all':
            rating_system_keys = [key for key in rating_system_keys if key in rating_systems]
        for rating_system_key in rating_system_keys:
            rating_system_name = game_sweep_config[rating_system_key]['model']
            if rating_system_name not in RATING_SYSTEM_NAME_CLASS_MAP:
                print(f'Skipping {rating_system_key}: model {rating_system_name} is not in RATING_SYSTEM_NAME_CLASS_MAP')
                continue
            param_configurations = construct_param_configurations(
                param_configs=game_sweep_config[rating_system_key], num_samples=num_samples
            )
            jobs.append((game, rating_system_key, rating_system_name, param_configurations, dataset_kwargs))

    # biggest games first so the long sweeps don't end up running alone at the end
    data_path = BASE_DATA_DIR / data_dir / 'parquet'
    jobs.sort(key=lambda job: os.path.getsize(data_path / f'{GAME_NAME_MAP.get(job[0], job[0])}.parquet'), reverse=True)
    print(f'Sweeping {len(jobs)} (game, rating system) pairs over {num_samples} configurations each')

    with ExitStack() as stack:
        if num_processes > 1:
            # spawn, not fork: forking after jax has initialized is unsafe
            pool = stack.enter_context(get_context('spawn').Pool(min(num_processes, len(jobs))))
            results = pool.imap_unordered(sweep_one, jobs)
        else:
            results = map(sweep_one, jobs)

        for idx, (game, rating_system_key, best_params, best_metrics) in enumerate(results):
            print(f'[{idx + 1}/{len(jobs)}] {game} {rating_system_key}: {round_dict(best_metrics)} {round_dict(best_params)}')
            out_dict = {'best_params': best_params.copy(), 'best_metrics': best_metrics.copy()}
            out_dir = results_dir / game
            os.makedirs(out_dir, exist_ok=True)
            json.dump(out_dict, open(out_dir / f'{rating_system_key}.json', 'w'), indent=2)
            sweep_results[game][rating_system_key] = best_params
    return sweep_results


