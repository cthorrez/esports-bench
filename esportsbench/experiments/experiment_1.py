import os
import math
import yaml
import json
import warnings
from typing import Dict
import numpy as np
import pandas as pd
from dacite import from_dict
from riix.eval import grid_search
from esportsbench.datasets import load_dataset
from esportsbench.eval.bench import RATING_SYSTEM_MAP
from esportsbench.eval.sweep_config import ExperimentSweepConfig, ParamSweepConfig
from esportsbench.arg_parsers import get_games_argparser
from esportsbench.eval.sweep import sweep

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main(
    games,
    config_file,
    drop_draws=False,
    rating_period='7D',
    train_end_date='2023-03-31',
    test_end_date='2024-03-31',
    num_samples=100,
    num_processes=8,
    seed=42,
):
    config = from_dict(data_class=ExperimentSweepConfig, data=yaml.full_load(open(config_file)))
    sweep(
        games,
        config,
        drop_draws=drop_draws,
        rating_period=rating_period,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
        num_samples=num_samples,
        num_processes=num_processes,
        seed=42,
    )



if __name__ == '__main__':
    parser = get_games_argparser()
    parser.add_argument('-c', '--config_file', type=str, default='configs/sweep_config.yaml', required=False)
    parser.add_argument('-rp', '--rating_period', type=str, required=False, default='7D')
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-ns', '--num_samples', type=int, default=100, required=False)
    parser.add_argument('-np', '--num_processes', type=int, default=12, required=False)
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    parser.add_argument('--seed', type=int, default=42, required=False)
    args = parser.parse_args()
    main(**vars(args))