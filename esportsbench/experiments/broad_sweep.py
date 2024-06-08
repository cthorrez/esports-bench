import yaml
import warnings
from dacite import from_dict
from esportsbench.eval.sweep_config import ExperimentSweepConfig
from esportsbench.arg_parsers import get_games_argparser
from esportsbench.eval.sweep import sweep

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)


def main(
    games,
    data_dir,
    config_file,
    train_end_date,
    test_end_date,
    drop_draws=False,
    num_processes=8,
):
    config = from_dict(data_class=ExperimentSweepConfig, data=yaml.full_load(open(config_file)))
    sweep(
        games,
        config,
        data_dir=data_dir,
        granularity='broad',
        drop_draws=drop_draws,
        rating_period=config.rating_period,
        train_end_date=train_end_date,
        test_end_date=test_end_date,
        num_samples=config.num_samples,
        num_processes=num_processes,
    )

if __name__ == '__main__':
    parser = get_games_argparser()
    parser.add_argument('-d', '--data_dir', type=str, default='hf_data')
    parser.add_argument('-c', '--config_file', type=str, default='configs/broad_sweep_config.yaml', required=False)
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-np', '--num_processes', type=int, default=12, required=False)
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    args = parser.parse_args()
    main(**vars(args))