"""
This script will run a full rating system experiment
  1. run a broad hyperparameter sweep
  2. run a fine hyperparameter sweep centered around the results of the broad sweep
  3. run the evaluation pipeline with the best hyperparameters identified by the fine sweep on the test set, report train and test numbers
"""
import warnings
import hydra
from omegaconf import DictConfig
from esportsbench.eval.sweep import sweep
from esportsbench.constants import GAME_SHORT_NAMES

# Suppress overflow warnings since many of the combinations swept over are expected to be numerically unstable
warnings.filterwarnings('ignore', category=RuntimeWarning)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    if config.games == 'all':
        games = GAME_SHORT_NAMES
    else:
        games = config.games

    common_sweep_args = {
        'data_dir' : config.data_dir,
        'rating_period': config.rating_period,
        'train_end_date' : config.train_end_date,
        'test_end_date' : config.test_end_date,
        'num_samples' : config.num_samples,
        'num_processes' : config.num_processes,
    }


    for game in games:
        sweep(
            games=[game],
            rating_systems=config.rating_systems,
            granularity='broad',
            sweep_config=config.broad_sweep_config,
            **common_sweep_args
        )

        fine_sweep_config = 'TODO'
        sweep(
            games=[game],
            rating_systems=config.rating_systems,
            granularity='fine',
            sweep_config=fine_sweep_config,
            **common_sweep_args,
        )

if __name__ == '__main__':
    main()