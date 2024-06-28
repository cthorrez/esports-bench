import yaml
import warnings
import hydra
from omegaconf import DictConfig, OmegaConf
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

    sweep(
        games=games,
        rating_systems=config.rating_systems,
        data_dir=config.data_dir,
        granularity='broad',
        sweep_config=config.sweep_config,
        rating_period=config.rating_period,
        train_end_date=config.train_end_date,
        test_end_date=config.test_end_date,
        num_samples=config.num_samples,
        num_processes=config.num_processes,
    )

if __name__ == '__main__':
    main()