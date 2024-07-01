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

def main():
    pass

if __name__ == '__main__':
    main()