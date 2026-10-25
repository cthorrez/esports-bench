"""where the data lives, defaults to data/ in the repo, override with the ESPORTSBENCH_DATA_DIR environment variable"""
import os
import pathlib

DATA_DIR = pathlib.Path(os.environ.get('ESPORTSBENCH_DATA_DIR', pathlib.Path(__file__).resolve().parents[1] / 'data'))
