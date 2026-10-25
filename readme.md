## EsportsBench
The EsportsBench datasets are meant to facilitate research and experimentation on real world competition data spanning many years of competitions including a diverse range of genres and competition formats.

## Licenses
This *code* is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/


### Setup
Install the locked environment with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

To collect the data yourself, you will need an Aligulac API key, a Liquipedia LPDB API key and a Fandom account for Leaguepedia. Aligulac keys can be generated at [http://aligulac.com/about/api/](http://aligulac.com/about/api/) and Liquipedia LPDB keys can be requested in the Liquipedia [discord server](https://discord.gg/hW3T8BQr).

Copy `.env-template` to `.env` and fill in your keys so that the data pipelines can access them. Data is stored in `data/`, set `ESPORTSBENCH_DATA_DIR` to use another location.

### Updating the data and making a release
One command refreshes every game, builds the next `data/final_data_v{N}` (test set = the year ending at `--as-of`, train set = everything before), sanity checks it against the previous release and optionally pushes it to Hugging Face:
```bash
uv run python -m esportsbench.release --as-of 2026-12-31               # build and validate only
uv run python -m esportsbench.release --as-of 2026-12-31 --push test   # also push to the test repo and verify the upload
uv run python -m esportsbench.release --as-of 2026-12-31 --push prod   # test repo first, then prod
```
Refreshing only re-fetches matches from the last 90 days before the newest stored match, so it takes minutes. Edits to older matches are only picked up by a full refresh, which is limited by the 60 requests/hour/wiki LPDB rate limit so takes hours. Run one occasionally, for all games or just some:
```bash
uv run python -m esportsbench.data_pipeline.get_data_from_source --full -g wc3,sc1
```

### Benchmarks and sweeps on the latest release
`bench.py` and the sweeps default to the newest release and its train/test dates:
```bash
uv run python -m esportsbench.eval.bench -c esportsbench/experiments/conf/sweep_results/fine_sweep_7D_1000
uv run python esportsbench/experiments/broad_sweep.py num_samples=1000
uv run python esportsbench/experiments/fine_sweep.py sweep_results_path=broad_sweep_7D_1000 num_samples=1000
```
Sweep options are `key=value` overrides of `esportsbench/experiments/conf/config.yaml`, e.g. `games=[lol,cs] rating_systems=[elo,glicko]`.


### Reproduce Experiments
To exactly reproduce the results of Table 2 of the paper, follow these steps exactly:

Clone the repo and enter:
```bash
git clone https://github.com/cthorrez/esports-bench
cd esports-bench
```

Checkout the v0.0.1 tag
```bash
git checkout tags/v0.0.1
```

Create conda env
```bash
conda create -n esportsbench python=3.11
```

Activate env
```bash
conda activate esportsbench
```

Install requirements
```bash
pip install riix==0.0.3
pip install -e .
```

Download the data from HuggingFace
```bash
cd esportsbench
python data_pipeline/get_data_from_hf.py
```

Run the broad hyperparameter sweep
```bash
python experiments/broad_sweep.py --config_file experiments/configs/broad_sweep_config.yaml
```

Run the fine hyperparameter sweep
```bash
python experiments/fine_sweep.py --config_dir experiments/sweep_results/broad_sweep_7D_1000
```

Run the evaluation script using the best parameters
```bash
python eval/bench.py --config_dir experiments/sweep_results/fine_sweep_7D_1000/
```

Note each sweep will take many hours, it takes 5 hours on my desktop and 17 on my laptop.
You can use the `-np <num_cores>` flag to increase the number of processes utilized to be the number of cores you have 

### Data Licences
The data collected by these pipelines is collected from different sources with their own licenses. If you reproduce the the data collection and experiments understand the retrieved data and results falls under those licenses.

The StarCraft II data is from [Aligulac](http://aligulac.com/)

The League of Legends data is from [Leaguepedia](https://lol.fandom.com/) under a [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

The data for all other games is from [Liquipedia](https://liquipedia.net/) under a [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

## Citation
If you use EsportsBench in your research, please cite this repo:

```bibtex
@article{thorrez2024esportsbench,
  title={EsportsBench: A Collection of Datasets for Benchmarking Rating Systems in Esports},
  author={Thorrez, Clayton},
  year={2024},
  url={https://github.com/cthorrez/esports-bench}
}
```
