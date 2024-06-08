## EsportsBench
The EsportsBench datasets are meant to facilitate research and experimentation on real world competition data spanning many years of competitions including a diverse range of genres and competition formats.

## Licenses
This *code* is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/


### Setup
To collect the data yourself, you will need to obtain Aliculac and Liquipedia LPDB API key. Aligulac keys can be generated at [http://aligulac.com/about/api/](http://aligulac.com/about/api/) and Liquipedia LPDB keys can be requested in the Liquipedia [discord server](https://discord.gg/hW3T8BQr).

Add your key(s) to `.dotenv-template` and rename it to `.env` so that the data pipelines can access them.


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

Install requirements
```bash
pip install riix==0.0.3
pip install -e .
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
