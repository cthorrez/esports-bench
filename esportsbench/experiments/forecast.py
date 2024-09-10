"""Ad hoc script to make some predictions"""
from collections import defaultdict
import json
import numpy as np
from riix.models.elo import Elo
from riix.models.glicko2 import Glicko2
from esportsbench.datasets import load_dataset
from esportsbench.constants import GAME_NAME_MAP

GAME_ACRONYM_MAP = {v:k for k,v in GAME_NAME_MAP.items()}


def main():
    matches = [
        ('dota2', 'team liquid', 'cloud9'),
        ('dota2', 'tundra esports', 'gaimin gladiators'),
        ('dota2', 'betboom team', 'team falcons'),
        ('dota2', 'beastcoast', 'heroic'),
        ('dota2', 'team spirit', 'g2 x ig'),
        ('rocket_league', 'G2 Esports', 'Limitless'),
        ('rocket_league', 'Team Falcons', 'Gaimin Gladiators'),
        ('rocket_league', 'Gentle Mates Alpine', 'Pioneers'),
        ('rocket_league', 'Team Vitality', 'Team Secret'),
        ('rocket_league', 'Gen.G Mobil1 Racing', 'PWR'),
        ('rocket_league', 'FURIA Esports', 'OG'),
        ('rocket_league', 'Team BDS', 'Oxygen Esports'),
        ('rocket_league', 'Spacestation Gaming', 'Karmine Corp'),
        ('league_of_legends', 'T1', 'Dplus KIA'),
        ('league_of_legends', 'KT Rolster', 'BNK FearX'),
        ('counterstrike', 'mouz', 'rooster'),
        ('counterstrike', 'big', 'imperial esports'),
        ('counterstrike', 'complexity gaming', 'm80'),
        ('counterstrike', 'fnatic', 'astralis'),
        ('overwatch', 'Team Falcons', 'ZETA DIVISION'),
        ('rainbow_six', 'Soniqs', 'M80'),
        ('rainbow_six', 'DarkZero Esports', 'Spacestation Gaming'),
        ('rainbow_six', 'Luminosity Gaming', 'Cloud9 Beastcoast'),
        ('rainbow_six', 'Luminosity Gaming', 'Beastcoast'),
        ('rainbow_six', 'Wildcard Gaming', 'Oxygen Esports'),
    ]

    games = set([m[0] for m in matches])
  
    split_matchups = defaultdict(list)
    for match in matches:
        split_matchups[match[0]].append((match[1], match[2]))

    for game in games:
        print(f'forecasting for {game}')
        dataset, _ = load_dataset(
            game=game,
            rating_period='1D',
            test_end_date='2024-09-09',
            data_dir='full_data'
        )
        
        elo_params = json.load(open(f'conf/sweep_results/fine_sweep_1D_1000/{GAME_ACRONYM_MAP[game]}/elo.json'))
        elo = Elo(competitors=dataset.competitors, **elo_params['best_params'])
        elo.fit_dataset(dataset)

        glicko2_params = json.load(open(f'conf/sweep_results/fine_sweep_1D_1000/{GAME_ACRONYM_MAP[game]}/glicko2.json'))
        glicko2 = Glicko2(competitors=dataset.competitors, **glicko2_params['best_params'])
        glicko2.fit_dataset(dataset)

        for match in split_matchups[game]:
            c1, c2 = match
            c1_id = dataset.competitor_to_idx[c1]
            c2_id = dataset.competitor_to_idx[c2]
            elo_prob = elo.predict(matchups=np.array([[c1_id, c2_id]]))
            glicko2_prob = glicko2.predict(
                matchups=np.array([[c1_id, c2_id]]),
                time_step=dataset.time_steps[-1] + 1
            )
            print(c1, c2, elo_prob, glicko2_prob)
    


if __name__ == '__main__':
    main()