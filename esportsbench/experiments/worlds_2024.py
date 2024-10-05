"""It's Worlds Baby"""
import json
from datetime import datetime, timedelta
import numpy as np
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.glicko2 import Glicko2
from riix.models.trueskill import TrueSkill
from esportsbench.datasets import load_dataset


# for copy paste reference
teams = [
    'MAD Lions KOI',
    'Vikings Esports (2023 Vietnamese Team)',
    'PSG Talon',
    'PaiN Gaming',
    'GAM Esports',
    'Fukuoka SoftBank HAWKS gaming',
    '100 Thieves',
    'Movistar R7',
    'Hanwha Life Esports',
    'T1',
    'Bilibili Gaming',
    'G2 Esports',
    'FlyQuest',
    'Gen.G',
    'Top Esports',
    'Fnatic',
    'Weibo Gaming',
    'Dplus KIA',
    'Team Liquid',
    'LNG Esports'
]

def main():
    matches = [
        # play in
        # ('MAD Lions KOI', 'Vikings Esports (2023 Vietnamese Team)'),
        # ('PSG Talon', 'PaiN Gaming'),
        # ('GAM Esports', 'Fukuoka SoftBank HAWKS gaming'),
        # ('100 Thieves', 'Movistar R7'),
        # ('MAD Lions KOI', 'PSG Talon'),
        # ('Fukuoka SoftBank HAWKS gaming', '100 Thieves'),
        # ('GAM Esports', 'Movistar R7'),
        # ('Vikings Esports (2023 Vietnamese Team)', 'PaiN Gaming'),
        # ('Movistar R7', 'PaiN Gaming'),
        # ('PSG Talon', '100 Thieves'),

        # swiss round 1
        # ('Bilibili Gaming', 'MAD Lions KOI'),
        # ('Top Esports', 'T1'),
        # ('Gen.G', 'Weibo Gaming'),
        # ('Fnatic', 'Dplus KIA'),
        # ('Team Liquid', 'LNG Esports'),
        # ('Hanwha Life Esports', 'PSG Talon'),
        # ('FlyQuest', 'GAM Esports'),
        # ('G2 Esports', 'PaiN Gaming'),

        # swiss round 2
        # ('Bilibili Gaming', 'LNG Esports'),
        # ('Gen.G', 'Top Esports',),
        # ('G2 Esports', 'Hanwha Life Esports'),
        # ('Weibo Gaming', 'Team Liquid'),
        # ('T1', 'PaiN Gaming'),
        # ('Dplus KIA', 'FlyQuest'),
        # ('Fnatic', 'GAM Esports'),
        # ('PSG Talon', 'MAD Lions KOI'),

        # swiss round 3
        ('Dplus KIA', 'LNG Esports'),
        ('Hanwha Life Esports', 'Gen.G'),
        ('Top Esports', 'Fnatic'),
        ('Bilibili Gaming', 'T1'),
        ('PSG Talon', 'FlyQuest'),
        ('Weibo Gaming', 'G2 Esports'),
        ('MAD Lions KOI', 'GAM Esports'),
        ('PaiN Gaming', 'Team Liquid'),
    ]


    model_config = {
        'elo': Elo,
        'glicko': Glicko,
        'glicko2': Glicko2,
        'trueskill': TrueSkill
    }
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    dataset, _ = load_dataset(
        game='league_of_legends',
        rating_period='1D',
        test_end_date=tomorrow,
        data_dir='full_data'
    )

    # fit each of the models
    models = {}
    for model_name, model_class in model_config.items():
        params = json.load(open(f'conf/sweep_results/fine_sweep_1D_1000/lol/{model_name}.json'))
        model = model_class(competitors=dataset.competitors, **params['best_params'])
        model.fit_dataset(dataset)
        models[model_name] = model

    for team in teams:
        print(f'{team} Elo: {models["elo"].ratings[dataset.competitor_to_idx[team]]}')



    for team_1, team_2 in matches:
        team_1_id = dataset.competitor_to_idx[team_1]
        team_2_id = dataset.competitor_to_idx[team_2]
        print(f'Predictions for {team_1} vs {team_2}:')
        for model_name, model in models.items():
            prob = model.predict(matchups=np.array([[team_1_id, team_2_id]]))
            print(f'{model_name}: {prob}')
        print('')


if __name__ == '__main__':
    main()