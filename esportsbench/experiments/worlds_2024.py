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
    'Movistar R7'

]

def main():
    matches = [
        ('MAD Lions KOI', 'Vikings Esports (2023 Vietnamese Team)'),
        ('PSG Talon', 'PaiN Gaming'),
        ('GAM Esports', 'Fukuoka SoftBank HAWKS gaming'),
        ('100 Thieves', 'Movistar R7'),
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