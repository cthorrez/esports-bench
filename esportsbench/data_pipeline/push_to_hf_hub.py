import argparse
from datasets import load_dataset, Features, Value
import huggingface_hub
from esportsbench.constants import GAME_NAMES


def main(version, tag=None, prod=False):
    features = Features({
        'date': Value('date32'),
        'competitor_1': Value('string'),
        'competitor_2': Value('string'),
        'outcome': Value('float64'),
        'match_id': Value('string'),
        'page': Value('string'),
    })
    gs = [g for g in GAME_NAMES if 'dota2' not in g]
    dataset = load_dataset(
        path=f'../../data/final_data_v{version}/parquet',
        data_files= {game_name: f'{game_name}.parquet' for game_name in gs},
        features=features,
    )
    print('Dataset loaded.')
    dataset.save_to_disk(f'../../data/EsportsBench{version}')
    print('Dataset saved to disk.')

    repo_id = 'cthorrez/EsportsBenchTest'
    if prod:
        repo_id = 'EsportsBench/EsportsBench'

    dataset.push_to_hub(
        repo_id,
        commit_message=f'update to {version}',
        revision=version.replace('_', '.')
    )
    print('Dataset pushed to hub.')
    if tag is not None:
        huggingface_hub.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset")
        print('Tag applied')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str, default='3_1')
    parser.add_argument('--tag', default=None, required=False)
    parser.add_argument('--prod', action='store_true')
    main(**vars(parser.parse_args()))