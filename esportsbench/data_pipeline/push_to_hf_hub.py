import argparse
from datetime import datetime
from datasets import load_dataset, Features, Value
import huggingface_hub
import pyarrow as pa
from esportsbench.constants import GAME_NAMES

def convert_date(date_string):
    if not date_string:
        return None
    try:
        date = datetime.strptime(date_string, "%Y-%m-%d")
        return pa.scalar(date, type=pa.date32())
    except ValueError:
        return None
    
def preprocess_function(example):
    example['date'] = convert_date(example['date'])
    return example
    

def main(version, tag=None, prod=False):
    features = Features({
        'date': Value('date32'),
        'competitor_1': Value('string'),
        'competitor_2': Value('string'),
        'outcome': Value('float64'),
        'match_id': Value('string'),
        'page': Value('string'),
    })

    dataset = load_dataset(
        path=f'../../data/final_data_v{version}',
        data_files= {game_name: f'{game_name}.csv' for game_name in GAME_NAMES}
    )
    print('Dataset loaded.')
    dataset = dataset.map(preprocess_function)
    for split in dataset.keys():
        dataset[split] = dataset[split].cast(features)
    print('String dates cast to date32.')
    dataset.save_to_disk(f'../../data/EsportsBench{version}')
    print('Dataset saved to disk.')

    repo_id = 'cthorrez/EsportsBenchTest'
    if prod:
        repo_id = 'EsportsBench/EsportsBench'

    dataset.push_to_hub(
        repo_id,
        commit_message=f'update to {version}',
    )
    print('Dataset pushed to hub.')
    if tag is not None:
        huggingface_hub.create_tag(repo_id=repo_id, tag=tag, repo_type="dataset")
        print('Tag applied')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v', type=str, default='3')
    parser.add_argument('--tag', default=None, required=False)
    parser.add_argument('--prod', action='store_true')
    main(**vars(parser.parse_args()))