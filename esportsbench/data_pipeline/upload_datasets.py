import huggingface_hub
from datasets import load_dataset

def main():

    # dataset_1 = load_dataset(
    #     'EsportsBench/EsportsBench'
    # )
    # dataset_1.save_to_disk('../../data/EsportsBench1')
    # dataset_1.push_to_hub(
    #     "EsportsBench/EsportsBench"
    # )
    # huggingface_hub.create_tag("EsportsBench/EsportsBench", tag="1.0", repo_type="dataset")

    dataset_1_1 = load_dataset(
        'EsportsBench/EsportsBench',
        revision="1.0"
    )
    print(dataset_1_1['league_of_legends'])

    # dataset_2 = load_dataset(
    #     '../../data/final_data',
    #     verification_mode='no_checks'
    # )
    # # dataset_2.save_to_disk('../../data/EsportsBench2')
    # dataset_2.push_to_hub(
    #     "EsportsBench/EsportsBench"
    # )

    # huggingface_hub.create_tag("EsportsBench/EsportsBench", tag="2.0", repo_type="dataset") 
    
    
    dataset_2_2 = load_dataset(
        'EsportsBench/EsportsBench',
        revision="2.0"
    )
    print(dataset_2_2['league_of_legends'])


if __name__ == '__main__':
    main()