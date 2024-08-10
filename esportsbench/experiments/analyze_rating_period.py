import matplotlib.pyplot as plt
from esportsbench.arg_parsers import get_games_argparser, comma_separated
from esportsbench.eval.bench import run_benchmark, ALL_RATING_SYSTEMS

def main(
    games,
    rating_systems,
    train_end_date,
    test_end_date,
    data_dir,
    hyperparameter_config,
    drop_draws=None
):
    rating_periods = [1, 7, 14, 28]
    sweep_results_base = 'conf/sweep_results/fine_sweep'

    for rating_period in rating_periods:
        config_path = f"{sweep_results_base}_{rating_period}D_1000"
        print(f'getting test set results for {config_path}')
        results = run_benchmark(
            games=games,
            rating_systems=rating_systems,
            rating_period=rating_period,
            train_end_date=train_end_date,
            test_end_date=test_end_date,
            data_dir=data_dir,
            drop_draws=drop_draws,
            hyperparameter_config=config_path
        )
        print(results)




if __name__ == '__main__':
    parser = get_games_argparser()
    parser.add_argument(
        '-rs',
        '--rating_systems',
        type=comma_separated(ALL_RATING_SYSTEMS),
        default=ALL_RATING_SYSTEMS,
    )
    parser.add_argument('-dd', '--drop_draws', action='store_true')
    parser.add_argument('--train_end_date', type=str, default='2023-03-31', help='inclusive end date for test set')
    parser.add_argument('--test_end_date', type=str, default='2024-03-31', help='inclusive end date for test set')
    parser.add_argument('-d', '--data_dir', type=str, default='hf_data')
    parser.add_argument('-c', '--hyperparameter_config', type=str, required=False, default='default')
    args = parser.parse_args()
    main(**vars(args))