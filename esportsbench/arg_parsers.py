"""argparsers that can be reused across different files"""
import argparse
from esportsbench.constants import GAME_SHORT_NAMES


def comma_separated(choices):
    """util for parsing comma separated args from a list of choices"""

    def parse_comma_separated(arg_value):
        values = list(map(str.strip, arg_value.split(',')))
        if len(values) == 1 and values[0] == 'all':
            return choices
        if len(values) == 1 and values[0] == 'riix':
            return [choice for choice in choices if choice.startswith('riix_')]
        for value in values:
            if value not in choices:
                raise argparse.ArgumentTypeError(f"'{value}' is not a valid choice. Choose from {choices}.")
        return values

    return parse_comma_separated


def get_games_argparser():
    game_choices = GAME_SHORT_NAMES
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-g', '--games', type=comma_separated(game_choices), required=False, default=game_choices)
    return parser
