"""constants and configs for use in other scripts"""

GAME_NAME_MAP = {
    'lol': 'league_of_legends',
    'cs': 'counterstrike',
    'rl': 'rocket_league',
    'sc1': 'starcraft1',
    'sc2': 'starcraft2',
    'ssbm': 'smash_melee',
    'ssbu': 'smash_ultimate',
    'dota2': 'dota2',
    'ow': 'overwatch',
    'val': 'valorant',
    'wc3': 'warcraft3',
    'r6': 'rainbow_six',
    'halo': 'halo',
    'cod': 'call_of_duty',
    'tetris': 'tetris',
    'sf': 'street_fighter',
    'tek': 'tekken',
    'kof': 'king_of_fighters',
    'gg': 'guilty_gear',
}

GAME_NAMES = list(GAME_NAME_MAP.values())
GAME_SHORT_NAMES = list(GAME_NAME_MAP.keys())
