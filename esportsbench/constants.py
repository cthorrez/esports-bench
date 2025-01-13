"""constants and configs for use in other scripts"""
from functools import partial
from riix.models.elo import Elo
from riix.models.glicko import Glicko
from riix.models.glicko2 import Glicko2
from riix.models.trueskill import TrueSkill
from riix.models.weng_lin import WengLin
from riix.models.melo import Melo
from riix.models.gen_elo import GenElo
from riix.models.constant_variance_glicko import ConstantVarianceGlicko
from riix.models.velo import vElo
from riix.models.online_disc_decomp import OnlineDiscDecomp
from riix.models.online_rao_kupper import OnlineRaoKupper
from riix.models.elo_davidson import EloDavidson
from riix.models.skf import VSKF
from riix.models.elomentum import EloMentum
from riix.models.yuksel_2024 import Yuksel2024
from riix.models.autograd_rating_system import AutogradRatingSystem
from riix.models.baselines import BaselineRatingSystem

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
    'eafc': 'ea_sports_fc',
}

GAME_NAMES = list(GAME_NAME_MAP.values())
GAME_SHORT_NAMES = list(GAME_NAME_MAP.keys())

OFFICIAL_NAME_MAP = {
    'starcraft1': 'Starcraft I',
    'starcraft2': 'Starcraft II',
    'warcraft3': 'WarCraft III',
    'smash_melee': 'Super Smash Bros. Melee',
    'smash_ultimate': 'Super Smash Bros. Ultimate',
    'tekken': 'Tekken',
    'street_fighter': 'Street Fighter',
    'king_of_fighters': 'King of Fighters',
    'guilty_gear': 'Guilty Gear',
    'tetris': 'Tetris',
    'ea_sports_fc': 'EA Sports FC', 
    'rocket_league': 'Rocket League',
    'dota2': 'Dota 2',
    'league_of_legends': 'League of Legends',
    'counterstrike': 'Counter-Strike',
    'call_of_duty': 'Call of Duty',
    'halo': 'Halo',
    'overwatch': 'Overwatch',
    'valorant': 'VALORANT',
    'rainbow_six': 'Rainbow Six: Siege',
}

RATING_SYSTEM_NAME_CLASS_MAP = {
    'elo': Elo,
    'glicko': Glicko,
    'glicko2': Glicko2,
    'trueskill': TrueSkill,
    'wl_bt': partial(WengLin, model='bt', tau=0.0),
    'wl_tm': partial(WengLin, model='tm', tau=0.0),
    'melo': Melo,
    'genelo': GenElo,
    # 'cvglicko': ConstantVarianceGlicko,
    'velo': vElo,
    # 'im': IterativeMarkov, # these 2 are so bad it's not even worth comparing in most experiments
    # 'tm': TemporalMassey,
    'vskf_bt': partial(VSKF, model='bt'),
    'vskf_tm': partial(VSKF, model='tm'),
    # 'odd': OnlineDiscDecomp,
    # 'ork': OnlineRaoKupper,
    # 'elod': EloDavidson,
    # 'elom': EloMentum,
    'yuksel': Yuksel2024,
    # 'autograd' : AutogradRatingSystem
    'random_base' : partial(BaselineRatingSystem, mode='random'),
    'wr_base' : partial(BaselineRatingSystem, mode='win_rate'),
    'win_base' : partial(BaselineRatingSystem, mode='wins'),
    'appearance_base' : partial(BaselineRatingSystem, mode='appearances'),
}
ALL_RATING_SYSTEM_NAMES = list(RATING_SYSTEM_NAME_CLASS_MAP.keys())
