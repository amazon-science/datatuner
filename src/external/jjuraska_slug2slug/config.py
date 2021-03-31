import os


# Directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EVAL_DIR = os.path.join(ROOT_DIR, 'eval')
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
PREDICTIONS_BATCH_DIR = os.path.join(PREDICTIONS_DIR, 'batch')
PREDICTIONS_BATCH_LEX_DIR = os.path.join(PREDICTIONS_DIR, 'batch_lex')
PREDICTIONS_BATCH_EVENT_DIR = os.path.join(PREDICTIONS_DIR, 'batch_event')
SLOT_ALIGNER_DIR = os.path.join(ROOT_DIR, 'slot_aligner')
SLOT_ALIGNER_ALTERNATIVES = os.path.join(SLOT_ALIGNER_DIR, 'alignment', 'alternatives.json')
T2T_DIR = os.path.join(ROOT_DIR, 't2t')
TOOLS_DIR = os.path.join(ROOT_DIR, 'tools')
TTEST_DIR = os.path.join(ROOT_DIR, 'ttest')
TTEST_DATA_DIR = os.path.join(ROOT_DIR, 'ttest', 'data')
TTEST_SCORES_DIR = os.path.join(ROOT_DIR, 'ttest', 'scores')

# Dataset paths
E2E_DATA_DIR = os.path.join(DATA_DIR, 'rest_e2e')
TV_DATA_DIR = os.path.join(DATA_DIR, 'tv')
LAPTOP_DATA_DIR = os.path.join(DATA_DIR, 'laptop')
HOTEL_DATA_DIR = os.path.join(DATA_DIR, 'hotel')
VIDEO_GAME_DATA_DIR = os.path.join(DATA_DIR, 'video_game')

# Script paths
METRICS_SCRIPT_PATH = os.path.join(METRICS_DIR, 'measure_scores.py')

# Constants
COMMA_PLACEHOLDER = ' __comma__'
DELEX_PREFIX = '__slot_'    # Important to use special symbols that do not get tokenized (such as '_')
DELEX_SUFFIX = '__'
EMPH_TOKEN = '__emph__'
CONTRAST_TOKEN = '__contrast__'
CONCESSION_TOKEN = '__concession__'
