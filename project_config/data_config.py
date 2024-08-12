import os 

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
DATA_PATH = ROOT_PATH + 'plain_text_datasets/'
TOKENIZER_PATH = ROOT_PATH + 'tokenizer_model/'
INSTRUCTION_DATA_PATH = ROOT_PATH + 'instruction_datasets/'
MODEL_STATES_PATH = ROOT_PATH + 'model_states/'
LOGS_PATH = ROOT_PATH + 'logs/'
SPM_DATA = DATA_PATH + 'data_spm.txt'
