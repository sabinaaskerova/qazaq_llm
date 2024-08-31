import os 

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
DATA_PATH = ROOT_PATH + 'plain_text_datasets/'
TOKENIZER_PATH = ROOT_PATH + 'tokenizer_model/'
INSTRUCTION_DATA_PATH = ROOT_PATH + 'instruction_datasets/'
MODEL_STATES_PATH = ROOT_PATH + 'model_states/'
LOGS_PATH = ROOT_PATH + 'logs/'
SPM_DATA = DATA_PATH + 'data_spm.txt'
GENERATED_TEXT_TEST = ROOT_PATH + 'generated_text_test.txt'
COLAB_PATH = '/content/drive/MyDrive/QazaqLLM/'

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
if not os.path.exists(INSTRUCTION_DATA_PATH):
    os.makedirs(INSTRUCTION_DATA_PATH)
if not os.path.exists(TOKENIZER_PATH):
    os.makedirs(TOKENIZER_PATH)
if not os.path.exists(MODEL_STATES_PATH):
    os.makedirs(MODEL_STATES_PATH)
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
if not os.path.exists(COLAB_PATH):
    os.makedirs(COLAB_PATH)