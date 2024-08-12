import sentencepiece as spm
import os
from project_config.data_config import *

tokenizer_path = TOKENIZER_PATH
if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path)
    
spm.SentencePieceTrainer.train(input=SPM_DATA, model_prefix=TOKENIZER_PATH+'/m', vocab_size=32000)
