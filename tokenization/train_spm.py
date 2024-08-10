import sentencepiece as spm
import os
from project_config.data_config import *

data_path = DATA_PATH
tokenizer_path = TOKENIZER_PATH
if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path)
    
spm.SentencePieceTrainer.train(input=data_path+'kazakh_corpus.txt', model_prefix=TOKENIZER_PATH+'/m', vocab_size=16000)
