import sentencepiece as spm
import os

data_path = "data/"
tokenizer_path = "tokenizer/"
if not os.path.exists(tokenizer_path):
    os.makedirs(tokenizer_path)
    
spm.SentencePieceTrainer.train(input=data_path+'kazakh_corpus.txt', model_prefix='tokenizer/m', vocab_size=1000)
