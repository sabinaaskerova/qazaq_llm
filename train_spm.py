import sentencepiece as spm

data_path = "data/"
spm.SentencePieceTrainer.train(input=data_path+'kazakh_corpus.txt', model_prefix='tokenizer/m', vocab_size=1000)
