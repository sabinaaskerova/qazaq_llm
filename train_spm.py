import sentencepiece as spm

spm.SentencePieceTrainer.train(input='kazakh_corpus.txt', model_prefix='m', vocab_size=1000)
