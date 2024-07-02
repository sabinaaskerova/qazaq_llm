import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)
    
    def detokenize(self, tokens):
        return self.sp.DecodePieces(tokens)
    
    def encode(self, text):
        return self.sp.EncodeAsIds(text)
    
    def decode(self, ids):
        return self.sp.DecodeIds(ids)
    

if __name__ == "__main__":
    model_path = "m.model"
    tokenizer = Tokenizer(model_path)

    text = "Алматыда дәрігерлер жүрек тамырлары 100 пайызға жуық тарылып кеткен науқасты аман алып қалды"
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    ids = tokenizer.encode(text)
    print(f"IDs: {ids}")
   