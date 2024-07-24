import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)
    
    def detokenize(self, tokens):
        return self.sp.DecodePieces(tokens)
    
    def token_to_id(self, token):
        return self.sp.PieceToId(token)
    
    def id_to_token(self, id):
        return self.sp.IdToPiece(id)
    
    def encode(self, text):
        return self.sp.EncodeAsIds(text)
    
    def decode(self, ids):
        return self.sp.DecodeIds(ids)
    

if __name__ == "__main__":
    tokenizer_path = "tokenizer/"
    model_path = tokenizer_path+"m.model"
    tokenizer = Tokenizer(model_path)

    text = "Алматыда дәрігерлер жүрек тамырлары 100 пайызға жуық тарылып кеткен науқасты аман алып қалды"
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    ids = tokenizer.encode(text)
    print(f"IDs: {ids}")
   