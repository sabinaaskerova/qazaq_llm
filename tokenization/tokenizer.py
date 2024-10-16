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
    
    def encode(self, text): # returns a list of ids
        return self.sp.EncodeAsIds(text)
    
    def decode(self, ids):
        return self.sp.DecodeIds(ids)
