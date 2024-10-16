import sentencepiece as spm
import torch
import torch.nn as nn
from typing import List, Dict, Union
from project_config.data_config import *

class SpecialTokenizerWrapper:
    def __init__(self, spm_model_path: str, special_tokens: List[str]):
        """
        Initialize wrapper with SentencePiece model and special tokens.
        
        Args:
            spm_model_path: Path to the SentencePiece model
            special_tokens: List of special tokens to add
        """
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(spm_model_path)
        
        # Store original vocabulary size
        self.original_vocab_size = self.sp_model.GetPieceSize()
        
        # Create special token mappings
        self.special_tokens = special_tokens
        self.special_token_to_id: Dict[str, int] = {}
        self.id_to_special_token: Dict[int, str] = {}
        
        # Assign IDs to special tokens starting from original vocab size
        for i, token in enumerate(special_tokens):
            token_id = self.original_vocab_size + i
            self.special_token_to_id[token] = token_id
            self.id_to_special_token[token_id] = token
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text with special token handling.
        """
        tokens = []
        current_pos = 0
        
        while current_pos < len(text):
            # First check for special tokens
            special_token_found = False
            for token in self.special_tokens:
                if text.startswith(token, current_pos):
                    tokens.append(self.special_token_to_id[token])
                    current_pos += len(token)
                    special_token_found = True
                    break
            
            if not special_token_found:
                # Find the next special token position
                next_special_pos = float('inf')
                for token in self.special_tokens:
                    pos = text.find(token, current_pos)
                    if pos != -1:
                        next_special_pos = min(next_special_pos, pos)
                
                # Get the text until the next special token
                if next_special_pos == float('inf'):
                    segment = text[current_pos:]
                    current_pos = len(text)
                else:
                    segment = text[current_pos:next_special_pos]
                    current_pos = next_special_pos
                
                if segment:
                    tokens.extend(self.sp_model.EncodeAsIds(segment))
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode IDs back to text, handling special tokens.
        """
        result = []
        regular_ids = []
        
        for id_ in ids:
            if id_ in self.id_to_special_token:
                if regular_ids:
                    result.append(self.sp_model.DecodeIds(regular_ids))
                    regular_ids = []
                result.append(self.id_to_special_token[id_])
            else:
                regular_ids.append(id_)
        
        if regular_ids:
            result.append(self.sp_model.DecodeIds(regular_ids))
        
        return ''.join(result)
    
    def get_vocab_size(self) -> int:
        return self.original_vocab_size + len(self.special_tokens)

