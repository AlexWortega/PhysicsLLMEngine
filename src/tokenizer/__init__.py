"""Physics tokenizer package for digit-level numerical encoding.

This tokenizer preserves numerical semantics destroyed by standard BPE by
encoding numbers digit-by-digit, ensuring consistent token boundaries.
"""

from src.tokenizer.decoder import decode_number
from src.tokenizer.encoder import encode_number, encode_text
from src.tokenizer.vocabulary import PhysicsVocabulary

__all__ = ["PhysicsVocabulary", "encode_number", "encode_text", "decode_number"]
