"""Custom physics tokenizer for nanochat compatibility.

Wraps the Phase 1 digit-level tokenizer to provide a unified interface
compatible with nanochat training infrastructure. Extends vocabulary
with text tokens needed for scene descriptions.
"""

import re
from typing import List, Dict, Optional

from src.tokenizer import PhysicsVocabulary, encode_number, decode_number


# Extended vocabulary for text tokens (scene descriptions)
# Start after the base vocabulary (19 tokens: 0-18)
TEXT_TOKENS = {
    # Scene structure
    "Scene": 19,
    "Frame": 20,
    "Predict": 21,
    "next": 22,
    "frame": 23,
    # Physics terms
    "pos": 24,
    "vel": 25,
    "Gravity": 26,
    "Timestep": 27,
    "obj_": 28,
    # Punctuation and delimiters
    "(": 29,
    ")": 30,
    ":": 31,
    ",": 32,
    "=": 33,
    " ": 34,
    "\n": 35,
    # Additional physics description tokens
    "circle": 36,
    "rectangle": 37,
    "mass": 38,
    "radius": 39,
    "width": 40,
    "height": 41,
    "elasticity": 42,
    "friction": 43,
    # Object count identifiers (0-9 already in digits, but obj_10, obj_20, etc.)
    "10": 44,
    "11": 45,
    "12": 46,
    "13": 47,
    "14": 48,
    "15": 49,
    "16": 50,
    "17": 51,
    "18": 52,
    "19": 53,
    "20": 54,
    "21": 55,
    "22": 56,
    "23": 57,
    "24": 58,
    "25": 59,
    "26": 60,
    "27": 61,
    "28": 62,
    "29": 63,
    "30": 64,
    "31": 65,
    "32": 66,
    "33": 67,
    "34": 68,
    "35": 69,
    "36": 70,
    "37": 71,
    "38": 72,
    "39": 73,
    "40": 74,
    "41": 75,
    "42": 76,
    "43": 77,
    "44": 78,
    "45": 79,
    "46": 80,
    "47": 81,
    "48": 82,
    "49": 83,
    # Scene description fragments
    "with": 84,
    "objects": 85,
    "bouncing": 86,
    "falling": 87,
    "colliding": 88,
    "in": 89,
    "a": 90,
    "box": 91,
    "and": 92,
    "the": 93,
}

# Reverse mapping for decoding
TEXT_ID_TO_TOKEN = {v: k for k, v in TEXT_TOKENS.items()}


class PhysicsTokenizer:
    """Tokenizer for physics scene data compatible with nanochat.

    Provides encode/decode interface expected by nanochat training infrastructure
    while using digit-level number encoding from Phase 1 tokenizer.
    """

    def __init__(self) -> None:
        """Initialize tokenizer with PhysicsVocabulary and extended text tokens."""
        self._vocab = PhysicsVocabulary()
        self._text_tokens = TEXT_TOKENS
        self._text_id_to_token = TEXT_ID_TO_TOKEN

        # Total vocab size: base (19) + text tokens
        self._vocab_size = 19 + len(TEXT_TOKENS)

        # Compile regex for number detection
        # Matches integers, decimals, negative numbers, and scientific notation
        self._number_pattern = re.compile(r"-?\d+\.?\d*(?:e[+-]?\d+)?")

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        """Return PAD token ID."""
        return self._vocab.PAD_ID

    @property
    def bos_token_id(self) -> int:
        """Return BOS (beginning of sequence) token ID."""
        return self._vocab.BOS_ID

    @property
    def eos_token_id(self) -> int:
        """Return EOS (end of sequence) token ID."""
        return self._vocab.EOS_ID

    @property
    def unk_token_id(self) -> int:
        """Return UNK (unknown) token ID."""
        return self._vocab.UNK_ID

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with digit-level number encoding.

        Numbers are encoded digit-by-digit with NUM boundary tokens.
        Text tokens use the extended vocabulary.

        Args:
            text: Input text to tokenize

        Returns:
            List of token IDs
        """
        tokens = []
        pos = 0

        while pos < len(text):
            # Try to match a number at current position
            match = self._number_pattern.match(text, pos)
            if match:
                num_str = match.group()
                num_value = float(num_str)
                # Encode number digit-by-digit with NUM boundaries
                num_tokens = encode_number(num_value, self._vocab)
                tokens.extend(num_tokens)
                pos = match.end()
                continue

            # Try to match text tokens (longest match first)
            matched = False
            for token, token_id in sorted(
                self._text_tokens.items(), key=lambda x: -len(x[0])
            ):
                if text[pos:].startswith(token):
                    tokens.append(token_id)
                    pos += len(token)
                    matched = True
                    break

            if not matched:
                # Character not in vocabulary - use UNK
                tokens.append(self._vocab.UNK_ID)
                pos += 1

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Handles both base vocabulary tokens (including numbers) and
        extended text tokens.

        Args:
            token_ids: List of token IDs

        Returns:
            Reconstructed text string
        """
        result = []
        i = 0

        while i < len(token_ids):
            token_id = token_ids[i]

            # Check if this is a NUM boundary token (start of a number)
            if token_id == self._vocab.NUM_ID:
                # Find the closing NUM token
                j = i + 1
                while j < len(token_ids) and token_ids[j] != self._vocab.NUM_ID:
                    j += 1

                # Extract number tokens (including boundaries for decode_number)
                if j < len(token_ids):
                    num_tokens = token_ids[i : j + 1]
                    try:
                        num_value = decode_number(num_tokens, self._vocab)
                        # Format with precision stripping like encoder
                        formatted = f"{num_value:.6f}".rstrip("0").rstrip(".")
                        result.append(formatted)
                    except ValueError:
                        # Fallback to UNK representation
                        result.append("<UNK>")
                    i = j + 1
                else:
                    # Unclosed NUM - skip
                    i += 1
                continue

            # Check if it's a text token
            if token_id in self._text_id_to_token:
                result.append(self._text_id_to_token[token_id])
                i += 1
                continue

            # Check base vocabulary (excluding NUM which we handled above)
            if 0 <= token_id < 19:
                # Special tokens
                if token_id == self._vocab.PAD_ID:
                    pass  # Skip PAD
                elif token_id == self._vocab.UNK_ID:
                    result.append("<UNK>")
                elif token_id == self._vocab.BOS_ID:
                    pass  # Skip BOS
                elif token_id == self._vocab.EOS_ID:
                    pass  # Skip EOS
                else:
                    # Single character token (digit or number-related)
                    char = self._vocab.id_to_token(token_id)
                    result.append(char)
                i += 1
                continue

            # Unknown token ID
            result.append("<UNK>")
            i += 1

        return "".join(result)

    def get_vocab(self) -> Dict[str, int]:
        """Return complete vocabulary mapping.

        Returns:
            Dict mapping token strings to IDs
        """
        vocab = {}

        # Add base vocabulary
        for token_id in range(19):
            token = self._vocab.id_to_token(token_id)
            vocab[token] = token_id

        # Add text tokens
        vocab.update(self._text_tokens)

        return vocab


def get_physics_vocab() -> Dict[str, int]:
    """Return the complete physics vocabulary mapping.

    Convenience function for accessing vocabulary without creating tokenizer.

    Returns:
        Dict mapping token strings to IDs
    """
    tokenizer = PhysicsTokenizer()
    return tokenizer.get_vocab()
