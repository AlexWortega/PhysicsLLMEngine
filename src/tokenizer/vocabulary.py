"""Physics vocabulary for digit-level tokenization.

Token ordering follows research recommendations:
- Special tokens first (indices 0-4): PAD, UNK, BOS, EOS, NUM
- Digit tokens (indices 5-14): 0-9
- Number-related tokens (indices 15-18): decimal, minus, e, plus
"""

from typing import Dict


class PhysicsVocabulary:
    """Vocabulary for physics numerical data tokenization.

    Provides bidirectional mapping between tokens and integer IDs.
    Designed for digit-level number encoding to preserve numerical semantics.
    """

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    NUM_TOKEN = "<NUM>"

    # Token IDs (fixed ordering per research)
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3
    NUM_ID = 4

    def __init__(self) -> None:
        """Initialize vocabulary with all required tokens."""
        # Build token to ID mapping
        self._token_to_id: Dict[str, int] = {
            # Special tokens (0-4)
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
            self.NUM_TOKEN: 4,
            # Digit tokens (5-14)
            "0": 5,
            "1": 6,
            "2": 7,
            "3": 8,
            "4": 9,
            "5": 10,
            "6": 11,
            "7": 12,
            "8": 13,
            "9": 14,
            # Number-related tokens (15-18)
            ".": 15,
            "-": 16,
            "e": 17,
            "+": 18,
        }

        # Build reverse mapping
        self._id_to_token: Dict[int, str] = {
            id_: token for token, id_ in self._token_to_id.items()
        }

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self._token_to_id)

    def token_to_id(self, token: str) -> int:
        """Convert token to its integer ID.

        Args:
            token: Token string to convert

        Returns:
            Integer ID for the token, or UNK_ID if unknown
        """
        return self._token_to_id.get(token, self.UNK_ID)

    def id_to_token(self, id_: int) -> str:
        """Convert integer ID to its token string.

        Args:
            id_: Integer ID to convert

        Returns:
            Token string, or UNK_TOKEN if ID is out of range
        """
        return self._id_to_token.get(id_, self.UNK_TOKEN)

    def __contains__(self, token: str) -> bool:
        """Check if token is in vocabulary."""
        return token in self._token_to_id

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
