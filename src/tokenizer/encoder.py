"""Encoder for converting numbers to digit-level tokens.

Uses consistent formatting with trailing zero stripping to ensure
round-trip guarantee: encode(decode(encode(x))) == encode(x)
"""

from typing import List

from src.tokenizer.vocabulary import PhysicsVocabulary


def encode_number(
    num: float, vocab: PhysicsVocabulary, precision: int = 6
) -> List[int]:
    """Encode a number as a sequence of digit-level tokens.

    Numbers are formatted with consistent precision, then stripped of
    trailing zeros to ensure round-trip consistency.

    Args:
        num: The number to encode
        vocab: Vocabulary for token lookup
        precision: Decimal places for formatting (default: 6)

    Returns:
        List of token IDs: [NUM_ID, digit_ids..., NUM_ID]

    Examples:
        >>> v = PhysicsVocabulary()
        >>> encode_number(123.45, v)  # [NUM, 1, 2, 3, ., 4, 5, NUM]
        >>> encode_number(-5, v)      # [NUM, -, 5, NUM]
        >>> encode_number(0, v)       # [NUM, 0, NUM]
    """
    # Format with consistent precision, strip trailing zeros and decimal
    formatted = f"{num:.{precision}f}".rstrip("0").rstrip(".")

    # Build token sequence with NUM boundaries
    tokens = [vocab.NUM_ID]

    for char in formatted:
        token_id = vocab.token_to_id(char)
        tokens.append(token_id)

    tokens.append(vocab.NUM_ID)

    return tokens


def encode_text(text: str, vocab: PhysicsVocabulary) -> List[int]:
    """Encode non-numerical text as UNK tokens.

    This is a placeholder for future expansion to handle text descriptions.
    Currently returns a single UNK token for any text input.

    Args:
        text: Text string to encode
        vocab: Vocabulary for token lookup

    Returns:
        List containing UNK token IDs (one per character)
    """
    return [vocab.UNK_ID for _ in text]
