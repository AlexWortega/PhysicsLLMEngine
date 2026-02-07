"""Decoder for converting digit-level tokens back to numbers.

Handles reconstruction of numerical values from token sequences,
with proper handling of edge cases and invalid input.
"""

from typing import List

from src.tokenizer.vocabulary import PhysicsVocabulary


def decode_number(tokens: List[int], vocab: PhysicsVocabulary) -> float:
    """Decode a sequence of digit-level tokens back to a number.

    Skips NUM boundary tokens and reconstructs the number from
    the digit tokens in between.

    Args:
        tokens: List of token IDs from encode_number
        vocab: Vocabulary for inverse lookup

    Returns:
        The decoded float value

    Raises:
        ValueError: If tokens list is empty or contains only boundaries

    Examples:
        >>> v = PhysicsVocabulary()
        >>> decode_number([4, 6, 7, 8, 15, 9, 10, 4], v)  # NUM,1,2,3,.,4,5,NUM
        123.45
    """
    if not tokens:
        raise ValueError("Cannot decode empty token list")

    # Extract digit characters, skipping NUM boundary tokens
    chars = []
    for token_id in tokens:
        if token_id == vocab.NUM_ID:
            continue
        char = vocab.id_to_token(token_id)
        # Skip special tokens that shouldn't appear in numbers
        if char in (vocab.PAD_TOKEN, vocab.UNK_TOKEN, vocab.BOS_TOKEN, vocab.EOS_TOKEN):
            continue
        chars.append(char)

    if not chars:
        raise ValueError("No valid digit tokens found")

    # Reconstruct number string and parse
    num_str = "".join(chars)

    try:
        return float(num_str)
    except ValueError as e:
        raise ValueError(f"Cannot parse '{num_str}' as float: {e}")
