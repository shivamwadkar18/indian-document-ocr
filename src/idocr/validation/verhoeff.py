"""Verhoeff checksum algorithm implementation.

The Verhoeff algorithm is a dihedral group (D_5) checksum designed by
Jacobus Verhoeff in 1969. It detects all single-digit errors and all
transposition errors between adjacent digits.

In the Indian identity ecosystem, the 12-digit Aadhaar number uses the
Verhoeff algorithm for its final check digit.

NOTE: Validating an Aadhaar number using the Verhoeff checksum algorithm
confirms mathematical consistency only. It does not verify that the number
belongs to a real individual or that an Aadhaar card is authentic.
"""

from __future__ import annotations

# D_5 multiplication table (10 x 10)
_MULTIPLICATION_TABLE: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

# Permutation table (8 x 10)
_PERMUTATION_TABLE: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

# Inverse table in D_5
_INVERSE_TABLE: tuple[int, ...] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def compute_verhoeff_check_digit(number_str: str) -> int:
    """Compute the Verhoeff check digit for a string of decimal digits.

    Args:
        number_str: String containing only decimal digits (e.g. 11 digits for Aadhaar).

    Returns:
        Integer check digit (0-9).

    Raises:
        ValueError: If number_str is empty or contains non-digit characters.
    """
    if not number_str or not number_str.isdigit():
        raise ValueError(f"number_str must contain only decimal digits, got {number_str!r}")

    c = 0
    digits = [int(d) for d in number_str]
    for i, digit in enumerate(reversed(digits), start=1):
        c = _MULTIPLICATION_TABLE[c][_PERMUTATION_TABLE[i % 8][digit]]
    return _INVERSE_TABLE[c]


def validate_verhoeff(number_str: str) -> bool:
    """Validate a numeric string using the Verhoeff checksum algorithm.

    The check digit is assumed to be the final (rightmost) digit of the string.

    Args:
        number_str: Numeric string including the check digit.

    Returns:
        True if the checksum is valid, False otherwise.
    """
    if not number_str or not number_str.isdigit():
        return False

    c = 0
    digits = [int(d) for d in number_str]
    for i, digit in enumerate(reversed(digits)):
        c = _MULTIPLICATION_TABLE[c][_PERMUTATION_TABLE[i % 8][digit]]
    return c == 0
