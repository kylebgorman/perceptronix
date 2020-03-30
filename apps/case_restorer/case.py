"""Helpers for recasing.

This module contains the TokenCase enum (representing the outcome variable)
and associated helper functions for case-folding (i.e., applying a casing
to an arbitrary Unicode string)."""


import enum
import unicodedata

from typing import Dict, List, Optional, Tuple


# Casing features at the the Unicode character level.


@enum.unique
class CharCase(enum.IntEnum):
    """Enum for the three character classes."""

    DC = 0  # All non-L Unicode categories ("don't care").
    LOWER = 1  # Unicode category 'Ll'.
    UPPER = 2  # Unicode category 'Lu'.

    def __str__(self):
        return self.name


class UnknownCharCaseError(ValueError):

    pass


def get_cc(nunichar: str) -> CharCase:
    """Computes CharCase for a Unicode character.

    This function computes the CharCase of a Unicode character.

    Args:
        nunichar: A Unicode character whose casing is to be computed.

    Returns:
      The CharCase for the input character.
    """
    catstr = unicodedata.category(nunichar)
    if catstr == "Ll":
        return CharCase.LOWER
    elif catstr == "Lu":
        return CharCase.UPPER
    else:
        return CharCase.DC


def apply_cc(nunichar: str, cc: CharCase) -> str:
    """Applies CharCase to a Unicode character.

    This function applies a CharCase to a Unicode character. Unless CharCase
    is `DC`, this is insensitive to the casing of the input character.

    Args:
        nunichar: A Unicode character to be cased.
        cc: A CharCase indicating the casing to be applied.

    Returns:
        An appropriately-cased Unicode character.

    Raises:
        UnknownCharCaseError.
    """
    if cc == CharCase.LOWER:
        return nunichar.lower()
    elif cc == CharCase.UPPER:
        return nunichar.upper()
    elif cc == CharCase.DC:
        return nunichar
    else:
        raise UnknownCharCaseError(cc)


# Casing features at the word ("token") level.


@enum.unique
class TokenCase(enum.IntEnum):
    """Enum for the five token classes."""

    DC = 0  # [DC]+
    LOWER = 1  # [Ll] ([Ll] | [DC])*
    UPPER = 2  # [Lu] ([Lu] | [DC])* except where bled by title.
    TITLE = 3  # [Lu] ([Ll] | [DC])*
    MIXED = 4  # All others.

    def __str__(self):
        return self.name


class UnknownTokenCaseError(ValueError):

    pass


# Type definitions for mixed-base patterns.


ObligatoryPattern = List[CharCase]
Pattern = Optional[ObligatoryPattern]
MixedPatternTable = Dict[str, ObligatoryPattern]


def get_tc(nunistr: str) -> Tuple[TokenCase, Pattern]:
    """Computes TokenCase for a Unicode string.

    This function computes the TokenCase of a Unicode character.

    Args:
        nunistr: A Unicode string whose casing is to be computed.

    Returns:
        A list consisting of the TokenCase for the input character, and either
        None (representing "n/a") or a list of CharCase instances representing
        the specifics of a `MIXED` TokenCase pattern.
    """
    if nunistr.islower():
        return (TokenCase.LOWER, None)
    # If title and upper have a fight, title wins. Arguably, "A" is usually
    # titlecase, not uppercase.
    elif nunistr.istitle():
        return (TokenCase.TITLE, None)
    elif nunistr.isupper():
        return (TokenCase.UPPER, None)
    pattern = [get_cc(nunichr) for nunichr in nunistr]
    if all(tc == CharCase.DC for tc in pattern):
        return (TokenCase.DC, None)
    return (TokenCase.MIXED, pattern)


def apply_tc(nunistr: str, tc: TokenCase, pattern: Pattern = None) -> str:
    """Applies TokenCase to a Unicode string.

    This function applies a TokenCase to a Unicode string. Unless TokenCase is
    `DC`, this is insensitive to the casing of the input string.

    Args:
        nunistr: A Unicode string to be cased.
        tc: A TokenCase indicating the casing to be applied.
        pattern: An iterable of CharCase characters representing the specifics
            of the `MIXED` TokenCase, when the `tc` argument is `MIXED`.

    Returns:
        An appropriately-cased Unicode string.

    Raises:
        UnknownTokenCaseError.
    """
    if tc == TokenCase.DC:
        return nunistr
    elif tc == TokenCase.LOWER:
        return nunistr.lower()
    elif tc == TokenCase.UPPER:
        return nunistr.upper()
    elif tc == TokenCase.TITLE:
        return nunistr.title()
    elif tc == TokenCase.MIXED:
        # Defaults to lowercase if no pattern is provided.
        if pattern is None:
            return nunistr.lower()
        assert pattern
        assert len(nunistr) == len(pattern)
        return "".join(apply_cc(ch, cc) for (ch, cc) in zip(nunistr, pattern))
    raise UnknownTokenCaseError(tc)
