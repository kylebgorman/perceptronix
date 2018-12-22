# Copyright (c) 2015-2016 Kyle Gorman <kylebgorman@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


"""Helpers for recasing.

This module contains the TokenCase enum (representing the outcome variable)
and associated helper functions for case-folding (i.e., applying a casing
to an arbitrary Unicode string).
"""


import enum
import unicodedata


# Casing features at the the Unicode character level.


@enum.unique
class CharCase(enum.IntEnum):
  """Enum for the three character classes."""
  dc = 0     # All non-L Unicode categories ("don't care").
  lower = 1  # Unicode category 'Ll'.
  upper = 2  # Unicode category 'Lu'.


class UnknownCharCaseError(ValueError):
  pass


def get_cc(nunichar):
  """Computes CharCase for a Unicode character.

  This function computes the CharCase of a Unicode character.

  Args:
    nunichar: A Unicode character whose casing is to be computed.

  Returns:
    The CharCase for the input character.
  """
  catstr = unicodedata.category(nunichar)
  if catstr == "Ll":
    return CharCase.lower
  elif catstr == "Lu":
    return CharCase.upper
  else:
    return CharCase.dc


def apply_cc(nunichar, cc):
  """Applies CharCase to a Unicode character.

  This function applies a CharCase to a Unicode character. Unless CharCase
  is `dc`, this is insensitive to the casing of the input character.

  Args:
    nunichar: A Unicode character to be cased.
    cc: A CharCase indicating the casing to be applied.

  Returns:
    An appropriately-cased Unicode character.

  Raises:
    UnknownCharCaseError.
  """
  if cc == CharCase.lower:
    return nunichar.lower()
  elif cc == CharCase.upper:
    return nunichar.upper()
  elif cc == CharCase.dc:
    return nunichar
  else:
    raise UnknownCharCaseError(cc)


# Casing features at the word ("token") level.


@enum.unique
class TokenCase(enum.IntEnum):
  """Enum for the five token classes."""
  dc = 0     # [dc]+
  lower = 1  # [Ll] ([Ll] | [dc])*
  upper = 2  # [Lu] ([Lu] | [dc])* except where bled by title.
  title = 3  # [Lu] ([Ll] | [dc])*
  mixed = 4  # All others.


class UnknownTokenCaseError(ValueError):
  pass


_NA_ = None


def get_tc(nunistr):
  """Computes TokenCase for a Unicode string.

  This function computes the TokenCase of a Unicode character.

  Args:
    nunistr: A Unicode string whose casing is to be computed.

  Returns:
    A tuple consisting of the TokenCase for the input character, and either None
    (representing "n/a") or an iterable of CharCase instances representing the
    specifics of a `mixed` TokenCase pattern.
  """
  if nunistr.islower():
    return (TokenCase.lower, _NA_)
  # If title and upper have a fight, title wins. Arguably, "A" is usually
  # titlecase, not uppercase, and that's the most frequent case of all.
  elif nunistr.istitle():
    return (TokenCase.title, _NA_)
  elif nunistr.isupper():
    return (TokenCase.upper, _NA_)
  pattern = tuple(get_cc(nunichr) for nunichr in nunistr)
  if all(tc == CharCase.dc for tc in pattern):
    return (TokenCase.dc, _NA_)
  else:
    return (TokenCase.mixed, pattern)


def apply_tc(nunistr, tc, pattern=_NA_):
  """Applies TokenCase to a Unicode string.

  This function applies a TokenCase to a Unicode string. Unless TokenCase is
  `dc`, this is insensitive to the casing of the input string.

  Args:
    nunistr: A Unicode string to be cased.
    tc: A Tokencase indicating the casing to be applied.
    pattern: An iterable of CharCase characters representing the specifics of
        the `mixed` TokenCase, when the `tc` argument is `mixed`.

  Returns:
    An appropriately-cased Unicode string.

  Raises:
    UnknownTokenCaseError.
  """
  if tc == TokenCase.dc:
    return nunistr
  elif tc == TokenCase.lower:
    return nunistr.lower()
  elif tc == TokenCase.upper:
    return nunistr.upper()
  elif tc == TokenCase.title:
    return nunistr.title()
  elif tc == TokenCase.mixed:
    assert pattern != _NA_
    assert len(nunistr) == len(pattern)
    return "".join(apply_cc(ch, cc) for (ch, cc) in zip(nunistr, pattern))
  else:
    raise UnknownTokenCaseError(tc)
