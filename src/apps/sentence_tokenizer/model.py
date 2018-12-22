# Copyright (c) 2015-2018 Kyle Gorman <kylebgorman@gmail.com>
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


import collections

from typing import Iterator

import regex
import perceptronix


NEWLINE = regex.compile("\n|\r|\r\n")

# Bias feature string.
BIAS = "*bias*"


# For other (more important) defaults, see __main__.py in this directory.


Candidate = collections.namedtuple(
    "Candidate", ("left_index", "right_index", "left", "boundary", "right")
)


class SentenceTokenizer(object):
    def __init__(
        self, candidate_regex: str, max_context: int, *args, **kwargs
    ):
        self._candidate_regex = regex.compile(candidate_regex)
        self._max_context = max_context
        self._classifier = perceptronix.SparseBinomialClassifier(
            *args, **kwargs
        )

    @classmethod
    def read(cls, filename: str, candidate_regex: str, max_context: int):
        """Reads sentence tokenizer model from serialized model file."""
        result = cls.__new__(cls)
        result._candidate_regex = regex.compile(candidate_regex)
        result._max_context = max_context
        result._classifier = perceptronix.SparseBinomialClassifier.read(
            filename
        )
        return result

    def candidates(self, text: str) -> Iterator[Candidate]:
        """Generates candidate sentence boundary string tuples from string."""
        for match in self._candidate_regex.finditer(text, overlapped=True):
            (left, boundary, right) = match.groups()
            left_index = match.span()[0] + len(left) + 1
            right_index = left_index + len(boundary)
            left_bound = min(len(left), self._max_context)
            right_bound = min(len(right), self._max_context)
            yield Candidate(
                left_index,
                right_index,
                left[-left_bound:],
                boundary,
                right[:right_bound],
            )

    def candidates_from_file(self, filename: str) -> Iterator[Candidate]:
        """Generates candidate sentence boundary string tuples from file."""
        with open(filename, "r") as source:
            text = source.read()
        return self.candidates(text)

    @staticmethod
    def extract_features(candidate: Candidate) -> Iterator[str]:
        """Generates feature vector for a candidate."""
        yield BIAS
        # All suffixes of the left context.
        lpieces = tuple(
            f"L={candidate.left[-i:]}"
            for i in range(1, 1 + len(candidate.left))
        )
        yield from lpieces
        # All prefixes of the right context.
        rpieces = tuple(
            f"R={candidate.right[:i]}"
            for i in range(1, 1 + len(candidate.right))
        )
        yield from rpieces
        # Composition of the two.
        yield from (
            f"{lpiece}^{rpiece}" for (lpiece, rpiece) in zip(lpieces, rpieces)
        )

    def tokenize(self, text: str) -> Iterator[str]:
        """Generates segmented strings of text."""
        start = 0
        for candidate in self.candidates(text):
            # Passes through any whitespace already present.
            if NEWLINE.match(candidate.boundary):
                continue
            if self.predict(SentenceTokenizer.extract_features(candidate)):
                yield text[start : candidate.left_index]
                start = candidate.right_index
        yield text[start:].rstrip()

    # Delegates all attributes not otherwise defined to the underlying
    # classifier.
    def __getattr__(self, name):
        return getattr(self._classifier, name)
