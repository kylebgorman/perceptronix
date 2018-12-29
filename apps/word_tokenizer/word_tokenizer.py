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

import collections
import unicodedata

import nlup
import regex
import perceptronix

# Default encoding and normalization scheme.
ENCODING = "utf-8"
NORMALIZATION = "NFKC"
NEWLINE = re.compile(b"[\n\r]")  # That'll do for now.

# How far we look to the left of the boundary when trying to match the
# left context regex.
LEFT_LIMIT = 32

# Bias feature string.
BIAS = b"*bias*"


# For other (more important) defaults, see __main__.py in this directory.


Candidate = collections.namedtuple(
    "Candidate", ("left_index", "right_index", "left", "boundary", "right")
)


class SentenceTokenizer(object):

    __slots__ = (
        "_left_regex",
        "_boundary_regex",
        "_right_regex",
        "_max_context",
        "_classifier",
    )

    def __init__(self, candidate_regex, max_context, *args, **kwargs):
        self._candidate_regex = re.compile(candidate_regex)
        self._max_context = max_context
        self._classifier = perceptronix.SparseBinomialClassifier(
            *args, **kwargs
        )

    def candidates(self, text):
        """Generates candidate sentence boundary string tuples."""
        for match in self._candidate_regex.finditer(text, overlapped=True):
            (left, boundary, right) = match.groups()
            left_index = match.span()[0] + len(left)
            right_index = left_index + len(boundary)
            left_bound = min(len(left), self._max_context)
            right_bound = min(len(right), self._max_context)
            yield Candidate(
                left_index,
                right_index,
                SentenceTokenizer.tobytes(left[-left_bound:]),
                SentenceTokenizer.tobytes(bmatch.group(1)),
                SentenceTokenizer.tobytes(right[:right_bound]),
            )

    @staticmethod
    def tobytes(string):
        return unicodedata.normalize(NORMALIZATION, string).encode(ENCODING)

    @staticmethod
    @nlup.tupleify
    def extract_features(candidate):
        """Generates feature vector for a candidate."""
        yield BIAS
        # All suffixes of the left context.
        lpieces = tuple(
            b"L=" + candidate.left[-i:]
            for i in range(1, 1 + len(candidate.left))
        )
        yield from lpieces
        rpieces = tuple(
            b"R=" + candidate.right[:i]
            for i in range(1, 1 + len(candidate.right))
        )
        yield from rpieces
        yield from (
            lpiece + b"^" + rpiece
            for (lpiece, rpiece) in zip(lpieces, rpieces)
        )

    def tokenize(self, text):
        """Generates segmented strings of text."""
        start = 0
        for candidate in self.candidates(text):
            if NEWLINE.match(candidate.boundary):
                continue
            if self.predict(self.extract_features(candidate)):
                yield text[start : candidate.left_index]
                start = candidate.right_index
        yield text[start:].rstrip()

    # Delegates all attributes not otherwise defined to the underlying
    # classifier.
    def __getattr__(self, name):
        return getattr(self._classifier, name)
