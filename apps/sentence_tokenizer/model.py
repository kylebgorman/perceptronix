"""Sentence tokenizer model."""

import collections
import logging

from typing import Iterator, List

import nlup  # type: ignore
import regex  # type: ignore
import perceptronix


# Constant feature strings.

NEWLINE = regex.compile(r"\n|\r|\r\n")


Candidate = collections.namedtuple(
    "Candidate", ("left_index", "right_index", "left", "right", "boundary")
)


class SentenceTokenizer(object):
    """Sentence tokenizer model."""

    slots = ["_candidate_regex", "_max_context", "_classifier"]

    def __init__(
        self,
        candidate_regex: str,
        max_context: int,
        nfeats: int = 0x1000,
        c: float = 0.0,
    ):
        self._candidate_regex = regex.compile(candidate_regex)
        self._max_context = max_context
        self._classifier = perceptronix.SparseBinomialModel(nfeats, c)

    @classmethod
    def read(cls, filename: str, candidate_regex: str, max_context: int):
        """Reads sentence tokenizer model from serialized model file."""
        (classifier, metadata) = perceptronix.SparseBinomialModel.read(
            filename
        )
        # TODO(kbg): Consider storing the candidate regex and max context
        # fields in the metadata string.
        if metadata:
            logging.warning("Ignoring metadata string: %s", metadata)
        new = cls.__new__(cls)
        new._classifier = classifier
        new._candidate_regex = regex.compile(candidate_regex)
        new._max_context = max_context
        return new

    # `write` is delegated to the underlying classifier.

    # Data readers.

    def candidates(self, text: str) -> Iterator[Candidate]:
        for match in self._candidate_regex.finditer(text, overlapped=True):
            (left, boundary, right) = match.groups()
            left_index = match.span()[0] + len(left)
            right_index = left_index + len(boundary)
            left_bound = min(len(left), self._max_context)
            right_bound = min(len(right), self._max_context)
            yield Candidate(
                left_index,
                right_index,
                left[-left_bound:],
                right[:right_bound],
                bool(NEWLINE.match(boundary)),
            )

    def candidates_from_file(self, filename: str) -> Iterator[Candidate]:
        with open(filename, "r") as source:
            return self.candidates(source.read())

    # Feature extraction

    @staticmethod
    @nlup.listify
    def extract_features(candidate: Candidate) -> Iterator[str]:
        """Generates feature vector for a candidate."""
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

    # Training and prediction.

    # `train` is delegated to the underlying classifier.

    def evaluate(self, vector: List[str], tag: bool) -> bool:
        return tag == self.predict(vector)

    def apply(self, text: str) -> Iterator[str]:
        """Tokenize a text."""
        start = 0
        for candidate in self.candidates(text):
            # Passes through any newlines already present.
            if candidate.boundary:
                continue
            vector = SentenceTokenizer.extract_features(candidate)
            if self.predict(vector):
                yield text[start : candidate.left_index + 1]
                start = candidate.right_index + 1
        yield text[start:].rstrip()

    # Delegates all attributes not otherwise defined to the underlying
    # classifier.

    def __getattr__(self, name):
        return getattr(self._classifier, name)
