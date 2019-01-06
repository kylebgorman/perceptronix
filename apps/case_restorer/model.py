"""Case restorer model."""

import json

from typing import Dict, Iterator, List, Tuple

import nlup
import perceptronix

from case_restorer import *


# Constant feature strings.


INITIAL = "*initial*"
PENINITIAL = "*peninitial*"

HYPHEN = "*hyphen*"
NUMBER = "*number*"


Tokens = List[str]
Tags = List[TokenCase]
Patterns = List[Pattern]

Vector = List[str]
Vectors = List[Vector]


class CaseRestorer(object):
    """Case restorer model."""

    slots = ["_classifier", "_mpt"]

    def __init__(
        self, nfeats: int = 0x1000, order: int = 2, mpt: MixedPatternTable = {}
    ):
        self._classifier = perceptronix.SparseDenseMultinomialSequentialClassifier(
            nfeats, len(TokenCase), order
        )
        self._mpt = mpt

    # (De)serialization methods, overwritten to handle MPT, stored in the
    # metadata.

    @classmethod
    def read(cls, filename: str, order: int):
        """Reads case restorer model from serialized model file."""
        (
            classifier,
            metadata,
        ) = perceptronix.SparseDenseMultinomialSequentialClassifier.read(
            filename, order
        )
        new = cls.__new__(cls)
        new._classifier = classifier
        new._mpt = json.loads(metadata)
        return new

    def write(self, filename) -> None:
        self._classifier.write(filename, json.dumps(self._mpt))

    # Data readers.

    @staticmethod
    def sentences_from_file(filename: str) -> Iterator[Tokens]:
        with open(filename, "r") as source:
            for line in source:
                yield line.split()

    @staticmethod
    def tagged_sentences_from_file(
        filename: str
    ) -> Iterator[Tuple[Tokens, Tags, Patterns]]:
        for tokens in CaseRestorer.sentences_from_file(filename):
            (tags, patterns) = zip(*(get_tc(token) for token in tokens))
            # Casefolds after we get the labels.
            tokens = [token.casefold() for token in tokens]
            yield (tokens, list(tags), list(patterns))

    # Feature extraction.

    @staticmethod
    @nlup.listify
    def extract_emission_features(tokens: Tokens) -> Iterator[Vector]:
        """Generates emission feature vectors for a sentence."""
        # Tokens are assumed to have already been case-folded.
        for (i, token) in enumerate(tokens):
            vector = [f"w_i={token}"]
            # Context features.
            if i == 0:
                vector.append(INITIAL)
            else:
                vector.append(f"w_i-1={tokens[i - 1]}")
                if i == 1:
                    vector.append(PENINITIAL)
                else:
                    vector.append(f"w_i-2={tokens[i - 2]}")
            if i < len(tokens) - 1:
                vector.append(f"w_i+1={tokens[i + 1]}")
                if i < len(tokens) - 2:
                    vector.append(f"w_i+2={tokens[i + 2]}")
            # Shape features.
            if "-" in token:
                vector.append(HYPHEN)
            if any(ch.isdigit() for ch in token):
                vector.append(NUMBER)
            yield vector

    # Training and prediction.

    def train(self, vectors: Vectors, tags: Tags) -> Iterator[bool]:
        return self._classifier.train(vectors, tags)

    def predict(self, vectors: Vectors) -> Iterator[TokenCase]:
        for label in self._classifier.predict(vectors):
            yield TokenCase(label)

    def evaluate(self, vectors: Vectors, tags: Tags) -> int:
        return sum(
            tag == predicted_tag
            for (tag, predicted_tag) in zip(tags, self.predict(vectors))
        )

    def apply(self, tokens: Tokens) -> Iterator[str]:
        vectors = CaseRestorer.extract_emission_features(tokens)
        for (token, tag) in zip(tokens, self.predict(vectors)):
            pattern = self._mpt.get(token)
            yield apply_tc(token, tag, pattern)

    # Delegates all attributes not otherwise defined to the underlying
    # classifier.

    def __getattr__(self, name):
        return getattr(self._classifier, name)
