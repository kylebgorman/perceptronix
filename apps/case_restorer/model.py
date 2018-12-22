"""Case restorer model."""

from typing import Iterable, Iterator, List, Tuple

import nlup
import perceptronix

from case_restorer import case

# Constant feature strings.

BIAS = "*bias*"

INITIAL = "*initial*"
PENINITIAL = "*peninitial*"

HYPHEN = "*hyphen*"
NUMBER = "*number*"

# Types.

Tokens = List[str]
Tags = Iterable[case.TokenCase]
Vector = List[str]
Vectors = Iterable[Vector]


class CaseRestorer(object):
    """Case restorer model."""

    def __init__(self, nfeats: int = 0x1000, alpha: float = 1):
        self._classifier = perceptronix.SparseDenseMultinomialClassifier(
            nfeats, len(case.TokenCase), alpha
        )

    @classmethod
    def read(cls, filename: str):
        """Reads case restorer model from serialized model file."""
        result = cls.__new__(cls)
        result._classifier = perceptronix.SparseDenseMultinomialClassifier.read(
            filename
        )
        return result

    # Data readers.

    @staticmethod
    def sentences_from_file(filename: str) -> Iterator[Tokens]:
        with open(filename, "r") as source:
            for line in source:
                tokens = line.split()
                if tokens:
                    yield tokens

    @staticmethod
    def tagged_sentences_from_file(
        filename: str
    ) -> Iterator[Tuple[Tokens, Tags]]:
        for tokens in CaseRestorer.sentences_from_file(filename):
            # TODO(kbg): Adds support for mixed case patterns.
            tags = [case.get_tc(token)[0] for token in tokens]
            yield (tokens, tags)

    # Feature extraction.

    @staticmethod
    @nlup.listify
    def extract_emission_features(tokens: Tokens) -> Iterator[Vector]:
        """Geneates emission feature vectors for a sentence."""
        tokens = [token.casefold() for token in tokens]
        for (i, token) in enumerate(tokens):
            vector = [BIAS, f"w_i={token}"]
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

    # Not yet supported: transition features, greedy or optimal.

    def train(self, vectors: Vectors, tags: Tags) -> Iterator[case.TokenCase]:
        for (vector, tag) in zip(vectors, tags):
            yield self._classifier.train(vector, tag.value)

    def tag_vectors(self, vectors: Vectors) -> Iterator[case.TokenCase]:
        for vector in vectors:
            yield case.TokenCase(self._classifier.predict(vector))

    def tag(self, tokens: Tokens) -> Iterator[case.TokenCase]:
        return self.tag_vectors(CaseRestorer.extract_emission_features(tokens))

    def apply(self, tokens: Tokens) -> Iterator[str]:
        """Case-restore a list of tokens."""
        # TODO(kbg): Will break in mixed-case scenarios.
        for (token, tag) in zip(tokens, self.tag(tokens)):
            yield case.apply_tc(token, tag)

    # Delegates all attributes not otherwise defined to the underlying
    # classifier.

    def __getattr__(self, name):
        return getattr(self._classifier, name)
