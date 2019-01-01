"""POS tagger model."""

import logging

from typing import Iterator, List, Tuple

import perceptronix


# Constant feature string.

INITIAL = "*initial*"
PENINITIAL = "*peninitial*"
PENULTIMATE = "*penultimate*"
ULTIMATE = "*ultimate*"

HYPHEN = "*hyphen*"
NUMBER = "*number*"
UPPERCASE = "*uppercase*"

# Types.

Tokens = List[str]
Tags = List[str]
Vector = List[str]
Vectors = List[Vector]


class POSTagger(object):
    """Part-of-speech tagger model."""

    slots = ["_classifier"]

    def __init__(
        self, nfeats: int = 0x1000, nlabels: int = 32, order: int = 2
    ):
        self._classifier = perceptronix.SparseMultinomialSequentialClassifier(
            nfeats, nlabels, order
        )

    @classmethod
    def read(cls, filename: str, order: int):
        """Reads POS tagger model from serialized model file."""
        (
            classifier,
            metadata,
        ) = perceptronix.SparseMultinomialSequentialClassifier.read(
            filename, order
        )
        if metadata:
            logging.warning("Ignoring metadata string: %s", metadata)
        new = cls.__new__(cls)
        new._classifier = classifier
        return new

    # `write` is dispatched to the underlying classifier.

    # Data readers.

    @staticmethod
    def sentences_from_file(filename: str) -> Iterator[Tokens]:
        tokens: Tokens = []
        with open(filename, "r") as source:
            for line in source:
                line = line.strip()
                if (not line) and tokens:
                    yield tokens
                    tokens = []
                else:
                    tokens.append(line)
        if tokens:
            yield tokens

    @staticmethod
    def tagged_sentences_from_file(
        filename: str
    ) -> Iterator[Tuple[Tokens, Tags]]:
        tokens: Tokens = []
        tags: Tags = []
        with open(filename, "r") as source:
            for line in source:
                line = line.strip()
                if (not line) and tokens and tags:
                    yield (tokens.copy(), tags.copy())
                    tokens.clear()
                    tags.clear()
                else:
                    (token, tag) = line.split("\t", 1)
                    tokens.append(token)
                    tags.append(tag)
        if tokens:
            yield (tokens, tags)

    # Feature extraction.

    @staticmethod
    def _shape_features(token: str) -> Iterator[str]:
        if len(token) > 4:  # TODO(kbg): Tune this.
            for i in range(1, 1 + 4):
                yield f"pre({i})={token[:i]}"
                yield f"suf({i})={token[-i:]}"
        if "-" in token:
            yield HYPHEN
        if any(ch.isdigit() for ch in token):
            yield NUMBER
        if any(ch.isupper() for ch in token):
            yield UPPERCASE

    @staticmethod
    def extract_emission_features(tokens: Tokens) -> List[Vector]:
        """Generates emission feature vectors for a sentence."""
        # TODO(kbg): Add casing features.
        if not tokens:
            return []
        vectors = [[f"w_i={token}"] for token in tokens]
        # Left edge features.
        initial = tokens[0]
        initial_vector = vectors[0]
        initial_vector.append(INITIAL)
        initial_vector.extend(POSTagger._shape_features(initial))
        if len(tokens) > 1:
            peninitial = tokens[1]
            initial_vector.append(f"w_i+1={peninitial}")
            peninitial_vector = vectors[1]
            peninitial_vector.append(PENINITIAL)
            peninitial_vector.append(f"w_i-1={initial}")
            peninitial_vector.extend(POSTagger._shape_features(peninitial))
            if len(tokens) > 2:
                antepeninitial = tokens[2]
                initial_vector.append(f"w_i+2={antepeninitial}")
                peninitial_vector.append(f"w_i+1={antepeninitial}")
        # Internal features.
        for (i, token) in enumerate(tokens[2:-2], 2):
            current_vector = vectors[i]
            current_vector.append(f"w_i-2={tokens[i - 2]}")
            current_vector.append(f"w_i-1={tokens[i - 1]}")
            current_vector.extend(POSTagger._shape_features(token))
            current_vector.append(f"w_i+1={tokens[i + 1]}")
            current_vector.append(f"w_i+2={tokens[i + 2]}")
        # Right edge features.
        ultimate = tokens[-1]
        ultimate_vector = vectors[-1]
        ultimate_vector.append(ULTIMATE)
        ultimate_vector.extend(POSTagger._shape_features(ultimate))
        if len(tokens) > 1:
            penultimate = tokens[-2]
            ultimate_vector.append(f"w_i-1={penultimate}")
            penultimate_vector = vectors[-2]
            penultimate_vector.append(PENULTIMATE)
            penultimate_vector.append(f"w_i+1={ultimate}")
            penultimate_vector.extend(POSTagger._shape_features(penultimate))
            if len(tokens) > 2:
                antepenultimate = tokens[-3]
                ultimate_vector.append(f"w_i-2={antepenultimate}")
                penultimate_vector.append(f"w_i-1={antepenultimate}")
        # And we're done!
        return vectors

    # Training and prediction.

    def train(self, vectors: Vectors, tags: Tags) -> Iterator[bool]:
        return self._classifier.train_sequence(vectors, tags)

    def predict(self, vectors: Vectors) -> Iterator[str]:
        for tag in self._classifier.predict_sequence(vectors):
            yield tag.decode("utf8")

    def evaluate(self, vectors: Vectors, tags: Tags) -> int:
        return sum(
            tag == predicted_tag
            for (tag, predicted_tag) in zip(tags, self.predict(vectors))
        )

    def apply(self, tokens: Tokens) -> Iterator[Tuple[str, str]]:
        vectors = POSTagger.extract_emission_features(tokens)
        yield from zip(tokens, self.predict(vectors))

    # Delegates all attributes not otherwise defined to the underlying
    # classifier.

    def __getattr__(self, name):
        return getattr(self._classifier, name)
