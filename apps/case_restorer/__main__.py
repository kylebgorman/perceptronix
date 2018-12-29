"""Case restorer main."""

import argparse
import logging
import random

from typing import List, Tuple

import nlup

from case_restorer import case

from .model import CaseRestorer

Tokens = List[str]
Tags = List[case.TokenCase]
Sentences = List[Tuple[Tokens, Tags]]


def _read_data(filename: str) -> Tuple[Sentences, case.MixedPatternTable]:
    data = []
    mixed_patterns: case.MixedPatternTable = {}
    gen = CaseRestorer.tagged_sentences_from_file(filename)
    for (tokens, tags, patterns) in gen:
        vectors = CaseRestorer.extract_emission_features(tokens)
        data.append((vectors, tags))
        for (token, pattern) in zip(tokens, patterns):
            if pattern is not None:
                mixed_patterns[token] = pattern
    return (data, mixed_patterns)


def _data_size(sentences: Sentences) -> int:
    return sum(len(tags) for (_, tags) in sentences)


argparser = argparse.ArgumentParser(
    prog="case_restorer", description="A case restoration model"
)
verbosity_group = argparser.add_mutually_exclusive_group()
verbosity_group.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose output"
)
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-r", "--read", help="Input serialized model")
input_group.add_argument("-t", "--train", help="Input tokenized training data")
argparser.add_argument("-d", "--dev", help="Input tokenized development data")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument(
    "-p", "--predict", help="Output cased text from tokenized training_data"
)
output_group.add_argument("-w", "--write", help="Output serialized model")
argparser.add_argument(
    "--nfeats", type=int, default=0x1000, help="Initial number of features"
)
argparser.add_argument(
    "--epochs", type=int, default=5, help="Number of epochs"
)
argparser.add_argument("--seed", type=int, default=1917, help="Random seed")
args = argparser.parse_args()

# Verbosity block.
if args.verbose:
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
else:
    logging.basicConfig(format="%(levelname)s: %(message)s")

# Input block.
if args.train:
    logging.info("Training model from %s", args.train)
    (train_sents, train_mpt) = _read_data(args.train)
    train_size = _data_size(train_sents)
    if args.dev:
        dev_sents = _read_data(args.dev)[0]
        dev_size = _data_size(dev_sents)
    model = CaseRestorer(args.nfeats, train_mpt)
    random.seed(args.seed)
    for epoch in range(1, 1 + args.epochs):
        random.shuffle(train_sents)
        logging.info("Epoch %d...", epoch)
        train_correct = 0
        with nlup.Timer():
            for (vectors, tags) in train_sents:
                train_correct += sum(model.train(vectors, tags))
        logging.info(
            "Resubstitution accuracy: %.4f", train_correct / train_size
        )
        if args.dev:
            dev_correct = 0
            for (vectors, tags) in dev_sents:
                dev_correct += sum(
                    tag == predicted
                    for (tag, predicted) in zip(
                        tags, model.tag_vectors(vectors)
                    )
                )
            logging.info("Development accuracy: %.4f", dev_correct / dev_size)
    logging.info("Averaging model...")
    model.average()
elif args.read:
    logging.info("Reading model from %s", args.read)
    model = CaseRestorer.read(args.read)
# Else unreachable.

# Output block.
if args.predict:
    logging.info("Casing text from %s", args.predict)
    for tokens in CaseRestorer.sentences_from_file(args.predict):
        print(" ".join(model.apply(tokens)))
elif args.write:
    logging.info("Writing model to %s", args.write)
    model.write(args.write)
# Else unreachable.
