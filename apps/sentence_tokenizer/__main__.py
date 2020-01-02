"""Sentence tokenizer main."""

import argparse
import logging
import random

from typing import List, Tuple

import nlup
import regex

from .model import SentenceTokenizer


CANDIDATE_REGEX = regex.compile(r"(?:\s+)(\S+[\.\!\?]['\"]?)(\s+)(\S+)(?:\s+)")


Data = List[Tuple[List[str], bool]]


def _read_data(filename: str, model: SentenceTokenizer) -> Data:
    return [
        (model.extract_features(candidate), candidate.boundary)
        for candidate in model.candidates_from_file(args.train)
    ]


argparser = argparse.ArgumentParser(
    prog="sentence_tokenizer", description="A sentence tokenizer model"
)
argparser.add_argument(
    "-v", "--verbose", action="store_true", help="enable verbose output"
)
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-r", "--read", help="input serialized model")
input_group.add_argument("-t", "--train", help="input text training data")
argparser.add_argument("-d", "--dev", help="input development training data")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument(
    "-p", "--predict", help="output tokenized text from untokenized train_data"
)
output_group.add_argument("-w", "--write", help="output serialized model")
argparser.add_argument(
    "--candidate_regex",
    default=CANDIDATE_REGEX,
    help="regular expression matching candidate boundaries",
)
argparser.add_argument(
    "--max_context",
    default=8,
    type=int,
    help="maximum size for context bytestrings (default: %(default)s)",
)
argparser.add_argument(
    "--nfeats",
    type=int,
    default=0x1000,
    help="initial number of features (default: %(default)s)",
)
argparser.add_argument(
    "--epochs",
    type=int,
    default=5,
    help="number of epochs (default: %(default)s)",
)
argparser.add_argument("--seed", type=int, default=0, help="random seed")
args = argparser.parse_args()

# Verbosity block.
if args.verbose:
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
else:
    logging.basicConfig(format="%(levelname)s: %(message)s")

# Input block.
if args.train:
    model = SentenceTokenizer(
        args.candidate_regex, args.max_context, args.nfeats
    )
    logging.info("Training model from %s", args.train)
    train_data = _read_data(args.train, model)
    dev_data = _read_data(args.dev, model) if args.dev else None
    random.seed(args.seed)
    for epoch in range(1, 1 + args.epochs):
        random.shuffle(train_data)
        logging.info("Epoch %d...", epoch)
        train_correct = 0
        with nlup.Timer():
            for (vector, boundary) in train_data:
                train_correct += model.train(vector, boundary)
        logging.info(
            "Resubstitution accuracy: %.4f", train_correct / len(train_data)
        )
        if args.dev:
            dev_correct = 0
            for (vector, boundary) in dev_data:
                dev_correct += model.evaluate(vector, boundary)
            logging.info(
                "Development accuracy: %.4f", dev_correct / len(dev_data)
            )
    logging.info("Averaging model")
    model.average()
elif args.read:
    logging.info("Reading model from %s", args.read)
    model = SentenceTokenizer.read(
        args.read, args.candidate_regex, args.max_context
    )
# Else unreachable.

# Output block.
if args.predict:
    logging.info("Tokenizing text from %s", args.predict)
    with open(args.predict, "r") as source:
        for line in model.apply(source.read()):
            print(line)
elif args.write:
    logging.info("Writing model to %s", args.write)
    model.write(args.write)
# Else unreachable.
