"""POS tagging main."""

import argparse
import logging
import random

from typing import List, Tuple

import nlup  # type: ignore

from .model import POSTagger

Data = List[Tuple[List[str], List[str]]]


def _read_data(filename: str) -> Data:
    # TODO(kbg): Why is this raising a type error?
    return [
        (POSTagger.extract_emission_features(tokens), tags)  # type: ignore
        for (tokens, tags) in POSTagger.tagged_sentences_from_file(filename)
    ]


def _data_size(data: Data) -> int:
    return sum(len(tags) for (_, tags) in data)


argparser = argparse.ArgumentParser(
    prog="pos_tagger", description="A POS tagger using a linear model"
)
argparser.add_argument(
    "-v", "--verbose", action="store_true", help="enable verbose output"
)
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-r", "--read", help="input serialized model")
input_group.add_argument(
    "-t", "--train", help="input two-column training training_data"
)
argparser.add_argument("-d", "--dev", help="input two-column development data")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument(
    "-p", "--predict", help="output tagged text from one-column data"
)
output_group.add_argument("-w", "--write", help="output serialized model")
# Other options.
argparser.add_argument(
    "--nlabels",
    type=int,
    default=32,
    help="initial number of labels (default: %(default)s)",
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
argparser.add_argument(
    "--order", type=int, default=2, help="model order (default: %(default)s)"
)
argparser.add_argument(
    "-c",
    type=float,
    default=0.0,
    help="margin coefficient (default: %(default)s)",
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
    model = POSTagger(args.nfeats, args.nlabels, args.order, args.c)
    logging.info("Training model from %s", args.train)
    train_data = _read_data(args.train)
    train_size = _data_size(train_data)
    if args.dev:
        dev_data = _read_data(args.dev)
        dev_size = _data_size(dev_data)
    random.seed(args.seed)
    for epoch in range(1, 1 + args.epochs):
        random.shuffle(train_data)
        logging.info("Epoch %d...", epoch)
        train_correct = 0
        with nlup.Timer():
            for (vectors, tags) in train_data:
                train_correct += model.train(vectors, tags)
        logging.info(
            "Resubstitution accuracy: %.4f", train_correct / train_size
        )
        if args.dev:
            dev_correct = 0
            for (vectors, tags) in dev_data:
                # TODO(kbg): Why is this raising a typing error?
                dev_correct += model.evaluate(vectors, tags)  # type: ignore
            logging.info("Development accuracy: %.4f", dev_correct / dev_size)
    logging.info("Averaging model...")
    model.average()
elif args.read:
    logging.info("Reading model from %s", args.read)
    model = POSTagger.read(args.read, args.order)
# Else unreachable.

# Output block.
if args.predict:
    logging.info("Tagging text from %s", args.predict)
    for tokens in POSTagger.sentences_from_file(args.predict):
        for (token, tag) in model.apply(tokens):
            print(f"{token}\t{tag}")
        print()
elif args.write:
    logging.info("Writing model to %s", args.write)
    model.write(args.write)
# Else unreachable.
