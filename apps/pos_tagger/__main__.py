"""POS tagging main."""

import argparse
import logging
import random

from typing import List, Tuple

import nlup

from .model import POSTagger

Data = List[Tuple[List[str], List[str]]]


def _read_data(filename: str) -> Data:
    return [
        (POSTagger.extract_emission_features(tokens), tags)
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
input_group.add_argument("-r", "--read", help="Input serialized model")
input_group.add_argument(
    "-t", "--train", help="Input two-column training training_data"
)
argparser.add_argument("-d", "--dev", help="Input two-column development data")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument(
    "-p", "--predict", help="Output tagged text from one-column data"
)
output_group.add_argument("-w", "--write", help="Output serialized model")
# Other options.
argparser.add_argument(
    "--nlabels", type=int, default=32, help="Initial number of labels"
)
argparser.add_argument(
    "--nfeats", type=int, default=0x1000, help="Initial number of features"
)
argparser.add_argument(
    "--epochs", type=int, default=5, help="Number of epochs"
)
argparser.add_argument("--order", type=int, default=2, help="Model order")
argparser.add_argument("--seed", type=int, default=1917, help="Random seed")
args = argparser.parse_args()

# Verbosity block.
if args.verbose:
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
else:
    logging.basicConfig(format="%(levelname)s: %(message)s")

# Input block.
if args.train:
    model = POSTagger(args.nfeats, args.nlabels, args.order)
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
                dev_correct += model.evaluate(vectors, tags)
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
