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

import argparse
import logging
import random
import re

import nlup

from .word_tokenizer import WordTokenizer

# Defaults
BOUNDARY_REGEX = r"(\s+)"
LEFT_REGEX = r"(?:\s+)(\S+[\.\!\?][\'\"]?)$"
RIGHT_REGEX = r"(\S+)(?:\s+)"
EPOCHS = 10

SEED = 2672555669

argparser = argparse.ArgumentParser(
    prog="word_tokenizer", description="A word tokenizer using a linear model."
)
argparser.add_argument(
    "-v", "--verbose", action="store_true", help="enable verbose output"
)
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("--train", help="input text training data")
input_group.add_argument("--read", help="input serialized model")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument("--tokenize", help="output tokenized text")
output_group.add_argument("--write", help="output serialized model")
# Other options.
argparser.add_argument(
    "--boundary_regex",
    default=BOUNDARY_REGEX,
    help="regular expression matching candidate boundaries",
)
argparser.add_argument(
    "--left_regex",
    default=LEFT_REGEX,
    help="regular expression matching left context",
)
argparser.add_argument(
    "--right_regex",
    default=RIGHT_REGEX,
    help="regular expression matching right context",
)
argparser.add_argument(
    "--max_context",
    default=8,
    type=int,
    help="maximum size for both context bytestrings (default: %(default)s)",
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
args = argparser.parse_args()

# Verbosity block.
if args.verbose:
    logging.basicConfig(level="INFO")

# Input block.
wtokenizer = WordTokenizer(
    args.left_regex,
    args.boundary_regex,
    args.right_regex,
    args.max_context,
    args.nfeats,
)
if args.train:
    logging.info("Training model from %s", args.train)
    with open(args.train, "r") as source:
        text = source.read()
    data = [
        (
            wtokenizer.extract_features(candidate),
            bool(NEWLINE.match(candidate.boundary)),
        )
        for candidate in wtokenizer.candidates(text)
    ]
    random.seed(SEED)
    for epoch in range(1, 1 + args.epochs):
        random.shuffle(data)
        logging.info("Epoch %d...", epoch)
        correct = 0
        with nlup.Timer():
            for (features, label) in data:
                correct += wtokenizer.train(features, label)
        logging.info("Resubstitution accuracy: %.4f", correct / len(data))
    logging.info("Averaging model")
    with nlup.Timer():
        wtokenizer.average()
elif args.read:
    logging.info("Reading model from %s", args.read)
    wtokenizer.read(args.read)
# Else unreachable.

# Output block.
if args.tokenize:
    logging.info("Tokenizing text from %s", args.tokenize)
    with open(args.tokenize, "r") as source:
        text = source.read()
    for line in wtokenizer.tokenize(text):
        print(line)
elif args.write:
    logging.info("Writing model to %s", args.write)
    wtokenizer.write(args.write)
# Else unreachable.
