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
import regex

import nlup

from .sentence_tokenizer import SentenceTokenizer


# Defaults.
CANDIDATE_REGEX = regex.compile("(?:\s+)(\S+[\.\!\?][\'\"]?)(\s+)(\S+)(?:\s+)")
MAX_CONTEXT = 5
NFEATS = 0x1000
ALPHA = 1.
EPOCHS = 10

SEED = 11215
NEWLINE = regex.compile("[\n\r]")


argparser = argparse.ArgumentParser(prog="sentence_tokenizer",
    description="A sentence tokenizer using a linear model")
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="enable verbose output")
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-t", "--train", help="input text training data")
input_group.add_argument("-r", "--read", help="input serialized model")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument("-s", "--tokenize", help="output tokenized text")
output_group.add_argument("-w", "--write", help="output serialized model")
# Other options.
argparser.add_argument("--candidate_regex", default=CANDIDATE_REGEX,
                       help="regular expression matching candidate boundaries")
argparser.add_argument("--max_context", default=MAX_CONTEXT, type=int,
                       help="Maximum size for both context bytestrings "
                       "(default: {})".format(MAX_CONTEXT))
argparser.add_argument("--nfeats", type=int, default=NFEATS,
                       help="Initial number of features "
                       "(default: {})".format(NFEATS))
argparser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="# of epochs (default: {})".format(EPOCHS))
argparser.add_argument("--alpha", type=float, default=ALPHA,
                       help="learning rate (default: {})".format(ALPHA))
args = argparser.parse_args()

# Verbosity block.
if args.verbose:
  logging.basicConfig(level="INFO")

# Input block.
if args.train:
  stokenizer = SentenceTokenizer(args.candidate_regex, args.max_context,
                                 args.nfeats, args.alpha)
  logging.info("Training model from %s", args.train)
  data = [(tuple(stokenizer.extract_features(candidate)),
           bool(NEWLINE.match(candidate.boundary)))
          for candidate in stokenizer.candidates_from_file(args.train)]
  random.seed(SEED)
  for epoch in range(1, 1 + args.epochs):
    random.shuffle(data)
    logging.info("Epoch %d...", epoch)
    correct = 0
    with nlup.Timer():
      for (features, label) in data:
         correct += stokenizer.train(features, label)
    logging.info("Resubstitution accuracy: %.4f", correct / len(data))
  logging.info("Averaging model")
  with nlup.Timer():
    stokenizer.average()
elif args.read:
  logging.info("Reading model from %s", args.read)
  stokenizer = SentenceTokenizer.read(args.read, args.candidate_regex,
                                      args.max_context)
# Else unreachable.

# Output block.
if args.tokenize:
  logging.info("Tokenizing text from %s", args.tokenize)
  with open(args.tokenize, "r") as source:
    text = source.read()
  for line in stokenizer.tokenize(text):
    print(line)
elif args.write:
  logging.info("Writing model to %s", args.write)
  stokenizer.write(args.write)
# Else unreachable.
