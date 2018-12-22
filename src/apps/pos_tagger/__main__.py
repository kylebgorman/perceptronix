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

import nlup

from .pos_tagger import POSTagger

# Defaults.
NLABELS = 5
NFEATS = 0x1000
ALPHA = 1.
EPOCHS = 10

SEED = 11215


argparser = argparse.ArgumentParser(prog="pos_tagger",
    description="A POS tagger using a linear model")
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="enable verbose output")
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-t", "--train", help="input text training data")
input_group.add_argument("-r", "--read", help="input serialized model")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument("-s", "--tag", help="output tagged text")
output_group.add_argument("-w", "--write", help="output serialized model")
# Other options.
argparser.add_argument("--nlabels", type=int, default=NLABELS,
                       help="Initial number of labels "
                       "(default: {})".format(NLABELS))
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
  ptagger = POSTagger(args.nlabels, args.nfeats, args.alpha)
  logging.info("Training model from %s", args.train)
  data = [(POSTagger.extract_emission_features(tokens), tags) for
          (tokens, tags) in POSTagger.tagged_sentences_from_file(args.train)]
  random.seed(SEED)
  for epoch in range(1, 1 + args.epochs):
    random.shuffle(data)
    logging.info("Epoch %d...", epoch)
    correct = 0
    size = 0
    with nlup.Timer():
      for (emission_vectors, tags) in data:
        correct += sum(ptagger.train(emission_vectors, tags))
        size += len(tags)
    logging.info("Resubstitution accuracy: %.4f", correct / size)
  logging.info("Averaging model")
  with nlup.Timer():
    ptagger.average()
elif args.read:
  logging.info("Reading model from %s", args.read)
  ptagger = POSTagger.read(args.read)
# Else unreachable.

# Output block.
if args.tag:
  logging.info("Tagging text from %s", args.tag)
  for tokens in POSTagger.untagged_sentences_from_file(args.tag):
    tags = ptagger.tag(tokens)
    print("\n".join(token + "\t" + tag for (token, tag) in zip(tokens, tags)))
    print()
elif args.write:
  logging.info("Writing model to %s", args.write)
  ptagger.write(args.write)
# Else unreachable.
