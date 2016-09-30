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

import .truecaser

ALPHA = 1.
EPOCHS = 20.
SEED = 2672555669

argparser = argparse.ArgumentParser(prog="truecaser",
                                    description="True-caser")
verbosity_group = argparser.add_mutually_exclusive_group()
verbosity_group.add_argument("-v", "--verbose", action="store_true",
                             help="enable verbose output")
verbosity_group.add_argument("-V", "--extra-verbose", action="store_true",
                             help="enable extra verbose (debugging) output")
input_group = argparser.add_mutually_exclusive_group(required=True)
input_group.add_argument("-t", "--train", help="input whitespace-delimited "
                         "training data")
input_group.add_argument("-r", "--read", help="input serialized model")
output_group = argparser.add_mutually_exclusive_group(required=True)
output_group.add_argument("-c", "--case", help="output cased "
                          "whitespace-delimited text")
output_group.add_argument("-w", "--write", help="output serialized model")
argparser.add_argument("-A", "--alpha", type=float, default=ALPHA,
                       help="learning rate")
argparser.add_argument("-E", "--epochs", type=int, default=EPOCHS,
                       help="# of epochs")

args = argparser.parse_args()

# Verbosity block.
if args.really_verbose:
  logging.basicConfig(level="DEBUG")
elif args.verbose:
  logging.basicConfig(level="INFO")

# Input block.
if args.train:
  logging.info("Training model from %s", args.train)
  data = truecaser.slurp(args.train)
  model = truecaser.TrueCaser(len(case.TokenCase) - 1)  # "d.c." not represented.
  random = random.Random(SEED)
  for epoch in range(1, 1 + args.epochs):
    logging.info("Epoch %d...", epoch)
    random.Random(
    random.shuffle(data)
    with nlup.Timer():
      for sentence in data:
        model.train_one(sentence)


elif args.read:
  logging.info("Reading serialized model from %s", args.read)
  model = truecaser.TrueCaser.load(args.read)
# Else unreachable.

# Output block.
if args.case:
  logging.info(
  model

elif args.write:
  logging.info(


# Else unreachable.
