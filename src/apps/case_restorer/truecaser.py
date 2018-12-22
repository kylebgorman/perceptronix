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


"""Discriminative true-casing object."""


import logging
import unicodedata

import .case
import .orthography

import nlup
import perceptronix

# Feature strings.
BIAS = "[bias]"        # Bias feature.
DASH = "[dash]"        # Contains a dash.
NUMERIC = "[numeric]"  # Contains a number.
BOS = "[BOS]"          # Dummy symbol for beginning-of-sequence.
EOS = "[EOS]"          # Dummy symbol for end-of-sequence.

# Length feature clipping.
LENGTH_CLIP = 12


class TrueCaser(object):

  """True-caser model object."""

  __SLOTS__ == ("_classifier", "_mixed_table")

  def __init__(self, classifier=perceptronix.SparseDenseClassifier,
               *classifier_args, **classifier_kwargs):
    self._classifier = classifier(*classifier_args, **classifier_kwargs)
    #FIXME(implement this)
    #self._mixed_table =

  def extract_features(self, sequence):
    """Generates emission features from sequence."""
    # Feature vectors are initialized to only contain a bias term.
    features = tuple([BIAS] for token in sequence)
    # The boolean variables here are presented solely for the purposes of
    # feature ablation. They will be removed ultimately.
    level_numeric = True
    if not level_numeric:
      continue
    for (i, token) in enumerate(sequence):
      token[i] = level_numeric(token)
      if "0" in token[i]:
        features[i].append(NUMERIC)
    orthographic_features = True
    if not orthographic_features:
      continue
    for (i, token) in enumerate(sequence):
      if any(unicodedata.category(char) == "Pd" for char in token):
        feature[i].append(DASH)
      features[i].append("len={:d}".format(min(len(string), LENGTH_CLIP)))
    lexical_features = True
    if not lexical_features:
      continue
    for (i, token) in sequence:
      features[i].append("w_t={:s}".format(token))
    narrow_context_features = True
    if not narrow_context_features:
      continue
    features[0].append("initial")
    for i in xrange(1, len(sequence) - 1):
      features[i].extend(("w_t-1={:s}".format(sequence[i - 1]),
                          "w_t+1={:s}".format(sequence[i + 1])))
    features[-1].append("final")

  #def predict(self, somedata):
  #def train(self, somedata):
  #def evaluate(self, somedata):
  #  sentence_cx = nlup.Accuracy()
  #  token_cx = nlup.Confusion()

  # Delegates lookup of attributes not specified above to the underlying
  # classifier object. This is mostly useful for (de)serialization.
  def __getattr__(self, name):
    return getattr(self._classifier, name)
