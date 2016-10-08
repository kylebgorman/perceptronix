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


import unicodedata

import perceptronix


# Default encoding and normalization scheme.
ENCODING = "utf8"
NORMALIZATION = "NFC"


# Constant feature string.
BIAS = "*bias*"
INITIAL = "*initial*"
PENINITIAL = "*peninitial*"
PENULTIMATE = "*penultimate*"
ULTIMATE = "*ultimate*"


# For other (more important) defaults, see __main__.py in this directory.


class POSTagger(object):

  __slots__ = ("_classifier",)

  def __init__(self, *args, **kwargs):
    self._classifier = perceptronix.SparseMultinomialClassifier(*args, **kwargs)

  @classmethod
  def read(cls, filename):
    """Reads POS tagger model from serialized model file."""
    result = cls.__new__(cls)
    result._classifier = perceptronix.SparseMultinomialClassifier.read(filename)
    return result

  @staticmethod
  def _normalize(string):
    return unicodedata.normalize(NORMALIZATION, string)

  @staticmethod
  def untagged_sentences_from_file(filename):
    tokens = []
    with open(filename, "r") as source:
       for line in source:
         line = line.strip()
         if (not line) and tokens:
           yield tokens
           tokens = []
         else:
           tokens.append(POSTagger._normalize(line))
    if tokens:
      yield tokens

  @staticmethod
  def tagged_sentences_from_file(filename):
    tokens = []
    tags = []
    with open(filename, "r") as source:
      for line in source:
        line = line.strip()
        if (not line) and tokens:
          yield (tokens, tags)
          tokens = []
          tags = []
        else:
          (token, tag) = POSTagger._normalize(line).split("\t", 2)
          tokens.append(token)
          tags.append(tag)
    if tokens:
      yield (tokens, tags)

  @staticmethod
  def _affix_features(token):
    if len(token) >= 6:
      yield "pre_i=" + token[:3]
      yield "suf_i=" + token[-3:]

  @staticmethod
  def extract_emission_features(tokens):
    """Generates emission feature vectors for a sentence."""
    # NB: This doesn't do any clever capitalization features, which are
    # probably important for state-of-the-art results on English, but YMMV.
    if not tokens:
      return []
    vectors = [[BIAS] for _ in range(len(tokens))]
    # Left edge features.
    initial = tokens[0]
    initial_vector = vectors[0]
    initial_vector.append(INITIAL)
    initial_vector.append("w_i=" + initial)
    initial_vector.extend(POSTagger._affix_features(initial))
    if len(tokens) > 1:
      peninitial = tokens[1]
      initial_vector.append("w_i+1=" + peninitial)
      peninitial_vector = vectors[1]
      peninitial_vector.append(PENINITIAL)
      peninitial_vector.append("w_i-1=" + initial)
      peninitial_vector.append("w_i=" + peninitial)
      peninitial_vector.extend(POSTagger._affix_features(peninitial))
      if len(tokens) > 2:
        antepeninitial = tokens[2]
        initial_vector.append("w_i+2=" + antepeninitial)
        peninitial_vector.append("w_i+1=" + antepeninitial)
    # Internal features.
    for (i, token) in enumerate(tokens[2:-2], 2):
      current_vector = vectors[i]
      current_vector.append("w_i-2=" + tokens[i - 2])
      current_vector.append("w_i-1=" + tokens[i - 1])
      current_vector.append("w_i=" + token)
      current_vector.extend(POSTagger._affix_features(token))
      current_vector.append("w_i+1=" + tokens[i + 1])
      current_vector.append("w_i+2=" + tokens[i + 2])
    # Right edge features.
    ultimate = tokens[-1]
    ultimate_vector = vectors[-1]
    ultimate_vector.append(ULTIMATE)
    ultimate_vector.append("w_i=" + ultimate)
    ultimate_vector.extend(POSTagger._affix_features(ultimate))
    if len(tokens) > 1:
      penultimate = tokens[-2]
      ultimate_vector.append("w_i-1=" + penultimate)
      penultimate_vector = vectors[-2]
      penultimate_vector.append(PENULTIMATE)
      penultimate_vector.append("w_i+1=" + ultimate)
      penultimate_vector.append("w_i=" + penultimate)
      penultimate_vector.extend(POSTagger._affix_features(penultimate))
      if len(tokens) > 2:
        antepenultimate = tokens[-3]
        ultimate_vector.append("w_i-2=" + antepenultimate)
        penultimate_vector.append("w_i-1" + antepenultimate)
    # And we're done!
    return vectors

  # Not yet supported: transition features, greedy or optimal.

  def tag(self, tokens):
    """Generates tags for a sentence."""
    for current_vector in POSTagger.extract_emission_features(tokens):
      yield self.predict(current_vector).decode(ENCODING)

  def train(self, emission_vectors, tags):
    for (current_vector, tag) in zip(emission_vectors, tags):
      yield self._classifier.train(current_vector, tag)

  # Delegates all attributes not otherwise defined to the underlying
  # classifier.
  def __getattr__(self, name):
    return getattr(self._classifier, name)
