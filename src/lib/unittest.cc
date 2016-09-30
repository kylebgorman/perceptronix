// Copyright (C) 2015-2016 Kyle Gorman
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// unittest.cc: Basic unit tests for this library.

#include <cassert>

#include <memory>

#include "binomial_perceptron.h"
#include "multinomial_perceptron.h"

using namespace perceptronix;

// BINOMIAL PERCEPTRON STUFF.

enum class DFeat { GREEN, RED, BLUE, YELLOW, PURPLE, WHITE, __SIZE__ };

constexpr size_t F = static_cast<size_t>(DFeat::__SIZE__);

void TestBinomial() {
  using DenseFeature = DenseBinomialAveragedPerceptron::Feature;

  DenseBinomialAveragedPerceptron dba = DenseBinomialAveragedPerceptron(F);
  dba.Train({static_cast<DenseFeature>(DFeat::GREEN)}, false);
  dba.Train({static_cast<DenseFeature>(DFeat::GREEN)}, true);
  dba.Train({static_cast<DenseFeature>(DFeat::RED)}, false);
  dba.Train({static_cast<DenseFeature>(DFeat::GREEN)}, true);
  dba.Train({static_cast<DenseFeature>(DFeat::YELLOW)}, false);
  dba.Train({static_cast<DenseFeature>(DFeat::RED)}, true);
  dba.Train({static_cast<DenseFeature>(DFeat::RED)}, true);
  dba.Train({static_cast<DenseFeature>(DFeat::GREEN)}, true);
  dba.Train({static_cast<DenseFeature>(DFeat::BLUE)}, false);
  dba.Train({static_cast<DenseFeature>(DFeat::BLUE)}, false);
  dba.Train({static_cast<DenseFeature>(DFeat::RED)}, true);
  std::unique_ptr<AveragedWeight> score(
      dba.Score(static_cast<DenseFeature>(DFeat::PURPLE)));
  const DenseBinomialPerceptron db(&dba);
  assert(db.Predict({static_cast<DenseFeature>(DFeat::GREEN),
                     static_cast<DenseFeature>(DFeat::RED)}));
  assert(db.Write("db.pb"));
  std::unique_ptr<DenseBinomialPerceptron> dbr(
      DenseBinomialPerceptron::Read("db.pb"));
  assert(dbr.get());
  assert(dbr->Predict({static_cast<DenseFeature>(DFeat::GREEN),
                       static_cast<DenseFeature>(DFeat::RED)}));

  SparseBinomialAveragedPerceptron sba = SparseBinomialAveragedPerceptron(10);
  sba.Train({"green"}, true);
  sba.Train({"green"}, true);
  sba.Train({"red"}, true);
  sba.Train({"yellow"}, false);
  sba.Train({"red"}, true);
  sba.Train({"red"}, true);
  sba.Train({"blue"}, false);
  sba.Train({"blue"}, false);
  sba.Train({"red"}, false);
  score.reset(sba.Score("purple"));
  const SparseBinomialPerceptron sb(&sba);
  assert(sb.Predict({"green", "red"}));
  assert(sb.Write("sb.pb"));
  std::unique_ptr<SparseBinomialPerceptron> sbr(
      SparseBinomialPerceptron::Read("sb.pb"));
  assert(sbr.get());
  assert(sbr->Predict({"green", "red"}));
}

// MULTINOMIAL PERCEPTRON STUFF.

enum class Case { LOWER, MIXED, TITLE, UPPER, __SIZE__ };

constexpr size_t N = static_cast<size_t>(Case::__SIZE__);

void TestMultinomial() {
  using DenseFeature = DenseMultinomialAveragedPerceptron::Feature;

  DenseMultinomialAveragedPerceptron dma(F, N);
  dma.Train({static_cast<DenseFeature>(DFeat::BLUE)},
            static_cast<DenseFeature>(Case::MIXED));
  dma.Train({static_cast<DenseFeature>(DFeat::GREEN)},
            static_cast<DenseFeature>(Case::TITLE));
  dma.Train({static_cast<DenseFeature>(DFeat::GREEN)},
            static_cast<DenseFeature>(Case::MIXED));
  dma.Train({static_cast<DenseFeature>(DFeat::GREEN)},
            static_cast<DenseFeature>(Case::MIXED));
  const DenseMultinomialPerceptron dm(&dma);
  assert(dm.Predict({static_cast<DenseFeature>(DFeat::BLUE),
                     static_cast<DenseFeature>(DFeat::GREEN)}) ==
         static_cast<DenseFeature>(Case::MIXED));
  assert(dm.Write("dm.pb"));
  std::unique_ptr<DenseMultinomialPerceptron> dmr(
      DenseMultinomialPerceptron::Read("dm.pb"));
  assert(dmr.get());
  assert(dmr->Predict({static_cast<DenseFeature>(DFeat::BLUE),
                       static_cast<DenseFeature>(DFeat::GREEN)}) ==
         static_cast<DenseFeature>(Case::MIXED));

  SparseDenseMultinomialAveragedPerceptron sdma(F, N);
  sdma.Train({"blue"}, static_cast<DenseFeature>(Case::MIXED));
  sdma.Train({"green"}, static_cast<DenseFeature>(Case::TITLE));
  sdma.Train({"green"}, static_cast<DenseFeature>(Case::MIXED));
  sdma.Train({"green"}, static_cast<DenseFeature>(Case::MIXED));
  const SparseDenseMultinomialPerceptron sdm(&sdma);
  assert(sdm.Predict({"blue", "green"}) ==
         static_cast<DenseFeature>(Case::MIXED));
  assert(sdm.Write("sdm.pb"));
  std::unique_ptr<SparseDenseMultinomialPerceptron> sdr(
      SparseDenseMultinomialPerceptron::Read("sdm.pb"));
  assert(sdr.get());
  assert(sdr->Predict({"blue", "green"}) ==
         static_cast<DenseFeature>(Case::MIXED));

  SparseMultinomialAveragedPerceptron sma(F, N);
  sma.Train({"blue"}, "lower");
  sma.Train({"green"}, "lower");
  sma.Train({"green"}, "mixed");
  sma.Train({"green"}, "lower");
  const SparseMultinomialPerceptron sm(&sma);
  assert(sm.Predict({"blue", "green"}) == "lower");
  assert(sm.Write("sm.pb"));
  std::unique_ptr<SparseMultinomialPerceptron> smr(
      SparseMultinomialPerceptron::Read("sm.pb"));
  assert(smr.get());
  assert(smr->Predict({"blue", "green"}) == "lower");
}

int main(void) {
  TestBinomial();
  TestMultinomial();
  return 0;
}
