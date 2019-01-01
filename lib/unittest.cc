// unittest.cc: Basic unit tests for this library.

#include <cassert>

#include <iostream>
#include <memory>
#include <vector>

#include "binomial_perceptron.h"
#include "decode.h"
#include "multinomial_perceptron.h"

using namespace perceptronix;

// BINOMIAL PERCEPTRON STUFF.

enum class DFeat { GREEN, RED, BLUE, YELLOW, PURPLE, WHITE, __SIZE__ };

constexpr size_t F = static_cast<size_t>(DFeat::__SIZE__);

void TestBinomial() {
  using DenseFeature = DenseBinomialAveragingPerceptron::Feature;

  DenseBinomialAveragingPerceptron dba(F);
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
  const DenseBinomialPerceptron db(&dba);
  assert(db.Predict({static_cast<DenseFeature>(DFeat::GREEN),
                     static_cast<DenseFeature>(DFeat::RED)}));
  assert(db.Write("db.pb"));
  std::unique_ptr<DenseBinomialPerceptron> dbr(
      DenseBinomialPerceptron::Read("db.pb"));
  assert(dbr.get());
  assert(dbr->Predict({static_cast<DenseFeature>(DFeat::GREEN),
                       static_cast<DenseFeature>(DFeat::RED)}));

  SparseBinomialAveragingPerceptron sba(10);
  sba.Train({"green"}, true);
  sba.Train({"green"}, true);
  sba.Train({"red"}, true);
  sba.Train({"yellow"}, false);
  sba.Train({"red"}, true);
  sba.Train({"red"}, true);
  sba.Train({"blue"}, false);
  sba.Train({"blue"}, false);
  sba.Train({"red"}, false);
  const SparseBinomialPerceptron sb(&sba);
  assert(sb.Predict({"green", "red"}));
  assert(sb.Write("sb.pb"));
  std::unique_ptr<SparseBinomialPerceptron> sbr(
      SparseBinomialPerceptron::Read("sb.pb"));
  assert(sbr.get());
  assert(sbr->Predict({"green", "red"}));
}

// MULTINOMIAL PERCEPTRON STUFF.

enum class Case { LOWER, MIXED, TITLE, UPPER, DC, __SIZE__ };

constexpr size_t N = static_cast<size_t>(Case::__SIZE__);

void TestMultinomial() {
  using DenseFeature = DenseMultinomialAveragingPerceptron::Feature;

  DenseMultinomialAveragingPerceptron dma(F, N);
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

  SparseDenseMultinomialAveragingPerceptron sdma(F, N);
  sdma.Train({"blue"}, static_cast<DenseFeature>(Case::MIXED));
  sdma.Train({"green"}, static_cast<DenseFeature>(Case::TITLE));
  sdma.Train({"green"}, static_cast<DenseFeature>(Case::LOWER));
  sdma.Train({"green"}, static_cast<DenseFeature>(Case::MIXED));
  sdma.Train({"brown"}, static_cast<DenseFeature>(Case::UPPER));
  const SparseDenseMultinomialPerceptron sdm(&sdma);
  assert(sdm.Predict({"blue", "green"}) ==
         static_cast<DenseFeature>(Case::MIXED));
  assert(sdm.Write("sdm.pb"));
  std::unique_ptr<SparseDenseMultinomialPerceptron> sdr(
      SparseDenseMultinomialPerceptron::Read("sdm.pb"));
  assert(sdr.get());
  assert(sdr->Predict({"blue", "green"}) ==
         static_cast<DenseFeature>(Case::MIXED));

  SparseMultinomialAveragingPerceptron sma(F, N);
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

template <class Label>
void AssertStructured(const std::vector<Label> &ys,
                      const std::vector<Label> &yhats) {
  const auto size = ys.size();
  assert(size == yhats.size());
  for (size_t i = 0; i < size; ++i) assert(ys[i] == yhats[i]);
}

void TestStructured() {
  // Sparse binomial: word segmentation (space before?).
  SparseBinomialAveragingPerceptron sba(32);
  const std::vector<bool> binomial_ys = {false, true, true, true, false};
  const std::vector<std::vector<string>> evectors = {{"w=this", "*initial*"},
                                                     {"w=sentence"},
                                                     {"w=is"},
                                                     {"w=good"},
                                                     {"w=.", "*ultimate*"}};
  const SparseTransitionFunctor<bool> binomial_tf(1);
  SparseBinomialAveragingDecoder sbad(&sba, binomial_tf);
  for (size_t i = 0; i < 10; ++i) sbad.Train(evectors, binomial_ys);
  std::vector<bool> binomial_yhats;
  sbad.Predict(evectors, &binomial_yhats);
  AssertStructured(binomial_ys, binomial_yhats);
  const SparseBinomialPerceptron sb(&sba);
  const SparseBinomialDecoder sbd(sb, binomial_tf);
  sbd.Predict(evectors, &binomial_yhats);
  AssertStructured(binomial_ys, binomial_yhats);

  // Sparse-dense multinomial; case-restoration (reusing the evectors and
  // transition functor from above).
  SparseDenseMultinomialAveragingPerceptron sdma(32, N);
  const std::vector<size_t> dense_ys = {static_cast<size_t>(Case::TITLE),
                                        static_cast<size_t>(Case::LOWER),
                                        static_cast<size_t>(Case::LOWER),
                                        static_cast<size_t>(Case::LOWER),
                                        static_cast<size_t>(Case::DC)};
  const SparseTransitionFunctor<size_t> dense_tf(2);
  SparseDenseMultinomialAveragingDecoder sdmad(&sdma, dense_tf);
  for (size_t i = 0; i < 10; ++i) sdmad.Train(evectors, dense_ys);
  std::vector<size_t> dense_yhats;
  sdmad.Predict(evectors, &dense_yhats);
  AssertStructured(dense_ys, dense_yhats);
  const SparseDenseMultinomialPerceptron sdm(&sdma);
  const SparseDenseMultinomialDecoder sdmd(sdm, dense_tf);
  sdmd.Predict(evectors, &dense_yhats);
  AssertStructured(dense_ys, dense_yhats);
}

int main(void) {
  TestBinomial();
  TestMultinomial();
  TestStructured();

  std::cout << "Success!" << std::endl;

  return 0;
}
