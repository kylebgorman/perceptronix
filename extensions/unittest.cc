// unittest.cc: Basic unit tests for this library.

#include <cassert>

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "binomial_model.h"
#include "multinomial_model.h"

namespace perceptronix {
namespace {

// BINOMIAL PERCEPTRON STUFF.

enum class DFeat { GREEN, RED, BLUE, YELLOW, PURPLE, WHITE, __SIZE__ };

constexpr size_t F = static_cast<size_t>(DFeat::__SIZE__);

void TestBinomial() {
  using DenseFeature = DenseBinomialModel::Feature;

  DenseBinomialModel dbm(F, 1);
  dbm.Train({static_cast<DenseFeature>(DFeat::GREEN)}, false);
  dbm.Train({static_cast<DenseFeature>(DFeat::GREEN)}, true);
  dbm.Train({static_cast<DenseFeature>(DFeat::RED)}, false);
  dbm.Train({static_cast<DenseFeature>(DFeat::GREEN)}, true);
  dbm.Train({static_cast<DenseFeature>(DFeat::YELLOW)}, false);
  dbm.Train({static_cast<DenseFeature>(DFeat::RED)}, true);
  dbm.Train({static_cast<DenseFeature>(DFeat::RED)}, true);
  dbm.Train({static_cast<DenseFeature>(DFeat::GREEN)}, true);
  dbm.Train({static_cast<DenseFeature>(DFeat::BLUE)}, false);
  dbm.Train({static_cast<DenseFeature>(DFeat::BLUE)}, false);
  dbm.Train({static_cast<DenseFeature>(DFeat::RED)}, true);
  dbm.Average();
  assert(dbm.Predict({static_cast<DenseFeature>(DFeat::GREEN),
                      static_cast<DenseFeature>(DFeat::RED)}));
  assert(dbm.Write("db.pb"));
  std::unique_ptr<DenseBinomialPerceptron> dbm2(
      DenseBinomialPerceptron::Read("db.pb"));
  assert(dbm2.get());
  assert(dbm2->Predict({static_cast<DenseFeature>(DFeat::GREEN),
                        static_cast<DenseFeature>(DFeat::RED)}));

  SparseBinomialModel sbm(10, 1);
  sbm.Train({"green"}, true);
  sbm.Train({"green"}, true);
  sbm.Train({"red"}, true);
  sbm.Train({"yellow"}, false);
  sbm.Train({"red"}, true);
  sbm.Train({"red"}, true);
  sbm.Train({"blue"}, false);
  sbm.Train({"blue"}, false);
  sbm.Train({"red"}, false);
  sbm.Average();
  assert(sbm.Predict({"green", "red"}));
  assert(sbm.Write("sb.pb"));
  std::unique_ptr<SparseBinomialPerceptron> sbm2(
      SparseBinomialPerceptron::Read("sb.pb"));
  assert(sbm2.get());
  assert(sbm2->Predict({"green", "red"}));
}

// MULTINOMIAL PERCEPTRON STUFF.

enum class Case { LOWER, MIXED, TITLE, UPPER, DC, __SIZE__ };

constexpr size_t N = static_cast<size_t>(Case::__SIZE__);

void TestMultinomial() {
  using DenseFeature = DenseMultinomialModel::Feature;

  DenseMultinomialModel dmm(F, N, 1);
  dmm.Train({static_cast<DenseFeature>(DFeat::BLUE)},
            static_cast<DenseFeature>(Case::MIXED));
  dmm.Train({static_cast<DenseFeature>(DFeat::GREEN)},
            static_cast<DenseFeature>(Case::TITLE));
  dmm.Train({static_cast<DenseFeature>(DFeat::GREEN)},
            static_cast<DenseFeature>(Case::MIXED));
  dmm.Train({static_cast<DenseFeature>(DFeat::GREEN)},
            static_cast<DenseFeature>(Case::MIXED));
  dmm.Average();
  assert(dmm.Predict({static_cast<DenseFeature>(DFeat::BLUE),
                      static_cast<DenseFeature>(DFeat::GREEN)}) ==
         static_cast<DenseFeature>(Case::MIXED));
  assert(dmm.Write("dm.pb"));
  std::unique_ptr<DenseMultinomialPerceptron> dmm2(
      DenseMultinomialPerceptron::Read("dm.pb"));
  assert(dmm2.get());
  assert(dmm2->Predict({static_cast<DenseFeature>(DFeat::BLUE),
                        static_cast<DenseFeature>(DFeat::GREEN)}) ==
         static_cast<DenseFeature>(Case::MIXED));

  SparseDenseMultinomialModel sdmm(F, N);
  sdmm.Train({"blue"}, static_cast<DenseFeature>(Case::MIXED));
  sdmm.Train({"green"}, static_cast<DenseFeature>(Case::TITLE));
  sdmm.Train({"green"}, static_cast<DenseFeature>(Case::LOWER));
  sdmm.Train({"green"}, static_cast<DenseFeature>(Case::MIXED));
  sdmm.Train({"brown"}, static_cast<DenseFeature>(Case::UPPER));
  sdmm.Average();
  assert(sdmm.Predict({"blue", "green"}) ==
         static_cast<DenseFeature>(Case::MIXED));
  assert(sdmm.Write("sdm.pb"));
  std::unique_ptr<SparseDenseMultinomialPerceptron> sdmm2(
      SparseDenseMultinomialPerceptron::Read("sdm.pb"));
  assert(sdmm2.get());
  assert(sdmm2->Predict({"blue", "green"}) ==
         static_cast<DenseFeature>(Case::MIXED));

  SparseMultinomialModel smm(F, N, 1);
  smm.Train({"blue"}, "lower");
  smm.Train({"green"}, "lower");
  smm.Train({"green"}, "mixed");
  smm.Train({"green"}, "lower");
  smm.Average();
  assert(smm.Predict({"blue", "green"}) == "lower");
  assert(smm.Write("sm.pb"));
  std::unique_ptr<SparseMultinomialPerceptron> smm2(
      SparseMultinomialPerceptron::Read("sm.pb"));
  assert(smm2.get());
  assert(smm2->Predict({"blue", "green"}) == "lower");
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
  SparseBinomialSequentialModel sbsm(32, 2, 1);
  const std::vector<bool> binomial_ys = {false, true, true, true, false};
  const std::vector<std::vector<std::string>> evectors = {
      {"w=this", "*initial*"},
      {"w=sentence"},
      {"w=is"},
      {"w=good"},
      {"w=.", "*ultimate*"}};
  for (size_t i = 0; i < 10; ++i) sbsm.Train(evectors, binomial_ys);
  std::vector<bool> binomial_yhats;
  sbsm.Predict(evectors, &binomial_yhats);
  AssertStructured(binomial_ys, binomial_yhats);
  sbsm.Average();
  sbsm.Predict(evectors, &binomial_yhats);
  AssertStructured(binomial_ys, binomial_yhats);

  // Sparse-dense multinomial; case-restoration (reusing the evectors and
  // transition functor from above).
  SparseDenseMultinomialSequentialModel sdmsm(32, N, 2, 1);
  const std::vector<size_t> dense_ys = {
      static_cast<size_t>(Case::TITLE), static_cast<size_t>(Case::LOWER),
      static_cast<size_t>(Case::LOWER), static_cast<size_t>(Case::LOWER),
      static_cast<size_t>(Case::DC)};
  for (size_t i = 0; i < 10; ++i) sdmsm.Train(evectors, dense_ys);
  std::vector<size_t> dense_yhats;
  sdmsm.Predict(evectors, &dense_yhats);
  AssertStructured(dense_ys, dense_yhats);
  sdmsm.Average();
  sdmsm.Predict(evectors, &dense_yhats);
  AssertStructured(dense_ys, dense_yhats);
}

}  // namespace
}  // namespace perceptronix

int main() {

  ::perceptronix::TestBinomial();
  ::perceptronix::TestMultinomial();
  ::perceptronix::TestStructured();

  std::cout << "Success!" << std::endl;

  return 0;
}
