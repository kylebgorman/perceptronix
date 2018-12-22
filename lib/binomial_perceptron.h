// binomial_perceptron.h: templates for binomial_perceptron classifiers
// with binary features.

#ifndef PERCEPTRONIX_BINOMIAL_PERCEPTRON_H_
#define PERCEPTRONIX_BINOMIAL_PERCEPTRON_H_

#include <cassert>
#include <cstdint>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "table.h"
#include "weight.h"

using std::string;

namespace perceptronix {

template <template <class> class InnerTableTpl, class Weight>
class BinomialPerceptronBaseTpl {
 public:
  using Table = InnerTableTpl<Weight>;
  using Feature = typename Table::Feature;
  using FeatureBundle = std::vector<Feature>;

  explicit BinomialPerceptronBaseTpl(size_t nfeats) : table_(nfeats) {
    assert(nfeats > 0);
  }

  void Score(Feature f, Weight *weight) const { *weight += table_[f]; }

  Weight *Score(Feature f) const {
    auto *weight = new Weight();
    Score(f, weight);
    return weight;
  }

  void Score(const FeatureBundle &fb, Weight *weight) const {
    for (const auto &f : fb) Score(f, weight);
  }

  Weight *Score(const FeatureBundle &fb) const {
    auto *weight = new Weight();
    Score(fb, weight);
    return weight;
  }

  bool Predict(const FeatureBundle &fb) const {
    std::unique_ptr<Weight> weight(Score(fb));
    return weight->Get() > 0;
  }

  size_t Size() const { return table_.Size(); }

 protected:
  Table table_;
};

template <template <class> class InnerTableTpl>
class BinomialAveragedPerceptronTpl
    : public BinomialPerceptronBaseTpl<InnerTableTpl, AveragedWeight> {
 public:
  using Base = BinomialPerceptronBaseTpl<InnerTableTpl, AveragedWeight>;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;

  using Base::Predict;
  using Base::Score;

  using Base::table_;

  friend class BinomialPerceptronBaseTpl<InnerTableTpl, Weight>;

  explicit BinomialAveragedPerceptronTpl(size_t nfeats,
                                         Weight::WeightType alpha = 1.)
      : Base(nfeats), alpha_(alpha), time_(0) {}

  // 1: Update a single feature given the correct label.
  void Update(Feature f, bool y) {
    table_[f].Update(y ? +alpha_ : -alpha_, time_);
  }

  // 2: Updates many features given the correct label.
  void Update(const FeatureBundle &fb, bool y) {
    const auto tau = y ? +alpha_ : -alpha_;
    for (const auto &f : fb) table_[f].Update(tau, time_);
  }

  // Predicts a single example,and updates if it is incorrectly labeled,
  // then updates the timer and returns a boolean indicating success or
  // failure (which callers may safely choose to ignore).
  bool Train(const FeatureBundle &fb, bool y) {
    bool success = (Predict(fb) == y);
    if (!success) Update(fb, y);
    Tick();
    return success;
  }

  uint64_t Time() const { return time_; }

 private:
  Weight::WeightType alpha_;
  uint64_t time_;

  // Advances the clock; invoked automatically by Train.
  inline void Tick() { ++time_; }
};

template <template <class> class InnerTableTpl>
class BinomialPerceptronTpl
    : public BinomialPerceptronBaseTpl<InnerTableTpl, Weight> {
 public:
  using Base = BinomialPerceptronBaseTpl<InnerTableTpl, Weight>;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;

  using Base::Predict;
  using Base::Score;

  using Base::table_;

  explicit BinomialPerceptronTpl(size_t nfeats, size_t nlabels)
      : Base(nfeats, nlabels) {}

  explicit BinomialPerceptronTpl(
      BinomialAveragedPerceptronTpl<InnerTableTpl> *avg);

  // Construct model by deserializing.

  static BinomialPerceptronTpl<InnerTableTpl> *Read(std::istream &istrm);

  static BinomialPerceptronTpl<InnerTableTpl> *Read(const string &filename) {
    std::ifstream istrm(filename);
    return Read(istrm);
  }

  // Serializes the model.

  bool Write(std::ostream &ostrm, const string &metadata = "") const;

  bool Write(const string &filename, const string &metadata = "") const {
    std::ofstream ostrm(filename);
    return Write(ostrm, metadata);
  }

 private:
  explicit BinomialPerceptronTpl(size_t nfeats) : Base(nfeats) {}
};

// Specializes the classifiers to use an array.

using DenseBinomialPerceptron = BinomialPerceptronTpl<DenseInnerTableTpl>;
using DenseBinomialAveragedPerceptron =
    BinomialAveragedPerceptronTpl<DenseInnerTableTpl>;

// Specializes the classifiers to use a hash table.

using SparseBinomialPerceptron = BinomialPerceptronTpl<SparseInnerTableTpl>;
using SparseBinomialAveragedPerceptron =
    BinomialAveragedPerceptronTpl<SparseInnerTableTpl>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_BINOMIAL_PERCEPTRON_H_
