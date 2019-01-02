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
  using Label = bool;

  explicit BinomialPerceptronBaseTpl(size_t nfeats) : table_(nfeats) {
    assert(nfeats > 0);
  }

  void Score(Feature f, Weight *weight) const { *weight += table_[f]; }

  void Score(const FeatureBundle &fb, Weight *weight) const {
    for (const auto &f : fb) Score(f, weight);
  }

  Weight *Score(const FeatureBundle &fb) const {
    auto *weight = new Weight(bias_);
    Score(fb, weight);
    return weight;
  }

  Label Predict(const FeatureBundle &fb) const {
    std::unique_ptr<Weight> weight(Score(fb));
    return weight->Get() > 0;
  }

  size_t Size() const { return table_.Size(); }

 protected:
  Weight bias_;
  Table table_;
};

template <template <class> class InnerTableTpl>
class BinomialAveragingPerceptronTpl
    : public BinomialPerceptronBaseTpl<InnerTableTpl, AveragingWeight> {
 public:
  using Base = BinomialPerceptronBaseTpl<InnerTableTpl, AveragingWeight>;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;
  using Label = typename Base::Label;

  using Base::Predict;
  using Base::Score;

  using Base::bias_;
  using Base::table_;

  friend class BinomialPerceptronBaseTpl<InnerTableTpl, Weight>;

  explicit BinomialAveragingPerceptronTpl(size_t nfeats)
      : Base(nfeats), time_(0) {}

  // 1: Update a single feature given the correct label.
  void Update(Feature f, bool y) {
    table_[f].Update(y ? +1: -1, time_);
  }

  // 1': Same as (1) but with optional (useless) yhat argument.
  void Update(Feature f, bool y, bool yhat) { Update(f, y); }

  // 2: Updates many features given the correct label.
  void Update(const FeatureBundle &fb, bool y) {
    const auto tau = y ? +1 : -1;
    bias_.Update(tau, time_);
    for (const auto &f : fb) table_[f].Update(tau, time_);
  }

  // 2': Same as (2) but with optional (useless) yhat argument.
  void Update(const FeatureBundle &fb, bool y, bool yhat) { Update(fb, y); }

  // Predicts a single example, and updates if it is incorrectly labeled,
  // then updates the timer and returns a boolean indicating success or
  // failure (which callers may safely choose to ignore).
  bool Train(const FeatureBundle &fb, bool y) {
    bool success = (Predict(fb) == y);
    if (!success) Update(fb, y);
    Tick();
    return success;
  }

  // Advances the clock; invoked automatically by Train.
  void Tick(uint64_t step = 1) { time_ += step; }

  uint64_t Time() const { return time_; }

 private:
  uint64_t time_;
};

template <template <class> class InnerTableTpl>
class BinomialPerceptronTpl
    : public BinomialPerceptronBaseTpl<InnerTableTpl, Weight> {
 public:
  using Base = BinomialPerceptronBaseTpl<InnerTableTpl, Weight>;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;
  using Label = typename Base::Label;

  using Base::Predict;
  using Base::Score;

  using Base::bias_;
  using Base::table_;

  explicit BinomialPerceptronTpl(size_t nfeats, size_t nlabels)
      : Base(nfeats, nlabels) {}

  explicit BinomialPerceptronTpl(
      BinomialAveragingPerceptronTpl<InnerTableTpl> *avg);

  // Construct model by deserializing.

  static BinomialPerceptronTpl<InnerTableTpl> *Read(std::istream &istrm,
                                                    string *metadata = nullptr);

  static BinomialPerceptronTpl<InnerTableTpl> *Read(
  	const string &filename,
        string *metadata = nullptr) {
    std::ifstream istrm(filename);
    return Read(istrm, metadata);
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
using DenseBinomialAveragingPerceptron =
    BinomialAveragingPerceptronTpl<DenseInnerTableTpl>;

// Specializes the classifiers to use a hash table.

using SparseBinomialPerceptron = BinomialPerceptronTpl<SparseInnerTableTpl>;
using SparseBinomialAveragingPerceptron =
    BinomialAveragingPerceptronTpl<SparseInnerTableTpl>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_BINOMIAL_PERCEPTRON_H_
