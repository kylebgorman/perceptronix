#ifndef PERCEPTRONIX_BINOMIAL_PERCEPTRON_H_
#define PERCEPTRONIX_BINOMIAL_PERCEPTRON_H_

// binomial_perceptron.h: templates for binomial_perceptron classifiers
// with binary features.

#include <cassert>
#include <cmath>
#include <cstdint>

#include <fstream>
#include <string>
#include <vector>

#include "table.h"
#include "weight.h"

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

  Weight Score(const FeatureBundle &fb) const {
    Weight weight(bias_);
    for (const auto &f : fb) weight += table_[f];
    return weight;
  }

  Label Predict(const FeatureBundle &fb) const { return Score(fb).Get() > 0; }

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

  explicit BinomialAveragingPerceptronTpl(size_t nfeats, int c = 0)
      : Base(nfeats), c_(c), time_(0) {}

  // Predicts a single example, updates, and then returns a boolean indicating
  // success or failure (which callers may safely choose to ignore).
  bool Train(const FeatureBundle &fb, Label y) {
    const auto score = Score(fb);
    const bool yhat = score.Get() > 0;
    if (y != yhat) {
      Update(fb, y, yhat);
    } else if (c_ && static_cast<int>(std::abs(score.Get()) / fb.size()) < c_) {
      Update(fb, y, yhat);
    }
    Tick();
    return y == yhat;
  }

  // Advances the clock; invoked automatically by Train.
  void Tick(uint64_t step = 1) { time_ += step; }

  uint64_t Time() const { return time_; }

  // Updates many features given correct and incorrect labels.
  void Update(const FeatureBundle &fb, Label y, Label yhat) {
    const auto tau = y ? +1 : -1;
    bias_.Update(tau, time_);
    for (const auto &f : fb) table_[f].Update(tau, time_);
  }

 private:
  const int c_;
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

  explicit BinomialPerceptronTpl(
      BinomialAveragingPerceptronTpl<InnerTableTpl> *avg);

  // Construct model by deserializing.

  static BinomialPerceptronTpl<InnerTableTpl> *Read(
      std::istream &istrm, std::string *metadata = nullptr);

  static BinomialPerceptronTpl<InnerTableTpl> *Read(
      const std::string &filename, std::string *metadata = nullptr) {
    std::ifstream istrm(filename);
    return Read(istrm, metadata);
  }

  // Serializes the model.

  bool Write(std::ostream &ostrm, const std::string &metadata = "") const;

  bool Write(const std::string &filename,
             const std::string &metadata = "") const {
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
