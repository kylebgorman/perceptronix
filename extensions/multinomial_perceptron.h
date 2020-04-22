#ifndef PERCEPTRONIX_MULTINOMIAL_PERCEPTRON_H_
#define PERCEPTRONIX_MULTINOMIAL_PERCEPTRON_H_

// multinomial_perceptron.h: templates for multinomial perceptron classifiers.

#include <cassert>
#include <cstdint>

#include <fstream>
#include <string>
#include <vector>

#include "table.h"
#include "weight.h"

namespace perceptronix {

template <template <class> class OuterTableTpl, class Weight>
class MultinomialPerceptronBaseTpl {
 public:
  using Table = OuterTableTpl<Weight>;
  using InnerTable = typename Table::InnerTable;
  using Feature = typename Table::Feature;
  // Vectors of features may be large so we pass them by reference.
  using FeatureBundle = std::vector<Feature>;
  using Label = typename Table::Label;

  MultinomialPerceptronBaseTpl(size_t nfeats, size_t nlabels)
      : bias_(nlabels), table_(nfeats, nlabels) {
    assert(nfeats > 0);
    assert(nlabels > 2);
  }

  Label Predict(const FeatureBundle &fb) const {
    const InnerTable inner(Score(fb));
    return inner.ArgMax();
  }

  InnerTable Score(const FeatureBundle &fb) const {
    InnerTable inner(bias_);
    for (const auto &f : fb) inner.AddWeights(table_[f]);
    return inner;
  }

  size_t OuterSize() const { return table_.OuterSize(); }

  size_t InnerSize() const { return table_.InnerSize(); }

 protected:
  InnerTable bias_;
  Table table_;
};

// Specialization with averaged weights.
template <template <class> class OuterTableTpl>
class MultinomialAveragingPerceptronTpl
    : public MultinomialPerceptronBaseTpl<OuterTableTpl, AveragingWeight> {
 public:
  using Base = MultinomialPerceptronBaseTpl<OuterTableTpl, AveragingWeight>;
  using InnerTable = typename Base::InnerTable;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;
  using Label = typename Base::Label;

  friend class MultinomialPerceptronBaseTpl<OuterTableTpl, Weight>;

  MultinomialAveragingPerceptronTpl(size_t nfeats, size_t nlabels, int c = 0)
      : Base(nfeats, nlabels), c_(c), time_(0) {}

  using Base::Predict;
  using Base::Score;

  using Base::bias_;
  using Base::table_;

  // Predicts a single example, updates, and then returns a boolean indicating
  // success or failure (which callers may safely choose to ignore).
  bool Train(const FeatureBundle &fb, Label y) {
    const auto scores = Score(fb);
    const Label yhat = scores.ArgMax();
    if (y != yhat) {
      Update(fb, y, yhat);
    } else if (c_) {
      const auto &yhat_score = scores[yhat];
      const auto &y_score = scores[y];
      const auto margin = yhat_score.Get() - y_score.Get();
      if (static_cast<int>(margin / fb.size()) < c_) Update(fb, y, yhat);
    }
    Tick();
    return y == yhat;
  }

  // Advances the clock; invoked automatically by Train.
  void Tick(uint64_t step = 1) { time_ += step; }

  uint64_t Time() const { return time_; }

  // Updates many features given correct and incorrect labels.
  void Update(const FeatureBundle &fb, Label y, Label yhat) {
    bias_[y].Update(+1, time_);
    bias_[yhat].Update(-1, time_);
    for (const auto &f : fb) {
      auto &ref = table_[f];
      ref[y].Update(+1, time_);
      ref[yhat].Update(-1, time_);
    }
  }

 private:
  const int c_;
  uint64_t time_;
};

template <template <class> class OuterTableTpl>
class MultinomialPerceptronTpl
    : public MultinomialPerceptronBaseTpl<OuterTableTpl, Weight> {
 public:
  using Base = MultinomialPerceptronBaseTpl<OuterTableTpl, Weight>;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;
  using Label = typename Base::Label;

  using Base::Predict;
  using Base::Score;

  MultinomialPerceptronTpl(size_t nfeats, size_t nlabels)
      : Base(nfeats, nlabels) {}

  // Constructs model from averaged model.

  explicit MultinomialPerceptronTpl(
      MultinomialAveragingPerceptronTpl<OuterTableTpl> *avg);

  // Constructs model by deserializing.

  static MultinomialPerceptronTpl<OuterTableTpl> *Read(
      std::istream &istrm, std::string *metadata = nullptr);

  static MultinomialPerceptronTpl<OuterTableTpl> *Read(
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
};

// Specializes the classifiers to use arrays for both inner and outer tables.

using DenseMultinomialPerceptron = MultinomialPerceptronTpl<DenseOuterTableTpl>;
using DenseMultinomialAveragingPerceptron =
    MultinomialAveragingPerceptronTpl<DenseOuterTableTpl>;

// Specializes the classifiers to use hash tables as the outer table, and arrays
// as the inner tables.

using SparseDenseMultinomialPerceptron =
    MultinomialPerceptronTpl<SparseDenseOuterTableTpl>;
using SparseDenseMultinomialAveragingPerceptron =
    MultinomialAveragingPerceptronTpl<SparseDenseOuterTableTpl>;

// Specializes the classifiers to use hash tables for both inner and outer
// tables.

using SparseMultinomialPerceptron =
    MultinomialPerceptronTpl<SparseOuterTableTpl>;
using SparseMultinomialAveragingPerceptron =
    MultinomialAveragingPerceptronTpl<SparseOuterTableTpl>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_MULTINOMIAL_PERCEPTRON_H_
