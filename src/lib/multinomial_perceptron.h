// Copyright (C) 2015-2018 Kyle Gorman
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
// multinomial_perceptron.h: templates for multinomial perceptron
//     classifiers with binary features.

#ifndef PERCEPTRONIX_MULTINOMIAL_PERCEPTRON_H_
#define PERCEPTRONIX_MULTINOMIAL_PERCEPTRON_H_

#include <cassert>
#include <cstdint>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "linmod.pb.h"
#include "table.h"
#include "weight.h"

using std::string;

namespace perceptronix {

template <template <class> class OuterTableTpl, class WeightT>
class MultinomialPerceptronBaseTpl {
 public:
  using Table = OuterTableTpl<WeightT>;
  using InnerTable = typename Table::InnerTable;
  using Feature = typename Table::Feature;
  // Vectors of features may be large so we pass them by reference.
  using FeatureBundle = std::vector<Feature>;
  using Label = typename Table::Label;

  MultinomialPerceptronBaseTpl(size_t nfeats, size_t nlabels)
      : table_(nfeats, nlabels) {
    assert(nfeats > 0);
    assert(nlabels > 2);
  }

  Label Predict(const FeatureBundle &fb) const {
    std::unique_ptr<InnerTable> table(Score(fb));
    return table->ArgMax();
  }

  void Score(Feature f, InnerTable *inner) const {
    inner->AddWeights(table_[f]);
  }

  InnerTable *Score(Feature f) const {
    auto *inner = new InnerTable(InnerSize());
    Score(f, inner);
    return inner;
  }

  void Score(const FeatureBundle &fb, InnerTable *inner) const {
    for (const auto &f: fb) Score(f, inner);
  }

  InnerTable *Score(const FeatureBundle &fb) const {
    auto *inner = new InnerTable(InnerSize());
    Score(fb, inner);
    return inner;
  }

  size_t OuterSize() const { return table_.OuterSize(); }

  size_t InnerSize() const { return table_.InnerSize(); }

 protected:
  Table table_;
};

// Specialization with averaged weights.
template <template <class> class OuterTableTpl>
class MultinomialAveragedPerceptronTpl
    : public MultinomialPerceptronBaseTpl<OuterTableTpl, AveragedWeight> {
 public:
  using Base = MultinomialPerceptronBaseTpl<OuterTableTpl, AveragedWeight>;
  using Feature = typename Base::Feature;
  using FeatureBundle = typename Base::FeatureBundle;
  using Label = typename Base::Label;

  friend class MultinomialPerceptronBaseTpl<OuterTableTpl, Weight>;

  MultinomialAveragedPerceptronTpl(size_t nfeats, size_t nlabels,
                                   typename Weight::WeightType alpha = 1)
      : Base(nfeats, nlabels), alpha_(alpha), time_(0) {}

  using Base::Predict;
  using Base::Score;

  // Predicts a single example, updates if it is incorrectly labeled; then
  // updates the timer and returns a boolean indicating success or failure
  // (which callers may safely choose to ignore).
  bool Train(const FeatureBundle &fb, Label y) {
    const auto yhat = Predict(fb);
    const bool success = (yhat == y);
    if (!success) Update(fb, y, yhat);
    Tick();
    return success;
  }

  using Base::table_;

  // 1: Updates a single feature given correct and incorrect labels.
  void Update(const Feature &f, Label y, Label yhat) {
    auto &ref = table_[f];
    ref[y].Update(+alpha_, time_);
    ref[yhat].Update(-alpha_, time_);
  }

  // 2: Updates many features given correct and incorrect labels.
  void Update(const FeatureBundle &fb, Label y, Label yhat) {
    for (const auto &f: fb) {
      auto &ref = table_[f];
      ref[y].Update(+alpha_, time_);
      ref[yhat].Update(-alpha_, time_);
    }
  }

  uint64_t Time() const { return time_; }

 private:
  typename Weight::WeightType alpha_;
  uint64_t time_;

  // Advances the clock; invoked automatically by Train.
  void Tick() { ++time_; }
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
      MultinomialAveragedPerceptronTpl<OuterTableTpl> *avg);

  // Constructs model by deserializing.

  static MultinomialPerceptronTpl<OuterTableTpl> *Read(std::istream &istrm);

  static MultinomialPerceptronTpl<OuterTableTpl> *Read(const string &filename) {
    std::ifstream istrm(filename);
    return Read(istrm);
  }

  // Serializes the model.

  bool Write(std::ostream &ostrm, const string &metadata = "") const;

  bool Write(const string &filename, const string &metadata = "") const {
    std::ofstream ostrm(filename);
    return Write(ostrm, metadata);
  }
};

// Specializes the classifiers to use arrays for both inner and outer tables.

using DenseMultinomialPerceptron = MultinomialPerceptronTpl<DenseOuterTableTpl>;
using DenseMultinomialAveragedPerceptron =
    MultinomialAveragedPerceptronTpl<DenseOuterTableTpl>;

// Specializes the classifiers to use hash tables as the outer table, and arrays
// as the inner tables.

using SparseDenseMultinomialPerceptron =
    MultinomialPerceptronTpl<SparseDenseOuterTableTpl>;
using SparseDenseMultinomialAveragedPerceptron =
    MultinomialAveragedPerceptronTpl<SparseDenseOuterTableTpl>;

// Specializes the classifiers to use hash tables for both inner and outer 
// tables.

using SparseMultinomialPerceptron =
    MultinomialPerceptronTpl<SparseOuterTableTpl>;
using SparseMultinomialAveragedPerceptron =
    MultinomialAveragedPerceptronTpl<SparseOuterTableTpl>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_MULTINOMIAL_PERCEPTRON_H_
