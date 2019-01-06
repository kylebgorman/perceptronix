#ifndef PERCEPTRONIX_MULTINOMIAL_MODEL_H_
#define PERCEPTRONIX_MULTINOMIAL_MODEL_H_

// multinomial_model.h: wrappers for multinomial models.

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "multinomial_perceptron.h"
#include "decoder.h"

using std::string;

namespace perceptronix {

// Multinomial model wrapper.
template <class A, class P>
class MultinomialModel {
 public:
  using AveragingPerceptron = A;
  using Perceptron = P;

  using Label = typename Perceptron::Label;
  using Feature = typename Perceptron::Feature;
  using FeatureBundle = typename Perceptron::FeatureBundle;

  static_assert(std::is_same<Label,
                             typename AveragingPerceptron::Label>::value,
                "Label must be same type");
  static_assert(std::is_same<Feature,
                             typename AveragingPerceptron::Feature>::value,
                "Feature must be same type");
  static_assert(std::is_same<
      FeatureBundle, typename AveragingPerceptron::FeatureBundle>::value,
      "FeatureBundle must be same type");

  explicit MultinomialModel(size_t nfeats, size_t nlabels) :
    aperceptron_(new AveragingPerceptron(nfeats, nlabels)) {}

  // Deserialization.

  static MultinomialModel *Read(std::istream &istrm, string *metadata = nullptr) {
    return new MultinomialModel(Perceptron::Read(istrm, metadata));
  }

  static MultinomialModel *Read(const string &filename,
                             string *metadata = nullptr) {
    return new MultinomialModel(Perceptron::Read(filename, metadata));
  }
 
  // Serialization.
  
  bool Write(std::ostream &ostrm, const string &metadata = "") const {
    assert(Averaged());
    return perceptron_->Write(ostrm, metadata);
  }

  bool Write(const string &filename, const string &metadata = "") const {
    assert(Averaged());
    return perceptron_->Write(filename, metadata);
  }

  // Before averaging...
 
  bool Train(const FeatureBundle &fb, Label y) {
    assert(!Averaged());
    return aperceptron_->Train(fb, y);
  }

  void Average() {
    assert(!Averaged());
    perceptron_.reset(new Perceptron(aperceptron_.get()));
    aperceptron_.reset();
  }

  // At any time...

  Label Predict(const FeatureBundle &fb) const {
    return Averaged() ? perceptron_->Predict(fb) : aperceptron_->Predict(fb);
  }

  bool Averaged() const { return aperceptron_.get() == nullptr; }

 protected:
  MultinomialModel(Perceptron *perceptron) : perceptron_(perceptron) {}

  std::unique_ptr<AveragingPerceptron> aperceptron_;
  std::unique_ptr<const Perceptron> perceptron_;
};

// Specializations for the above.

using DenseMultinomialModel = MultinomialModel<
    DenseMultinomialAveragingPerceptron,
    DenseMultinomialPerceptron
>;
using SparseDenseMultinomialModel = MultinomialModel<
    SparseDenseMultinomialAveragingPerceptron,
    SparseDenseMultinomialPerceptron
>;
using SparseMultinomialModel = MultinomialModel<
    SparseMultinomialAveragingPerceptron,
    SparseMultinomialPerceptron
>;

// Sequential multinomial model wrapper.

template <class AveragingDecoder, class Decoder, class TransitionFunctor>
class MultinomialSequentialModel :
    public MultinomialModel<typename AveragingDecoder::Perceptron,
                         typename Decoder::Perceptron> {
 public:
  using Labels = typename Decoder::Labels;
  using Vectors = typename Decoder::Vectors;

  static_assert(std::is_same<Labels,
                             typename AveragingDecoder::Labels>::value,
                             "Labels must be same type");
  static_assert(std::is_same<Vectors,
                             typename AveragingDecoder::Vectors>::value,
                             "Vectors must be same type");

  using Base = MultinomialModel<typename AveragingDecoder::Perceptron,
                                typename Decoder::Perceptron>;

  using Perceptron = typename Base::Perceptron;

  using Base::Averaged;
  using Base::Write;

  using Base::aperceptron_;
  using Base::perceptron_;

  MultinomialSequentialModel(size_t nfeats, size_t nlabels, size_t order) :
     Base(nfeats, nlabels),
     tf_(order),
     adecoder_(new AveragingDecoder(aperceptron_.get(), tf_)) {}

  static MultinomialSequentialModel *Read(std::istream &istrm,
                                          size_t order,
                                          string *metadata = nullptr) {
    return new MultinomialSequentialModel(
        Base::Perceptron::Read(istrm, metadata), order);
  }

  static MultinomialSequentialModel *Read(const string &filename,
                                          size_t order,
                                          string *metadata = nullptr) {
    return new MultinomialSequentialModel(
        Base::Perceptron::Read(filename, metadata), order);
  }

  // Before averaging...

  // Returns the number of observations in the sequence correctly classified.
  size_t Train(const Vectors &evectors, const Labels &ys) {
    assert(!Base::Averaged());
    return adecoder_->Train(evectors, ys);
  }

  void Average() {
    Base::Average();
    decoder_.reset(new Decoder(*perceptron_, tf_));
    adecoder_.reset();
  }

  // At any time...

  void Predict(const Vectors &evectors, Labels *yhats) const {
    Averaged() ?
        decoder_->Predict(evectors, yhats) :
        adecoder_->Predict(evectors, yhats);
  }

 private:
  MultinomialSequentialModel(Perceptron *perceptron, size_t order) :
      Base(perceptron),
      tf_(order),
      decoder_(new Decoder(*perceptron_, tf_)) {}

  const TransitionFunctor tf_;
  std::unique_ptr<AveragingDecoder> adecoder_;
  std::unique_ptr<const Decoder> decoder_;
};

// Specializations for the above.

using SparseDenseMultinomialSequentialModel = MultinomialSequentialModel<
   SparseDenseMultinomialAveragingDecoder,
   SparseDenseMultinomialDecoder,
   SparseTransitionFunctor<typename SparseDenseMultinomialDecoder::Label>
>;

using SparseMultinomialSequentialModel = MultinomialSequentialModel<
   SparseMultinomialAveragingDecoder,
   SparseMultinomialDecoder,
   SparseTransitionFunctor<typename SparseMultinomialDecoder::Label>
>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_MULTINOMIAL_MODEL_H_
