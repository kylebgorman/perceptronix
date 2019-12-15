#ifndef PERCEPTRONIX_BINOMIAL_MODEL_H_
#define PERCEPTRONIX_BINOMIAL_MODEL_H_

// binomial_model.h: wrappers for binomial models.

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>

#include "binomial_perceptron.h"
#include "decoder.h"

namespace perceptronix {

// Binomial model wrapper.
template <class A, class P>
class BinomialModel {
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

  explicit BinomialModel(size_t nfeats) :
      aperceptron_(new AveragingPerceptron(nfeats)) {}

  // Deserialization.

  static BinomialModel *Read(std::istream &istrm,
                             std::string *metadata = nullptr) {
    return new BinomialModel(Perceptron::Read(istrm, metadata));
  }

  static BinomialModel *Read(const std::string &filename,
                             std::string *metadata = nullptr) {
    return new BinomialModel(Perceptron::Read(filename, metadata));
  }

  // Serialization.

  bool Write(std::ostream &ostrm, const std::string &metadata = "") const {
    assert(Averaged());
    return perceptron_->Write(ostrm, metadata);
  }
 
  bool Write(const std::string &filename, 
             const std::string &metadata = "") const {
    assert(Averaged());
    return perceptron_->Write(filename, metadata);
  }

  // Before averaging...

  bool Train(const FeatureBundle &fb, bool y) {
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
  BinomialModel(Perceptron *perceptron) : perceptron_(perceptron) {}

  std::unique_ptr<AveragingPerceptron> aperceptron_;
  std::unique_ptr<const Perceptron> perceptron_;
};

// Specializations for the above.

using DenseBinomialModel = BinomialModel<DenseBinomialAveragingPerceptron,
                                         DenseBinomialPerceptron>;
using SparseBinomialModel = BinomialModel<SparseBinomialAveragingPerceptron,
                                          SparseBinomialPerceptron>;

// Sequential binomial model wrapper.

template <class AveragingDecoder, class Decoder, class TransitionFunctor>
class BinomialSequentialModel :
    public BinomialModel<typename AveragingDecoder::Perceptron,
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

  using Base = BinomialModel<typename AveragingDecoder::Perceptron,
                              typename Decoder::Perceptron>;

  using Perceptron = typename Base::Perceptron;

  using Base::Averaged;
  using Base::Write;

  using Base::aperceptron_;
  using Base::perceptron_;

  BinomialSequentialModel(size_t nfeats, size_t order) :
       Base(nfeats),
       tf_(order),
       adecoder_(new AveragingDecoder(aperceptron_.get(), tf_)) {}

  static BinomialSequentialModel *Read(std::istream &istrm, 
                                       size_t order,
                                       std::string *metadata = nullptr) {
    return new BinomialSequentialModel(Base::Perceptron::Read(istrm, metadata),
                                       order);
  }

  static BinomialSequentialModel *Read(const std::string &filename,
                                       size_t order,
                                       std::string *metadata = nullptr) {
    return new BinomialSequentialModel(
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
  BinomialSequentialModel(Perceptron *perceptron, size_t order) : 
        Base(perceptron),
        tf_(order),
        decoder_(new Decoder(*perceptron_, tf_)) {}

  const TransitionFunctor tf_;
  std::unique_ptr<AveragingDecoder> adecoder_;
  std::unique_ptr<const Decoder> decoder_;
};

// Specialization for the above.

using SparseBinomialSequentialModel = BinomialSequentialModel<
    SparseBinomialAveragingDecoder,
    SparseBinomialDecoder,
    SparseTransitionFunctor<typename SparseBinomialAveragingDecoder::Label>
>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_BINOMIAL_MODEL_H_
