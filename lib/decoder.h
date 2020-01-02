#ifndef PERCEPTRONIX_DECODER_H_
#define PERCEPTRONIX_DECODER_H_

// decoder.h: decoding functions and classes.

#include <cassert>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "binomial_perceptron.h"
#include "multinomial_perceptron.h"

namespace perceptronix {

// Transition feature functors should have the following interface:
//
// template <class L, class F>
// struct TransitionFunctor {
//   void operator()(const std::vector<L> &labels,
//                   std::vector<F> *tfeats);
// };

// TODO(kbg): Come up with a DenseTransitionFunctor, somehow.

// Transition feature functor for sparse (i.e., string) features. Should work
// with both sparse and dense (integral) labels.
template <class Label>
class SparseTransitionFunctor {
 public:
  explicit SparseTransitionFunctor(size_t order) : order_(order) {}

  void operator()(const std::vector<Label> &labels,
                  std::vector<std::string> *tvector) const {
    tvector->clear();
    // Second condition is purely for compatibility.
    if (labels.empty() || order_ == 0) return;
    const auto size = labels.size();
    const auto bound = std::min(order_, size);
    tvector->reserve(bound);
    std::stringstream sstrm;
    sstrm << "t_i-1=" << labels[size - 1];
    tvector->emplace_back(sstrm.str());
    for (size_t i = 2; i <= bound; ++i) {
      // Here the feature conjunctions are in the reverse order from what you
      // might expect. We sacrifice readability for the ability to reuse the
      // stringstream buffer.
      sstrm << "^"
            << "t_i-" << i << "=" << labels[size - i];
      tvector->emplace_back(sstrm.str());
    }
  }

 private:
  const size_t order_;
};

// Base greedy decoder, without training functionality.
template <class P, class F>
class Decoder {
 public:
  using Perceptron = P;
  using Label = typename P::Label;
  using Feature = typename P::Feature;
  using TransitionFunctor = F;

  using Labels = std::vector<Label>;
  using Vectors = std::vector<std::vector<Feature>>;

  Decoder(const Perceptron &perceptron, const TransitionFunctor &tfunctor)
      : perceptron_(perceptron), tfunctor_(tfunctor) {}

  // Performs greedy prediction.
  void Predict(const Vectors &evectors, Labels *yhats) const {
    Vectors cvectors;
    Predict(evectors, &cvectors, yhats);
  }

  // Same but exposes the cvectors.
  void Predict(const Vectors &evectors, Vectors *cvectors,
               Labels *yhats) const {
    const auto size = evectors.size();
    cvectors->clear();
    cvectors->resize(size);
    yhats->clear();
    yhats->reserve(size);
    for (size_t i = 0; i < size; ++i) {
      const auto &evector = evectors[i];
      auto &cvector = (*cvectors)[i];
      // Gets transition features.
      tfunctor_(*yhats, &cvector);
      // Appends emission features to it.
      cvector.insert(cvector.end(), evector.begin(), evector.end());
      // Makes prediction.
      yhats->emplace_back(perceptron_.Predict(cvector));
    }
  }

 private:
  const Perceptron &perceptron_;
  const TransitionFunctor &tfunctor_;
};

// Specializations of the above

using SparseBinomialDecoder =
    Decoder<SparseBinomialPerceptron,
            SparseTransitionFunctor<typename SparseBinomialPerceptron::Label>>;
using SparseDenseMultinomialDecoder = Decoder<
    SparseDenseMultinomialPerceptron,
    SparseTransitionFunctor<typename SparseDenseMultinomialPerceptron::Label>>;
using SparseMultinomialDecoder = Decoder<
    SparseMultinomialPerceptron,
    SparseTransitionFunctor<typename SparseMultinomialPerceptron::Label>>;

// Enhanced greedy decoder, with training functionality.
template <class P, class F>
class AveragingDecoder {
 public:
  using Perceptron = P;
  using Label = typename P::Label;
  using Feature = typename P::Feature;
  using TransitionFunctor = F;

  using Labels = std::vector<Label>;
  using Vectors = std::vector<std::vector<Feature>>;

  AveragingDecoder(Perceptron *perceptron, const TransitionFunctor &tfunctor)
      : base_(*perceptron, tfunctor), perceptron_(perceptron) {}

  // Performs greedy prediction.
  void Predict(const Vectors &evectors, Labels *yhat) const {
    base_.Predict(evectors, yhat);
  }

  // Performs greedy training and returns the number of correct classifications.
  size_t Train(const Vectors &evectors, const Labels &ys) {
    const auto size = ys.size();
    assert(size == evectors.size());
    Vectors cvectors;
    Labels yhats;
    base_.Predict(evectors, &cvectors, &yhats);
    size_t correct = 0;
    for (size_t i = 0; i < size; ++i) {
      const auto &y = ys[i];
      const auto &yhat = yhats[i];
      if (y == yhat) {
        correct += 1;
      } else {
        perceptron_->Update(cvectors[i], y, yhat);
      }
    }
    perceptron_->Tick(size);
    return correct;
  }

 private:
  const Decoder<Perceptron, TransitionFunctor> base_;
  Perceptron *perceptron_;
};

// Specializations of the above.

using SparseBinomialAveragingDecoder = AveragingDecoder<
    SparseBinomialAveragingPerceptron,
    SparseTransitionFunctor<typename SparseBinomialAveragingPerceptron::Label>>;
using SparseDenseMultinomialAveragingDecoder = AveragingDecoder<
    SparseDenseMultinomialAveragingPerceptron,
    SparseTransitionFunctor<
        typename SparseDenseMultinomialAveragingPerceptron::Label>>;
using SparseMultinomialAveragingDecoder =
    AveragingDecoder<SparseMultinomialAveragingPerceptron,
                     SparseTransitionFunctor<
                         typename SparseMultinomialAveragingPerceptron::Label>>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_DECODER_H_
