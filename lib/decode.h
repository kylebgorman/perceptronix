#include <cassert>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

using std::string;

namespace perceptronix {

// TODO(kbg): This entire thing feels like an abstractional leak that ought to live
// lower in the stack. Consider adding this to (bi|multi)nomial_perceptron.h
// instead.

// Transition feature functors should have the following interface:
//
// template <class L, class F>
// struct TransitionFunctor {
//   void operator()(const std::vector<L> &labels,
//                   std::vector<F> *tfeats);
// };

// Transition feature functor for sparse (i.e., string) features. Should work with
// both sparse and dense (integral) labels.
template <class Label>
class SparseTransitionFunctor {
 public:  
  explicit SparseTransitionFunctor(size_t order) : order_(order) {}

  void operator()(const std::vector<Label> &labels,
                  std::vector<string> *tvector) const {
    tvector->clear();
    // Second condition is purely for compatibility.
    if (labels.empty() || order_ == 0) return;
    const auto size = labels.size();
    const auto bound = std::min(size, order_);
    tvector->reserve(bound);
    std::stringstream sstrm;
    sstrm << "t_i-1=" << labels[size - 1];
    tvector->emplace_back(sstrm.str());
    for (size_t i = 2; i <= bound; ++i) {
      // Here the feature conjunctions are in the reverse order from what you might
      // expect. We sacrifice readability for the ability to reuse the stringstream
      // buffer.
      sstrm << "^" << "t_i-" << i << "=" << labels[size - i];
      tvector->emplace_back(sstrm.str());
    }
  }
 
 private:
  const size_t order_;
};

// Performs greedy prediction.
//
// The caller provides emission vectors, a transition functor, and a classifier.
//
// The outputs include pointers to vectors for the combined features and a vector
// of predicted labels.
template <class Classifier, class TransitionFunctor>
void GreedyPredict(
    const std::vector<std::vector<typename Classifier::Feature>> &evectors,
    const TransitionFunctor &tfunctor,
    const Classifier &classifier,
    std::vector<std::vector<typename Classifier::Feature>> *cvectors,
    std::vector<typename Classifier::Label> *yhats) {
  const auto size = evectors.size();
  cvectors->clear();
  cvectors->resize(size);
  yhats->clear();
  yhats->reserve(size);
  for (size_t i = 0; i < size; ++i) {
    const auto &evector = evectors[i];
    auto &cvector = (*cvectors)[i];
    // Gets transition features.
    tfunctor(*yhats, &cvector);
    // Appends emission features to it.
    cvector.insert(cvector.end(), evector.begin(), evector.end());
    // Makes prediction.
    yhats->emplace_back(classifier.Predict(cvector));
  }
}

// Variant of greedy prediction that hides the combined feature vectors for when
// they are not needed (i.e., inference without training).
template <class Classifier, class TransitionFunctor>
void GreedyPredict(
    const std::vector<std::vector<typename Classifier::Feature>> &evectors,
    const TransitionFunctor &tfunctor,
    const Classifier &classifier,
    std::vector<typename Classifier::Label> *yhats) {
  std::vector<std::vector<typename Classifier::Feature>> cvectors;
  GreedyPredict(evectors, tfunctor, classifier, &cvectors, yhats);
}

// Performs binomial-model training with greedy decoding.
//

// Performs training with greedy decoding.
//
// The caller provides emission vectors, a transition functor, a label vector,
// and a pointer to the classifier. 
//
// The output is the number of correctly predicted labels.
template <class Classifier, class TransitionFunctor>
size_t GreedyTrain(
    const std::vector<std::vector<typename Classifier::Feature>> &evectors,
    const TransitionFunctor &tfunctor,
    const std::vector<typename Classifier::Label> &ys,
    Classifier *classifier) {
  const auto size = evectors.size();
  assert(size == ys.size());
  std::vector<std::vector<typename Classifier::Feature>> cvectors;
  std::vector<typename Classifier::Label> yhats;
  GreedyPredict(evectors, tfunctor, *classifier, &cvectors, &yhats);
  size_t correct = 0;
  for (size_t i = 0; i < size; ++i) {
    const auto &y = ys[i];;
    const auto &yhat = yhats[i];
    if (y == yhat) {
      correct += 1;
    } else {
      const auto &cvector = cvectors[i];
      // yhat is ignored in the binomial case.
      classifier->Update(cvector, y, yhat);
    }
  }
  classifier->Tick(size);
  return correct;
}

// TODO(kbg): Viterbi someday?

}  // namespace perceptronix
