// Copyright (c) 2015-2016 Kyle Gorman
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
// weight.h: Weight class templates for perceptron classifiers.
//
// The base class Weight(Tpl) is just a number with methods. The derived
// class AveragedWeight(Tpl) also holds the averaged weight, in a delayed,
// overflow-resistent form. Since AveragedWeightTpl contains about three
// machine words, it should usually be passed by reference.
//
// When training an averaged perceptron, the normal workflow is to use the
// methods of AveragedWeight(Tpl) during training, and then to finalize
// the model by creating new Weight(Tpl) instances like so:
//
//     uint64_t t = 0;
//     AveragedWeight aw(0);
//     // ...
//     // Many rounds of training using aw.Get() and aw.Update().
//     // ...
//     Weight w(aw.GetAveragedWeight(t));

#ifndef PERCEPTRONIX_WEIGHT_H_
#define PERCEPTRONIX_WEIGHT_H_

namespace perceptronix {

// Concrete base class for weights.

template <class T>
class WeightTpl {
 public:
  using WeightType = T;

  explicit WeightTpl(WeightType weight = 0.) : weight_(weight) {}

  WeightType Get() const { return weight_; }

  void Set(WeightType weight) { weight_ = weight; }

  void Update(WeightType tau) { weight_ += tau; }

  bool operator<(const WeightTpl<T> &rhs) const {
    return weight_ < rhs.weight_;
  }

  WeightTpl &operator+=(WeightTpl<T> rhs) {
    weight_ += rhs.weight_;
    return *this;
  }

  WeightTpl &operator-=(WeightTpl<T> rhs) {
    weight_ -= rhs.weight_;
    return *this;
  }

  WeightTpl &operator*=(WeightTpl<T> rhs) {
    weight_ *= rhs.weight_;
    return *this;
  }

  WeightTpl &operator/=(WeightTpl<T> rhs) {
    weight_ /= rhs.weight_;
    return *this;
  }

 protected:
  WeightType weight_;
};

template <class T>
WeightTpl<T> operator+(WeightTpl<T> lhs, WeightTpl<T> rhs) {
  return lhs += rhs;
}

template <class T>
WeightTpl<T> operator-(WeightTpl<T> lhs, WeightTpl<T> rhs) {
  return lhs -= rhs;
}

template <class T>
WeightTpl<T> operator*(WeightTpl<T> lhs, WeightTpl<T> rhs) {
  return lhs *= rhs;
}

template <class T>
WeightTpl<T> operator/(WeightTpl<T> lhs, WeightTpl<T> rhs) {
  return lhs /= rhs;
}

// Weight with online mean estimation using Welford's algorithm:
//
// B.P. Welford. 1962. Note on a method for calculating corrected sums of
// squares and products. Technometrics 4(3): 419-420.
//
// The inherited weight is the one to be used during training, and can be
// accessed using Get(). GetAverage(time) applies any queued updates, then
// returns the averaged weight.

template <class T>
class AveragedWeightTpl : public WeightTpl<T> {
 public:
  using WeightType = T;

  // Extends the base constructor. All weights are averaged as if they were
  // initialized at 0 at time 0; time is zero-based and may not be negative.
  explicit AveragedWeightTpl(WeightType weight = 0., uint64_t time = 0)
      : WeightTpl<T>(weight), aweight_(0.), time_(time) {
    Freshen(time);
  }

  using WeightTpl<T>::weight_;

  // Implements the online mean.
  void Freshen(uint64_t time) {
    while (time_ < time) {
      aweight_ += (weight_ - aweight_) / (time_ + 1);
      ++time_;
    }
  }

  void Update(WeightType tau, uint64_t time) {
    Freshen(time);
    WeightTpl<T>::Update(tau);
  }

  // This function should be used for retrieving a copy of the final weight,
  // but not for inference during training time; during training, call Get()
  // instead.
  WeightType GetAverage(uint64_t time) {
    Freshen(time);
    return aweight_;
  }

 protected:
  WeightType aweight_;  // The averaged weight.
  uint64_t time_;
};

// Default specializations.

using Weight = WeightTpl<float>;

using AveragedWeight = AveragedWeightTpl<float>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_WEIGHT_H_
