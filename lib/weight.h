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

  explicit WeightTpl(WeightType weight = 0) : weight_(weight) {}

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

// Weight with a running average.
//
// The inherited weight is the one to be used during training, and can be
// accessed using Get(). GetAverage(time) applies any queued updates, then
// returns the averaged weight.

template <class T>
class AveragedWeightTpl : public WeightTpl<T> {
 public:
  using WeightType = T;

  using Base = WeightTpl<T>;

  // Extends the base constructor. All weights are averaged as if they were
  // initialized at 0 at time 0.
  explicit AveragedWeightTpl(WeightType weight = 0, uint64_t time = 0)
      : Base(weight), summed_weight_(weight), time_(time) {
    Freshen(time);
  }

  using Base::Get;

  // Implements the mean operation.
  void Freshen(uint64_t time) {
    const auto elapsed = time - time_;
    summed_weight_ += elapsed * Base::Get();
    time_ = time;
  }

  void Update(WeightType tau, uint64_t time) {
    Freshen(time);
    Base::Update(tau);
  }

  // Used to retrieve the average of the final weight after training is
  // complete.
  float GetAverage(uint64_t time) {
    Freshen(time);
    return static_cast<double>(summed_weight_) / time_;
  }

 protected:
  WeightType summed_weight_;
  uint64_t time_;
};

// Default specializations.

// This is naturally a float because it is produced by averaging.
using Weight = WeightTpl<float>;
// Whereas this is naturally integral.
using AveragedWeight = AveragedWeightTpl<int32_t>;

}  // namespace perceptronix

#endif  // PERCEPTRONIX_WEIGHT_H_
