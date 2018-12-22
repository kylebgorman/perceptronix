// multinomial_perceptron.cc: specializations for binomial_perceptron
// classifiers with binary features.

#include "multinomial_perceptron.h"

namespace perceptronix {

// Specializations for DenseMultinomialPerceptron.

template <>
DenseMultinomialPerceptron::MultinomialPerceptronTpl(
    DenseMultinomialAveragedPerceptron *avg)
    : Base(avg->OuterSize(), avg->InnerSize()) {
  const auto time = avg->Time();
  for (size_t i = 0; i < avg->OuterSize(); ++i) {
    auto &ref = table_[i];
    auto &avg_ref = avg->table_[i];
    for (size_t j = 0; j < avg->InnerSize(); ++j) {
      ref[j].Set(avg_ref[j].GetAverage(time));
    }
  }
}

template <>
DenseMultinomialPerceptron *DenseMultinomialPerceptron::Read(
    std::istream &istrm) {
  DenseMultinomialPerceptron_pb pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  const size_t outer_size = pb.table_size();
  const size_t inner_size = pb.inner_size();
  auto *model = new DenseMultinomialPerceptron(outer_size, inner_size);
  for (size_t i = 0; i < outer_size; ++i) {
    auto &inner_table = model->table_[i];
    const auto &inner_table_pb = pb.table(i);
    for (size_t j = 0; j < inner_size; ++j) {
      inner_table[j].Set(inner_table_pb.table(j));
    }
  }
  return model;
}

template <>
bool DenseMultinomialPerceptron::Write(std::ostream &ostrm,
                                       const string &metadata) const {
  DenseMultinomialPerceptron_pb pb;
  pb.set_metadata(metadata);
  pb.set_inner_size(InnerSize());
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    auto inner_table_pb = pb.add_table();
    for (auto iit = it->cbegin(); iit != it->cend(); ++iit) {
      inner_table_pb->add_table(iit->Get());
    }
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

// Specializations for SparseMultinomialPerceptron.

template <>
SparseDenseMultinomialPerceptron::MultinomialPerceptronTpl(
    SparseDenseMultinomialAveragedPerceptron *avg)
    : Base(avg->OuterSize(), avg->InnerSize()) {
  const auto time = avg->Time();
  for (auto it = avg->table_.begin(); it != avg->table_.end(); ++it) {
    auto &ref = table_[it->first];
    for (size_t j = 0; j < InnerSize(); ++j) {
      ref[j].Set(it->second[j].GetAverage(time));
    }
  }
}

template <>
SparseDenseMultinomialPerceptron *SparseDenseMultinomialPerceptron::Read(
    std::istream &istrm) {
  SparseDenseMultinomialPerceptron_pb pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  const size_t inner_size = pb.inner_size();
  auto *model = new SparseDenseMultinomialPerceptron(pb.table_size(),
                                                     inner_size);
  auto outer_table_pb = pb.table();
  for (auto it = outer_table_pb.cbegin(); it != outer_table_pb.cend(); ++it) {
    auto &inner_table = model->table_[it->first];
    const auto &inner_table_pb = outer_table_pb[it->first];
    for (size_t j = 0; j < inner_size; ++j) {
      inner_table[j].Set(inner_table_pb.table(j));
    }
  }
  return model;
}

template <>
bool SparseDenseMultinomialPerceptron::Write(std::ostream &ostrm,
                                             const string &metadata) const {
  SparseDenseMultinomialPerceptron_pb pb;
  pb.set_metadata(metadata);
  pb.set_inner_size(InnerSize());
  auto *outer_table_pb = pb.mutable_table();
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    auto *inner_table_pb = (*outer_table_pb)[it->first].mutable_table();
    for (size_t j = 0; j < InnerSize(); ++j) {
      inner_table_pb->Add(it->second[j].Get());
    }
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

// Specializations for SparseMultinomialPerceptron.

template <>
SparseMultinomialPerceptron::MultinomialPerceptronTpl(
    SparseMultinomialAveragedPerceptron *avg)
    : Base(avg->OuterSize(), avg->InnerSize()) {
  const size_t time = avg->Time();
  for (auto it = avg->table_.begin(); it != avg->table_.end(); ++it) {
    auto &ref = table_[it->first];
    for (auto iit = it->second.begin(); iit != it->second.end(); ++iit) {
      const auto &label = iit->first;
      // Ignores the reserved empty string label.
      if (label.empty()) continue;
      ref[label].Set(iit->second.GetAverage(time));
    }
  }
}

template <>
SparseMultinomialPerceptron *SparseMultinomialPerceptron::Read(
    std::istream &istrm) {
  SparseMultinomialPerceptron_pb pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  auto *model = new SparseMultinomialPerceptron(pb.table_size(),
                                                pb.inner_size());
  auto outer_table_pb = pb.table();
  for (auto it = outer_table_pb.cbegin(); it != outer_table_pb.cend(); ++it) {
    const auto &feature = it->first;
    auto &inner_table = model->table_[feature];
    const auto &inner_table_pb = outer_table_pb[feature].table();
    for (auto iit = inner_table_pb.cbegin(); iit != inner_table_pb.cend();
         ++iit) {
      inner_table[iit->first].Set(iit->second);
    }
  }
  return model;
}

template <>
bool SparseMultinomialPerceptron::Write(std::ostream &ostrm,
                                        const string &metadata) const {
  SparseMultinomialPerceptron_pb pb;
  pb.set_metadata(metadata);
  pb.set_inner_size(InnerSize());
  auto *outer_table_pb = pb.mutable_table();
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    auto *inner_table_pb = (*outer_table_pb)[it->first].mutable_table();
    for (auto iit = it->second.cbegin(); iit != it->second.cend(); ++iit) {
      const auto &label = iit->first;
      // Ignores the reserved empty string label.
      if (label.empty()) continue;
      (*inner_table_pb)[label] = iit->second.Get();
    }
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

}  // namespace perceptronix
