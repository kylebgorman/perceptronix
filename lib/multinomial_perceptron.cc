// multinomial_perceptron.cc: specializations for binomial_perceptron
// classifiers with binary features.

#include "multinomial_perceptron.h"
#include "linear_model.pb.h"

namespace perceptronix {

// Specializations for DenseMultinomialPerceptron.

template <>
DenseMultinomialPerceptron::MultinomialPerceptronTpl(
    DenseMultinomialAveragingPerceptron *avg)
    : Base(avg->OuterSize(), avg->InnerSize()) {
  const auto time = avg->Time();
  const auto outer_size = avg->OuterSize();
  const auto inner_size = avg->InnerSize();
  for (size_t j = 0; j < inner_size; ++j) {
    auto &ref = bias_[j];
    auto &avg_ref = avg->bias_[j];
    ref.Set(avg_ref.GetAverage(time));
  }
  for (size_t i = 0; i < outer_size; ++i) {
    auto &ref = table_[i];
    auto &avg_ref = avg->table_[i];
    for (size_t j = 0; j < inner_size; ++j) {
      ref[j].Set(avg_ref[j].GetAverage(time));
    }
  }
}

template <>
DenseMultinomialPerceptron *DenseMultinomialPerceptron::Read(
    std::istream &istrm, string *metadata) {
  DenseMultinomialPerceptronProto pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  const size_t outer_size = pb.table_size();
  const size_t inner_size = pb.inner_size();
  if (metadata) *metadata = pb.metadata();
  auto *model = new DenseMultinomialPerceptron(outer_size, inner_size);
  for (size_t j = 0; j < inner_size; ++j) {
    model->bias_[j].Set(pb.bias().table(j));
  }
  for (size_t i = 0; i < outer_size; ++i) {
    auto &inner_table = model->table_[i];
    const auto &inner_tableProto = pb.table(i);
    for (size_t j = 0; j < inner_size; ++j) {
      inner_table[j].Set(inner_tableProto.table(j));
    }
  }
  return model;
}

template <>
bool DenseMultinomialPerceptron::Write(std::ostream &ostrm,
                                       const string &metadata) const {
  const auto outer_size = OuterSize();
  const auto inner_size = InnerSize();
  DenseMultinomialPerceptronProto pb;
  if (!metadata.empty()) pb.set_metadata(metadata);
  pb.set_inner_size(inner_size);
  auto *bias = pb.mutable_bias();
  for (size_t j = 0; j < inner_size; ++j) bias->add_table(bias_[j].Get());
  for (size_t i = 0; i < outer_size; ++i) {
    const auto &inner_table = table_[i];
    auto *inner_table_proto = pb.add_table();
    for (size_t j = 0; j < inner_size; ++j) {
      inner_table_proto->add_table(inner_table[j].Get());
    }
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

// Specializations for SparseMultinomialPerceptron.

template <>
SparseDenseMultinomialPerceptron::MultinomialPerceptronTpl(
    SparseDenseMultinomialAveragingPerceptron *avg)
    : Base(avg->OuterSize(), avg->InnerSize()) {
  const auto time = avg->Time();
  const auto inner_size = InnerSize();
  for (size_t j = 0; j < inner_size; ++j) {
    auto &ref = bias_[j];
    auto &avg_ref = avg->bias_[j];
    ref.Set(avg_ref.GetAverage(time));
  }
  for (auto it = avg->table_.begin(); it != avg->table_.end(); ++it) {
    auto &ref = table_[it->first];
    for (size_t j = 0; j < inner_size; ++j) {
      ref[j].Set(it->second[j].GetAverage(time));
    }
  }
}

template <>
SparseDenseMultinomialPerceptron *SparseDenseMultinomialPerceptron::Read(
    std::istream &istrm, string *metadata) {
  SparseDenseMultinomialPerceptronProto pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  const auto inner_size = pb.inner_size();
  if (metadata) *metadata = pb.metadata();
  auto *model =
      new SparseDenseMultinomialPerceptron(pb.table_size(), inner_size);
  for (size_t j = 0; j < inner_size; ++j) {
    model->bias_[j].Set(pb.bias().table(j));
  }
  auto outer_table_proto = pb.table();
  for (auto it = outer_table_proto.cbegin(); it != outer_table_proto.cend();
       ++it) {
    auto &inner_table = model->table_[it->first];
    const auto &inner_table_proto = outer_table_proto[it->first];
    for (size_t j = 0; j < inner_size; ++j) {
      inner_table[j].Set(inner_table_proto.table(j));
    }
  }
  return model;
}

template <>
bool SparseDenseMultinomialPerceptron::Write(std::ostream &ostrm,
                                             const string &metadata) const {
  const auto inner_size = InnerSize();
  SparseDenseMultinomialPerceptronProto pb;
  if (!metadata.empty()) pb.set_metadata(metadata);
  pb.set_inner_size(inner_size);
  auto *bias = pb.mutable_bias();
  for (size_t j = 0; j < inner_size; ++j) bias->add_table(bias_[j].Get());
  auto *outer_table_proto = pb.mutable_table();
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    auto *inner_table_proto = (*outer_table_proto)[it->first].mutable_table();
    for (size_t j = 0; j < inner_size; ++j) {
      inner_table_proto->Add(it->second[j].Get());
    }
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

// Specializations for SparseMultinomialPerceptron.

template <>
SparseMultinomialPerceptron::MultinomialPerceptronTpl(
    SparseMultinomialAveragingPerceptron *avg)
    : Base(avg->OuterSize(), avg->InnerSize()) {
  const auto time = avg->Time();
  for (auto iit = avg->bias_.begin(); iit != avg->bias_.end(); ++iit) {
    const auto &label = iit->first;
    // Ignores the reserved empty string label.
    if (label.empty()) continue;
    bias_[label].Set(iit->second.GetAverage(time));
  }
  for (auto it = avg->table_.begin(); it != avg->table_.end(); ++it) {
    auto &ref = table_[it->first];
    for (auto iit = it->second.begin(); iit != it->second.end(); ++iit) {
      const auto &label = iit->first;
      if (label.empty()) continue;
      ref[label].Set(iit->second.GetAverage(time));
    }
  }
}

template <>
SparseMultinomialPerceptron *SparseMultinomialPerceptron::Read(
    std::istream &istrm, string *metadata) {
  SparseMultinomialPerceptronProto pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  if (metadata) *metadata = pb.metadata();
  auto *model =
      new SparseMultinomialPerceptron(pb.table_size(), pb.inner_size());
  auto bias = pb.bias().table();
  for (auto iit = bias.cbegin(); iit != bias.cend(); ++iit) {
    model->bias_[iit->first].Set(iit->second);
  }
  auto outer_table_proto = pb.table();
  for (auto it = outer_table_proto.cbegin(); it != outer_table_proto.cend();
       ++it) {
    const auto &feature = it->first;
    auto &inner_table = model->table_[feature];
    const auto &inner_table_proto = outer_table_proto[feature].table();
    for (auto iit = inner_table_proto.cbegin(); iit != inner_table_proto.cend();
         ++iit) {
      inner_table[iit->first].Set(iit->second);
    }
  }
  return model;
}

template <>
bool SparseMultinomialPerceptron::Write(std::ostream &ostrm,
                                        const string &metadata) const {
  SparseMultinomialPerceptronProto pb;
  if (!metadata.empty()) pb.set_metadata(metadata);
  pb.set_inner_size(InnerSize());
  auto *inner_table_proto = pb.mutable_bias()->mutable_table();
  for (auto iit = bias_.cbegin(); iit != bias_.cend(); ++iit) {
    (*inner_table_proto)[iit->first] = iit->second.Get();
  }
  auto *outer_table_proto = pb.mutable_table();
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    auto *inner_table_proto = (*outer_table_proto)[it->first].mutable_table();
    for (auto iit = it->second.cbegin(); iit != it->second.cend(); ++iit) {
      const auto &label = iit->first;
      // Ignores the reserved empty string label.
      if (label.empty()) continue;
      (*inner_table_proto)[label] = iit->second.Get();
    }
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

}  // namespace perceptronix
