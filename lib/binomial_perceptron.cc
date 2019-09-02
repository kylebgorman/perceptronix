// binomial_perceptron.cc: specializations for binomial_perceptron
// classifiers with binary features.

#include "binomial_perceptron.h"
#include "linear_model.pb.h"

namespace perceptronix {

// Specializations for DenseBinomialPerceptron.

template <>
DenseBinomialPerceptron::BinomialPerceptronTpl(
    DenseBinomialAveragingPerceptron *avg)
    : Base(avg->Size()) {
  const auto time = avg->Time();
  bias_.Set(avg->bias_.GetAverage(time));
  const size_t size = table_.Size();
  for (size_t i = 0; i < size; ++i) {
    table_[i].Set(avg->table_[i].GetAverage(time));
  }
}

template <>
DenseBinomialPerceptron *DenseBinomialPerceptron::Read(
    std::istream &istrm,
    std::string *metadata) {
  DenseBinomialPerceptronProto pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  if (metadata) *metadata = pb.metadata();
  const int size = pb.table_size();
  auto *model = new DenseBinomialPerceptron(size);
  model->bias_.Set(pb.bias());
  for (int i = 0; i < size; ++i) model->table_[i].Set(pb.table(i));
  return model;
}

template <>
bool DenseBinomialPerceptron::Write(std::ostream &ostrm,
                                    const std::string &metadata) const {
  DenseBinomialPerceptronProto pb;
  if (!metadata.empty()) pb.set_metadata(metadata);
  pb.set_bias(bias_.Get());
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    pb.add_table(it->Get());
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

template <>
SparseBinomialPerceptron::BinomialPerceptronTpl(
    SparseBinomialAveragingPerceptron *avg)
    : Base(avg->Size()) {
  const auto time = avg->Time();
  bias_.Set(avg->bias_.GetAverage(time));
  for (auto it = avg->table_.begin(); it != avg->table_.end(); ++it) {
    const auto weight = it->second.GetAverage(time);
    if (weight) table_[it->first].Set(weight);
  }
}

template <>
SparseBinomialPerceptron *SparseBinomialPerceptron::Read(
    std::istream &istrm,
    std::string *metadata) {
  SparseBinomialPerceptronProto pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  if (metadata) *metadata = pb.metadata();
  auto *model = new SparseBinomialPerceptron(pb.table_size());
  model->bias_.Set(pb.bias());
  auto pb_table = pb.table();
  auto &table = model->table_;
  for (auto it = pb_table.cbegin(); it != pb_table.cend(); ++it) {
    const auto &feature = it->first;
    table[feature].Set(pb_table[feature]);
  }
  return model;
}

template <>
bool SparseBinomialPerceptron::Write(std::ostream &ostrm,
                                     const std::string &metadata) const {
  SparseBinomialPerceptronProto pb;
  if (!metadata.empty()) pb.set_metadata(metadata);
  pb.set_bias(bias_.Get());
  auto *pb_table = pb.mutable_table();
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    (*pb_table)[it->first] = it->second.Get();
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

}  // namespace perceptronix
