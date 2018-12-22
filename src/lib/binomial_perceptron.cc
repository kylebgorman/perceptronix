// binomial_perceptron.cc: specializations for binomial_perceptron
// classifiers with binary features.

#include "binomial_perceptron.h"
#include "linmod.pb.h"

namespace perceptronix {

// Specializations for DenseBinomialPerceptron.

template <>
DenseBinomialPerceptron::BinomialPerceptronTpl(
    DenseBinomialAveragedPerceptron *avg)
    : Base(avg->Size()) {
  const size_t size = table_.Size();
  for (size_t i = 0; i < size; ++i) {
    table_[i].Set(avg->table_[i].GetAverage(avg->Time()));
  }
}

template <>
DenseBinomialPerceptron *DenseBinomialPerceptron::Read(std::istream &istrm) {
  DenseBinomialPerceptron_pb pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  const int size = pb.table_size();
  auto *model = new DenseBinomialPerceptron(size);
  for (int i = 0; i < size; ++i) model->table_[i].Set(pb.table(i));
  return model;
}

template <>
bool DenseBinomialPerceptron::Write(std::ostream &ostrm,
                                    const string &metadata) const {
  DenseBinomialPerceptron_pb pb;
  pb.set_metadata(metadata);
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    pb.add_table(it->Get());
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

template <>
SparseBinomialPerceptron::BinomialPerceptronTpl(
    SparseBinomialAveragedPerceptron *avg)
    : Base(avg->Size()) {
  for (auto it = avg->table_.begin(); it != avg->table_.end(); ++it) {
    const auto weight = it->second.GetAverage(avg->Time());
    if (weight) table_[it->first].Set(weight);
  }
}

template <>
SparseBinomialPerceptron *SparseBinomialPerceptron::Read(std::istream &istrm) {
  SparseBinomialPerceptron_pb pb;
  if (!pb.ParseFromIstream(&istrm)) return nullptr;
  auto *model = new SparseBinomialPerceptron(pb.table_size());
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
                                     const string &metadata) const {
  SparseBinomialPerceptron_pb pb;
  pb.set_metadata(metadata);
  auto *pb_table = pb.mutable_table();
  for (auto it = table_.cbegin(); it != table_.cend(); ++it) {
    (*pb_table)[it->first] = it->second.Get();
  }
  return pb.SerializeToOstream(&ostrm) && ostrm.good();
}

}  // namespace perceptronix
