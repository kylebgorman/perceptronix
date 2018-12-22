// Copyright (C) 2015-2018 Kyle Gorman
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
