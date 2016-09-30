// Copyright (C) 2015-2016 Kyle Gorman
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
// table.h: template classes for tables of linear model weights.

#ifndef PERCEPTRONIX_TABLE_H_
#define PERCEPTRONIX_TABLE_H_

#include <cstdint>

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <valarray>
#include <vector>

using std::string;

namespace perceptronix {

// Inner table using an array.

template <class WeightT>
class DenseInnerTableTpl {
 public:
  using Feature = size_t;
  using Weight = WeightT;
  using Table = std::valarray<WeightT>;
  using Iterator = decltype(std::begin(Table()));

  // DO NOT USE THIS. It's just to appease std::valarray.
  DenseInnerTableTpl() : table_(0) {}

  explicit DenseInnerTableTpl(uint32_t nfeats) : table_(nfeats) {}

  WeightT &operator[](Feature f) { return table_[f]; }

  const WeightT &operator[](Feature f) const { return table_[f]; }

  Iterator begin() { return std::begin(table_); }

  const Iterator cbegin() const { return std::begin(table_); }

  Iterator end() { return std::end(table_); }

  const Iterator cend() const { return std::end(table_); }

  size_t Size() const { return table_.size(); }

  auto ArgMax() const -> decltype(std::distance(Iterator(), Iterator())) {
    return std::distance(cbegin(), std::max_element(cbegin(), cend()));
  }

  void AddWeights(const DenseInnerTableTpl<WeightT> &weights) {
    if (!weights.Size()) return;
    table_ += weights.table_;
  }

 private:
  Table table_;
};

// Inner table using a hash table.

template <class WeightT>
class SparseInnerTableTpl {
 public:
  using Feature = string;
  using Weight = WeightT;
  using Table = std::unordered_map<Feature, WeightT>;
  using Iterator = typename Table::iterator;
  using ConstIterator = typename Table::const_iterator;
  using Pair = typename Table::value_type;

  // DO NOT USE THIS. It's just to appease std::unordered_map.
  SparseInnerTableTpl() : table_(0) {}

  // Here, this is just a hint for the initial size of the table.
  explicit SparseInnerTableTpl(uint32_t nfeats) : table_(nfeats) {}

  WeightT &operator[](Feature f) { return table_[f]; }

  const WeightT &operator[](Feature f) const {
    auto it = table_.find(f);
    if (it == table_.cend()) return default_weight_;
    return it->second;
  }

  size_t Size() const { return table_.size(); }

  Iterator begin() { return table_.begin(); }

  ConstIterator cbegin() const { return table_.cbegin(); }

  Iterator end() { return table_.end(); }

  ConstIterator cend() const { return table_.cend(); }

  Feature ArgMax() const {
    // The empty string is used as a place-keeper.
    if (table_.empty()) return Feature();
    static auto cmp = [](const Pair &lhs, const Pair &rhs) {
      return lhs.second < rhs.second;
    };
    return std::max_element(cbegin(), cend(), cmp)->first;
  }

  void AddWeights(const SparseInnerTableTpl<WeightT> &weights) {
    if (!weights.Size()) return;
    for (auto it = weights.cbegin(); it != weights.cend(); ++it)
      table_[it->first] += it->second;
  }

 private:
  Table table_;

  static const WeightT default_weight_;
};

template <class WeightT>
const WeightT SparseInnerTableTpl<WeightT>::default_weight_ = \
    WeightT();


// Outer table using arrays.

template <class WeightT>
class DenseOuterTableTpl {
 public:
  using Feature = size_t;
  using Label = size_t;
  using Weight = WeightT;
  using InnerTable = DenseInnerTableTpl<WeightT>;
  using Table = std::valarray<InnerTable>;
  using Iterator = decltype(std::begin(Table()));

  explicit DenseOuterTableTpl(uint32_t nfeats, uint32_t nlabels)
      : table_(InnerTable(nfeats), nlabels), nlabels_(nlabels) {}

  InnerTable &operator[](Feature f) { return table_[f]; }

  const InnerTable &operator[](Feature f) const { return table_[f]; }

  Iterator begin() { return std::begin(table_); }

  const Iterator cbegin() const { return std::begin(table_); }

  Iterator end() { return std::end(table_); }

  const Iterator cend() const { return std::end(table_); }

  size_t OuterSize() const { return table_.size(); }

  size_t InnerSize() const { return nlabels_; }

 private:
  Table table_;
  uint32_t nlabels_;
};

// Outer table using hash table with an inner table using an array.
template <class WeightT>
class SparseDenseOuterTableTpl {
 public:
  using Feature = string;
  using Label = size_t;
  using Weight = WeightT;
  using InnerTable = DenseInnerTableTpl<WeightT>;
  using Table = std::unordered_map<Feature, InnerTable>;
  using Iterator = typename Table::iterator;
  using ConstIterator = typename Table::const_iterator;

  // Here, nfeats is just a hint for the initial sizes of the hash table.
  explicit SparseDenseOuterTableTpl(uint32_t nfeats, uint32_t nlabels)
      : table_(nfeats), nlabels_(nlabels) {}

  InnerTable &operator[](Feature f) {
    auto it = table_.find(f);
    if (it == table_.end()) {
      table_.emplace(f, std::move(InnerTable(InnerSize())));
      return table_[f];
    }
    return it->second;
  }

  const InnerTable &operator[](Feature f) const {
    auto it = table_.find(f);
    if (it == table_.cend()) return default_inner_table_;
    return it->second;
  }

  Iterator begin() { return table_.begin(); }

  ConstIterator cbegin() const { return table_.cbegin(); }

  Iterator end() { return table_.end(); }

  ConstIterator cend() const { return table_.cend(); }

  size_t OuterSize() const { return table_.size(); }

  size_t InnerSize() const { return nlabels_; }

 private:
  Table table_;
  uint32_t nlabels_;

  static const InnerTable default_inner_table_;
};

template <class WeightT>
const typename SparseDenseOuterTableTpl<WeightT>::InnerTable \
    SparseDenseOuterTableTpl<WeightT>::default_inner_table_ = \
    SparseDenseOuterTableTpl<WeightT>::InnerTable();

// Outer table using hash tables.

template <class WeightT>
class SparseOuterTableTpl {
 public:
  using Feature = string;
  using Label = string;
  using Weight = WeightT;
  using InnerTable = SparseInnerTableTpl<WeightT>;
  using Table = std::unordered_map<Feature, InnerTable>;
  using Iterator = typename Table::iterator;
  using ConstIterator = typename Table::const_iterator;

  // Here, these are just hints for the initial sizes of the tables.
  explicit SparseOuterTableTpl(uint32_t nfeats, uint32_t nlabels)
      : table_(nfeats), nlabels_(nlabels) {}

  InnerTable &operator[](Feature f) {
    auto it = table_.find(f);
    if (it == table_.end()) {
      table_.emplace(f, std::move(InnerTable(InnerSize())));
      return table_[f];
    }
    return it->second;
  }

  const InnerTable &operator[](Feature f) const {
    auto it = table_.find(f);
    if (it == table_.cend()) return default_inner_table_;
    return it->second;
  }

  Iterator begin() { return table_.begin(); }

  const ConstIterator cbegin() const { return table_.cbegin(); }

  Iterator end() { return table_.end(); }

  const ConstIterator cend() const { return table_.cend(); }

  size_t OuterSize() const { return table_.size(); }

  size_t InnerSize() const { return nlabels_; }

 private:
  Table table_;
  uint32_t nlabels_;

  static const InnerTable default_inner_table_;
};

template <class WeightT>
const typename SparseOuterTableTpl<WeightT>::InnerTable \
    SparseOuterTableTpl<WeightT>::default_inner_table_ = \
    SparseOuterTableTpl<WeightT>::InnerTable();

}  // namespace perceptronix

#endif  // PERCEPTRONIX_TABLE_H_
