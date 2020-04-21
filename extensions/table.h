#ifndef PERCEPTRONIX_TABLE_H_
#define PERCEPTRONIX_TABLE_H_

// table.h: template classes for tables of linear model weights.

#include <cstdint>

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <utility>
#include <valarray>
#include <vector>

namespace perceptronix {

// Inner table using an array.

template <class W>
class DenseInnerTableTpl {
 public:
  using Feature = size_t;
  using Weight = W;
  using Table = std::valarray<Weight>;
  using Iterator = decltype(std::begin(Table()));

  explicit DenseInnerTableTpl(size_t nfeats = 0) : table_(nfeats) {}

  Weight &operator[](Feature f) { return table_[f]; }

  const Weight &operator[](Feature f) const { return table_[f]; }

  size_t Size() const { return table_.size(); }

  Iterator begin() { return std::begin(table_); }

  const Iterator cbegin() const { return std::begin(table_); }

  Iterator end() { return std::end(table_); }

  const Iterator cend() const { return std::end(table_); }

  auto ArgMax() const { return std::distance(cbegin(), Max()); }

  auto Max() const { return std::max_element(cbegin(), cend()); }

  auto Margin() const {
    const auto max = Max();
    static auto cmp = [&max](const Weight &lhs, const Weight &rhs) {
       return rhs != *max && lhs < rhs;
    };
    return *max - *std::max_element(cbegin(), cend(), cmp);
  }

  void AddWeights(const DenseInnerTableTpl<Weight> &weights) {
    if (!weights.Size()) return;
    table_ += weights.table_;
  }

 private:
  Table table_;
};

// Inner table using a hash table.

template <class W>
class SparseInnerTableTpl {
 public:
  using Feature = std::string;
  using Weight = W;
  using Table = std::unordered_map<Feature, Weight>;
  using Iterator = typename Table::iterator;
  using ConstIterator = typename Table::const_iterator;
  using Pair = typename Table::value_type;

  explicit SparseInnerTableTpl(size_t nfeats = 0) : table_(nfeats) {}

  Weight &operator[](Feature f) { return table_[f]; }

  const Weight &operator[](Feature f) const {
    auto it = table_.find(f);
    if (it == table_.cend()) return kDefaultWeight;
    return it->second;
  }

  size_t Size() const { return table_.size(); }

  Iterator begin() { return table_.begin(); }

  ConstIterator cbegin() const { return table_.cbegin(); }

  Iterator end() { return table_.end(); }

  ConstIterator cend() const { return table_.cend(); }

  auto ArgMax() const {
    // The empty string is used as a place-keeper.
    if (table_.empty()) return Feature();
    constexpr static auto cmp = [](const Pair &lhs, const Pair &rhs){
      return lhs.second < rhs.second;
    };
    return std::max_element(cbegin(), cend(), cmp)->first;
  }

  auto Max() const {
    if (table_.empty()) return kDefaultWeight;
    constexpr static auto cmp = [](const Pair &lhs, const Pair &rhs){
      return lhs.second < rhs.second;
    };
    return std::max_element(cbegin(), cend(), cmp)->second;
  }

  auto Margin() const {
    const auto max = Max();
    static auto cmp = [&max](const Pair &lhs, const Pair &rhs) {
       return lhs.second != max && lhs.second < rhs.second;
    };
    return max - std::max_element(cbegin(), cend(), cmp)->second;
  }

  void AddWeights(const SparseInnerTableTpl<Weight> &weights) {
    if (!weights.Size()) return;
    for (auto it = weights.cbegin(); it != weights.cend(); ++it) {
      table_[it->first] += it->second;
    }
  }

 private:
  Table table_;

  static const Weight kDefaultWeight;
};

template <class Weight>
const Weight SparseInnerTableTpl<Weight>::kDefaultWeight{};

// Outer table using arrays.

template <class W>
class DenseOuterTableTpl {
 public:
  using Feature = size_t;
  using Label = size_t;
  using Weight = W;
  using InnerTable = DenseInnerTableTpl<Weight>;
  using Table = std::valarray<InnerTable>;
  using Iterator = decltype(std::begin(Table()));

  explicit DenseOuterTableTpl(size_t nfeats, size_t nlabels)
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
  size_t nlabels_;
};

// Outer table using hash table with an inner table using an array.
template <class W>
class SparseDenseOuterTableTpl {
 public:
  using Feature = std::string;
  using Label = size_t;
  using Weight = W;
  using InnerTable = DenseInnerTableTpl<Weight>;
  using Table = std::unordered_map<Feature, InnerTable>;
  using Iterator = typename Table::iterator;
  using ConstIterator = typename Table::const_iterator;

  explicit SparseDenseOuterTableTpl(size_t nfeats, size_t nlabels)
      : table_(nfeats), nlabels_(nlabels) {}

  InnerTable &operator[](Feature f) {
    auto it = table_.find(f);
    if (it == table_.end()) {
      table_.emplace(f, InnerSize());
      return table_[f];
    }
    return it->second;
  }

  const InnerTable &operator[](Feature f) const {
    auto it = table_.find(f);
    if (it == table_.cend()) return kDefaultInnerTable;
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
  size_t nlabels_;

  static const InnerTable kDefaultInnerTable;
};

template <class Weight>
const typename SparseDenseOuterTableTpl<Weight>::InnerTable
    SparseDenseOuterTableTpl<Weight>::kDefaultInnerTable{};

// Outer table using hash tables.

template <class W>
class SparseOuterTableTpl {
 public:
  using Feature = std::string;
  using Label = std::string;
  using Weight = W;
  using InnerTable = SparseInnerTableTpl<Weight>;
  using Table = std::unordered_map<Feature, InnerTable>;
  using Iterator = typename Table::iterator;
  using ConstIterator = typename Table::const_iterator;

  // Here, these are just hints for the initial sizes of the tables.
  explicit SparseOuterTableTpl(size_t nfeats, size_t nlabels)
      : table_(nfeats), nlabels_(nlabels) {}

  InnerTable &operator[](Feature f) {
    auto it = table_.find(f);
    if (it == table_.end()) {
      table_.emplace(f, InnerSize());
      return table_[f];
    }
    return it->second;
  }

  const InnerTable &operator[](Feature f) const {
    auto it = table_.find(f);
    if (it == table_.cend()) return kDefaultInnerTable;
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
  size_t nlabels_;

  static const InnerTable kDefaultInnerTable;
};

template <class Weight>
const typename SparseOuterTableTpl<Weight>::InnerTable
    SparseOuterTableTpl<Weight>::kDefaultInnerTable{};

}  // namespace perceptronix

#endif  // PERCEPTRONIX_TABLE_H_
