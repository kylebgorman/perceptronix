"""Cython header file for perceptronix C++ symbols."""


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "perceptronix.h" namespace "perceptronix" nogil:

  cdef cppclass DenseBinomialAveragedPerceptron:

    DenseBinomialAveragedPerceptron(size_t, float)

    bool Predict(const vector[size_t] &)

    bool Train(const vector[size_t] &, bool)


  cdef cppclass DenseBinomialPerceptron:

    DenseBinomialPerceptron(size_t, float)

    DenseBinomialPerceptron(DenseBinomialAveragedPerceptron *)

    bool Predict(const vector[size_t] &)

    @staticmethod
    DenseBinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass SparseBinomialAveragedPerceptron:

    SparseBinomialAveragedPerceptron(size_t, float)

    bool Predict(const vector[string] &)

    bool Train(const vector[string] &, bool)


  cdef cppclass SparseBinomialPerceptron:

    SparseBinomialPerceptron(size_t, float)

    SparseBinomialPerceptron(SparseBinomialAveragedPerceptron *)

    bool Predict(const vector[string] &)

    @staticmethod
    SparseBinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass DenseMultinomialAveragedPerceptron:

    DenseMultinomialAveragedPerceptron(size_t, size_t, float)

    size_t Predict(const vector[size_t] &)

    bool Train(const vector[size_t] &, size_t)


  cdef cppclass DenseMultinomialPerceptron:

    DenseMultinomialPerceptron(size_t, size_t, float)

    DenseMultinomialPerceptron(DenseMultinomialAveragedPerceptron *)

    size_t Predict(const vector[size_t] &)

    @staticmethod
    DenseMultinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass SparseDenseMultinomialAveragedPerceptron:

    SparseDenseMultinomialAveragedPerceptron(size_t, size_t, float)

    size_t Predict(const vector[string] &)

    bool Train(const vector[string] &, size_t)


  cdef cppclass SparseDenseMultinomialPerceptron:

    SparseDenseMultinomialPerceptron(size_t, size_t, float)

    SparseDenseMultinomialPerceptron(SparseDenseMultinomialAveragedPerceptron *)

    size_t Predict(const vector[string] &)

    @staticmethod
    SparseDenseMultinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass SparseMultinomialAveragedPerceptron:

    SparseMultinomialAveragedPerceptron(size_t, size_t, float)

    string Predict(const vector[string] &)

    bool Train(const vector[string] &, string)


  cdef cppclass SparseMultinomialPerceptron:

    SparseMultinomialPerceptron(size_t, size_t, float)

    SparseMultinomialPerceptron(SparseMultinomialAveragedPerceptron *)

    string Predict(const vector[string] &)

    @staticmethod
    SparseMultinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)
