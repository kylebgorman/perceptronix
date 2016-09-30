"""Cython header file for perceptronix C++ symbols."""


from libc.stdint cimport int32_t

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "perceptronix.h" namespace "perceptronix" nogil:

  cdef cppclass DenseBinomialAveragedPerceptron:

    DenseBinomialAveragedPerceptron(int32_t, float)

    bool Predict(vector[size_t] &)

    bool Train(vector[size_t] &, bool)


  cdef cppclass DenseBinomialPerceptron:

    DenseBinomialPerceptron(int32_t, float)

    DenseBinomialPerceptron(DenseBinomialAveragedPerceptron *)

    bool Predict(vector[size_t] &)

    @staticmethod
    DenseBinomialPerceptron *Read(const string &)

    bool Write(const string &)


  cdef cppclass SparseBinomialAveragedPerceptron:

    SparseBinomialAveragedPerceptron(int32_t, float)

    bool Predict(vector[string] &)

    bool Train(vector[string] &, bool)


  cdef cppclass SparseBinomialPerceptron:

    SparseBinomialPerceptron(int32_t, float)

    SparseBinomialPerceptron(SparseBinomialAveragedPerceptron *)

    bool Predict(vector[string] &)

    @staticmethod
    SparseBinomialPerceptron *Read(const string &)

    bool Write(const string &)


  cdef cppclass DenseMultinomialAveragedPerceptron:

    DenseMultinomialAveragedPerceptron(int32_t, int32_t, float)

    size_t Predict(vector[size_t])

    bool Train(vector[size_t], size_t)


  cdef cppclass DenseMultinomialPerceptron:

    DenseMultinomialPerceptron(int32_t, int32_t, float)

    DenseMultinomialPerceptron(DenseMultinomialAveragedPerceptron *)

    size_t Predict(vector[size_t])

    @staticmethod
    DenseMultinomialPerceptron *Read(const string &)

    bool Write(const string &)


  cdef cppclass SparseDenseMultinomialAveragedPerceptron:

    SparseDenseMultinomialAveragedPerceptron(int32_t, int32_t, float)

    size_t Predict(vector[string])

    bool Train(vector[string], size_t)


  cdef cppclass SparseDenseMultinomialPerceptron:

    SparseDenseMultinomialPerceptron(int32_t, int32_t, float)

    SparseDenseMultinomialPerceptron(SparseDenseMultinomialAveragedPerceptron *)

    size_t Predict(vector[string])

    @staticmethod
    SparseDenseMultinomialPerceptron *Read(const string &)

    bool Write(const string &)


  cdef cppclass SparseMultinomialAveragedPerceptron:

    SparseMultinomialAveragedPerceptron(int32_t, int32_t, float)

    string Predict(vector[string])

    bool Train(vector[string], string)


  cdef cppclass SparseMultinomialPerceptron:

    SparseMultinomialPerceptron(int32_t, int32_t, float)

    SparseMultinomialPerceptron(SparseMultinomialAveragedPerceptron *)

    string Predict(vector[string])

    @staticmethod
    SparseMultinomialPerceptron *Read(const string &)

    bool Write(const string &)
