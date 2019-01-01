"""Cython header file for perceptronix C++ symbols."""


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "perceptronix.h" namespace "perceptronix" nogil:

  cdef cppclass SparseTransitionFunctor[Label]:
    
    SparseTransitionFunctor(size_t)

    operator()(const vector[Label] &, vector[string] *)


  cdef cppclass DenseBinomialAveragedPerceptron:

    DenseBinomialAveragedPerceptron(size_t)

    bool Predict(const vector[size_t] &)

    bool Train(const vector[size_t] &, bool)


  cdef cppclass DenseBinomialPerceptron:

    DenseBinomialPerceptron(size_t)

    DenseBinomialPerceptron(DenseBinomialAveragedPerceptron *)

    bool Predict(const vector[size_t] &)

    @staticmethod
    DenseBinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass SparseBinomialAveragedPerceptron:

    SparseBinomialAveragedPerceptron(size_t)

    bool Predict(const vector[string] &)

    bool Train(const vector[string] &, bool)


  cdef cppclass SparseBinomialPerceptron:

    SparseBinomialPerceptron(size_t)

    SparseBinomialPerceptron(SparseBinomialAveragedPerceptron *)

    bool Predict(const vector[string] &)

    @staticmethod
    SparseBinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass DenseMultinomialAveragedPerceptron:

    DenseMultinomialAveragedPerceptron(size_t, size_t)

    size_t Predict(const vector[size_t] &)

    bool Train(const vector[size_t] &, size_t)


  cdef cppclass DenseMultinomialPerceptron:

    DenseMultinomialPerceptron(size_t, size_t)

    DenseMultinomialPerceptron(DenseMultinomialAveragedPerceptron *)

    size_t Predict(const vector[size_t] &)

    @staticmethod
    DenseMultinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass SparseDenseMultinomialAveragedPerceptron:

    SparseDenseMultinomialAveragedPerceptron(size_t, size_t)

    size_t Predict(const vector[string] &)

    bool Train(const vector[string] &, size_t)


  cdef cppclass SparseDenseMultinomialPerceptron:

    SparseDenseMultinomialPerceptron(size_t, size_t)

    SparseDenseMultinomialPerceptron(SparseDenseMultinomialAveragedPerceptron *)

    size_t Predict(const vector[string] &)

    @staticmethod
    SparseDenseMultinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)


  cdef cppclass SparseMultinomialAveragedPerceptron:

    SparseMultinomialAveragedPerceptron(size_t, size_t)

    string Predict(const vector[string] &)

    bool Train(const vector[string] &, string)


  cdef cppclass SparseMultinomialPerceptron:

    SparseMultinomialPerceptron(size_t, size_t)

    SparseMultinomialPerceptron(SparseMultinomialAveragedPerceptron *)

    string Predict(const vector[string] &)

    @staticmethod
    SparseMultinomialPerceptron *Read(const string &, string *)

    bool Write(const string &, const string &)
