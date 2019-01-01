"""Cython header file for perceptronix C++ symbols."""


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "perceptronix.h" namespace "perceptronix" nogil:


    # Models.


    cdef cppclass DenseBinomialAveragingPerceptron:

        DenseBinomialAveragingPerceptron(size_t)

        bool Predict(const vector[size_t] &)

        bool Train(const vector[size_t] &, bool)


    cdef cppclass DenseBinomialPerceptron:

        DenseBinomialPerceptron(size_t)

        DenseBinomialPerceptron(DenseBinomialAveragingPerceptron *)

        bool Predict(const vector[size_t] &)

        @staticmethod
        DenseBinomialPerceptron *Read(const string &, string *)

        bool Write(const string &, const string &)


    cdef cppclass SparseBinomialAveragingPerceptron:

        SparseBinomialAveragingPerceptron(size_t)

        bool Predict(const vector[string] &)

        bool Train(const vector[string] &, bool)


    cdef cppclass SparseBinomialPerceptron:

        SparseBinomialPerceptron(size_t)

        SparseBinomialPerceptron(SparseBinomialAveragingPerceptron *)

        bool Predict(const vector[string] &)

        @staticmethod
        SparseBinomialPerceptron *Read(const string &, string *)

        bool Write(const string &, const string &)


    cdef cppclass DenseMultinomialAveragingPerceptron:

        DenseMultinomialAveragingPerceptron(size_t, size_t)

        size_t Predict(const vector[size_t] &)

        bool Train(const vector[size_t] &, size_t)


    cdef cppclass DenseMultinomialPerceptron:

        DenseMultinomialPerceptron(size_t, size_t)

        DenseMultinomialPerceptron(DenseMultinomialAveragingPerceptron *)

        size_t Predict(const vector[size_t] &)

        @staticmethod
        DenseMultinomialPerceptron *Read(const string &, string *)

        bool Write(const string &, const string &)


    cdef cppclass SparseDenseMultinomialAveragingPerceptron:

        SparseDenseMultinomialAveragingPerceptron(size_t, size_t)

        size_t Predict(const vector[string] &)

        bool Train(const vector[string] &, size_t)


    cdef cppclass SparseDenseMultinomialPerceptron:

        SparseDenseMultinomialPerceptron(size_t, size_t)

        SparseDenseMultinomialPerceptron(
            SparseDenseMultinomialAveragingPerceptron *)

        size_t Predict(const vector[string] &)

        @staticmethod
        SparseDenseMultinomialPerceptron *Read(const string &, string *)

        bool Write(const string &, const string &)


    cdef cppclass SparseMultinomialAveragingPerceptron:

        SparseMultinomialAveragingPerceptron(size_t, size_t)

        string Predict(const vector[string] &)

        bool Train(const vector[string] &, string)


    cdef cppclass SparseMultinomialPerceptron:

        SparseMultinomialPerceptron(size_t, size_t)

        SparseMultinomialPerceptron(SparseMultinomialAveragingPerceptron *)

        string Predict(const vector[string] &)

        @staticmethod
        SparseMultinomialPerceptron *Read(const string &, string *)

        bool Write(const string &, const string &)


    # Decoding.


    cdef cppclass SparseTransitionFunctor[Label]:

        SparseTransitionFunctor(size_t)


    cdef cppclass SparseBinomialDecoder:

        SparseBinomialDecoder(const SparseBinomialPerceptron &,
                              const SparseTransitionFunctor[bool] &)

        void Predict(const vector[vector[string]] &,
                     vector[bool] *)


    cdef cppclass SparseBinomialAveragingDecoder:

        SparseBinomialAveragingDecoder(SparseBinomialAveragingPerceptron *,
                                      const SparseTransitionFunctor[bool] &)

        void Predict(const vector[vector[string]] &,
                     vector[bool] *)

        size_t Train(const vector[vector[string]] &,
                     const vector[bool] &)


    cdef cppclass SparseDenseMultinomialDecoder:

        SparseDenseMultinomialDecoder(const SparseDenseMultinomialPerceptron &,
                                      const SparseTransitionFunctor[size_t] &)

        void Predict(const vector[vector[string]] &,
                     vector[size_t] *)


    cdef cppclass SparseDenseMultinomialAveragingDecoder:

        SparseDenseMultinomialAveragingDecoder(
            SparseDenseMultinomialAveragingPerceptron *,
            const SparseTransitionFunctor[size_t] &)

        void Predict(const vector[vector[string]] &,
                     vector[size_t] *)

        size_t Train(const vector[vector[string]] &,
                     const vector[size_t] &)


    cdef cppclass SparseMultinomialDecoder:

        SparseMultinomialDecoder(const SparseMultinomialPerceptron &,
                                 const SparseTransitionFunctor[string] &)

        void Predict(const vector[vector[string]] &,
                     vector[string] *)


    cdef cppclass SparseMultinomialAveragingDecoder:

        SparseMultinomialAveragingDecoder(
            SparseMultinomialAveragingPerceptron *,
            const SparseTransitionFunctor[string] &)

        void Predict(const vector[vector[string]] &,
                     vector[string] *)

        size_t Train(const vector[vector[string]] &,
                     const vector[string] &)
