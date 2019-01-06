"""Cython header file for perceptronix C++ symbols."""


from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "perceptronix.h" namespace "perceptronix" nogil:

    # Models.

    cdef cppclass DenseBinomialModel:

        DenseBinomialModel(size_t)

        @staticmethod
        DenseBinomialModel *Read(const string &, string *)

        bool Write(const string &, const string &)

        bool Train(const vector[size_t] &, bool)

        bool Averaged()

        void Average()

        bool Predict(const vector[size_t] &)
    

    cdef cppclass SparseBinomialModel:

        SparseBinomialModel(size_t)

        @staticmethod
        SparseBinomialModel *Read(const string &, string *)

        bool Write(const string &, const string &)

        bool Train(const vector[string] &, bool)

        bool Averaged()

        void Average()

        bool Predict(const vector[string] &)


    cdef cppclass SparseBinomialSequentialModel:

        SparseBinomialSequentialModel(size_t, size_t)

        @staticmethod
        SparseBinomialSequentialModel *Read(const string &, size_t, string *)

        bool Write(const string &, const string &)

        bool Train(const vector[vector[string]] &, const vector[bool] &)

        bool Averaged()

        void Average()

        void Predict(const vector[vector[string]] &, vector[bool] *)


    cdef cppclass DenseMultinomialModel:
    
        DenseMultinomialModel(size_t, size_t)

        @staticmethod
        DenseMultinomialModel *Read(const string &, string *)

        bool Write(const string &, const string &)

        bool Train(const vector[size_t] &, size_t)

        bool Averaged()

        void Average()

        size_t Predict(const vector[size_t] &)


    cdef cppclass SparseDenseMultinomialModel:
    
        SparseDenseMultinomialModel(size_t, size_t)

        @staticmethod
        SparseDenseMultinomialModel *Read(const string &, string *)

        bool Write(const string &, const string &)

        bool Train(const vector[string] &, size_t)

        bool Averaged()

        void Average()

        size_t Predict(const vector[string] &)


    cdef cppclass SparseDenseMultinomialSequentialModel:

        SparseDenseMultinomialSequentialModel(size_t, size_t, size_t)

        @staticmethod
        SparseDenseMultinomialSequentialModel *Read(const string &,
                                                    size_t,
                                                    string *)

        bool Write(const string &, const string &)

        size_t Train(const vector[vector[string]] &evectors,
                     const vector[size_t] &ys)

        bool Averaged()

        void Average()

        void Predict(const vector[vector[string]] &evectors,
                     vector[size_t] *yhats)

    
    cdef cppclass SparseMultinomialModel:
    
        SparseMultinomialModel(size_t, size_t)

        @staticmethod
        SparseMultinomialModel *Read(const string &, string *)

        bool Write(const string &, const string &)

        bool Train(const vector[string] &, const string &)

        bool Averaged()

        void Average()

        string Predict(const vector[string] &)


    cdef cppclass SparseMultinomialSequentialModel:

        SparseMultinomialSequentialModel(size_t, size_t, size_t)

        @staticmethod
        SparseMultinomialSequentialModel *Read(const string &,
                                               size_t,
                                               string *)

        bool Write(const string &, const string &)

        size_t Train(const vector[vector[string]] &evectors,
                     const vector[string] &ys)

        bool Averaged()

        void Average()

        void Predict(const vector[vector[string]] &evectors,
                     vector[string] *yhats)
