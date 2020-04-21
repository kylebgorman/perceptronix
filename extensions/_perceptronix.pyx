#cython: language_level=3, c_string_type=str, c_string_encoding=utf8
"""Python interface to Perceptronix.

Each instance is initially an averaging model; calling the instance method
`average` "finalizes" the weights (by calling the averaging constructor
of the unaveraged class), disabling further training and freeing the
averaging model's memory.
"""

from cython.operator cimport address as addr
from cython.operator cimport dereference as deref

cimport cperceptronix as cc

from libcpp cimport bool
from libcpp.memory cimport make_unique
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


# Helper for converting strings.


cdef string tobytes(data) except *:
    if isinstance(data, bytes):
        return data
    try:
        return data.encode("utf8")
    except Exception:
        raise ValueError(f"Cannot encode {data} as a bytestring")


# Helpers for encoding vectors.


cdef vector[string] tobytevector(svector):
    return [tobytes(s) for s in svector]


cdef vector[vector[string]] tobytevectors(svectors):
    return [[tobytes(s) for s in svector] for svector in svectors]


# Custom exception classes.


class PerceptronixIOError(IOError):

    pass


class PerceptronixOpError(RuntimeError):

    pass


# User-facing classes.


cdef class DenseBinomialModel(object):

    """
    DenseBinomialModel(nfeats, c = 0)

    Args:
        nfeats: Maximum number of unique features.
        c: Margin constant (default: 0).

    Binomial linear classifier backed by a fixed-size contiguous array of
    weights.

    This class provides the simplest form of a binomial linear classifier. At
    construction time, clients specify the maximum number of unique features.
    During training and inference, users pass features as iterables of
    non-negative integers.
    """

    cdef unique_ptr[cc.DenseBinomialModel] _model

    def __init__(self, size_t nfeats, int c = 0):
        self._model = make_unique[cc.DenseBinomialModel](nfeats, c)

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    @classmethod
    def read(cls, filename):
        """
        DenseBinomialModel.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef DenseBinomialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(cc.DenseBinomialModel.Read(tobytes(filename),
                                                       addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef bool train(self, feats, bool label) except *:
        """
        train(feats, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature vector and a boolean label for the observation.

        Args:
            feats: An iterable of non-negative integer features for the
                observation.
            label: The boolean label for the observation.

        Returns:
            Whether the instance was correctly classified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[size_t] fvector = feats
        return self._model.get().Train(fvector, label)

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Average()

    cpdef bool predict(self, feats):
        """
        predict(feats)

        Predicts the label for an observation.

        Args:
            feats: An iterable of non-negative integer features for the
                observation.

        Returns:
            The prediction.
        """
        cdef vector[size_t] fvector = feats
        return self._model.get().Predict(fvector)


cdef class SparseBinomialModel(object):

    """
    SparseBinomialModel(nfeats, c = 0)

    Args:
        nfeats: Hint for the number of unique features.
        c: Margin constant (default: 0).

    Binomial linear classifier backed by a hash table of weights.

    This class provides a more flexible binomial linear classifier than
    DenseBinomialModel. At construction time, clients specify a hint for
    the number of features, which is used to compute the initial size for the
    hash table; unlike DenseBinomialModel, however, this is not a hard
    constraint. During training and inference, users pass features as iterables
    of strings.
    """

    cdef unique_ptr[cc.SparseBinomialModel] _model

    def __init__(self, size_t nfeats, int c = 0):
        self._model = make_unique[cc.SparseBinomialModel](nfeats, c)

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    @classmethod
    def read(cls, filename):
        """
        SparseBinomialModel.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseBinomialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(cc.SparseBinomialModel.Read(tobytes(filename),
                                                        addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef bool train(self, feats, bool label) except *:
        """
        train(feats, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature vector and a boolean label for the observation.

        Args:
            feats: An iterable of string features for the observation.
            label: The boolean label for the observation.

        Returns:
            Whether the instance was correctly labeled.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[string] fvector = tobytevector(feats)
        return self._model.get().Train(fvector, label)

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Average()

    cpdef bool predict(self, feats):
        """
        predict(feats)

        Predicts the label for an observation.

        Args:
            feats: An iterable of string features for the observation.

        Returns:
             Boolean prediction.
        """
        cdef vector[string] fvector = tobytevector(feats)
        return self._model.get().Predict(fvector)


cdef class SparseBinomialSequentialModel:

    """
    SparseBinomialSequentialModel(nfeats, order, c = 0)

    Args:
        nfeats: Hint for the number of unique features.
        order: Model order (e.g., 1 implies bigram model).
        c: Margin constant (default: 0).
        
    Binomial linear classifier backed by a hash table of weights and greedy
    sequential decoding.
    """

    cdef unique_ptr[cc.SparseBinomialSequentialModel] _model

    def __init__(self, size_t nfeats, size_t order, int c = 0):
        self._model = make_unique[cc.SparseBinomialSequentialModel](
            nfeats, order, c
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseBinomialSequentialModel.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order (e.g., 1 implies bigram model).

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseBinomialSequentialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            cc.SparseBinomialSequentialModel.Read(tobytes(filename),
                                                  order,
                                                  addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef size_t train(self, efeats, labels) except *:
        """
        train(efeats, labels)

        Trains model using a single labeled sequence.

        This method trains the internal model on a single labeled sequence, in
        which each observation consists of a feature vector and a boolean label.

        Args:
            efeats: An iterable of string emission features for each
                observation.
            labels: An iterable of boolean labels for each observation.

        Returns:
            The number of observations in the sequence correctly classsified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        return self._model.get().Train(fvectors, labels)

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Average()

    cpdef vector[bool] predict(self, efeats):
        """
        predict(efeats)

        Predicts the labels for a sequence.

        Args:
            efeats: An iterable of string emission features for each
                observation.
        
        Returns:
            A list of the predicted labels.
        """
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[bool] yhats
        self._model.get().Predict(fvectors, addr(yhats))
        return yhats


cdef class DenseMultinomialModel(object):

    """
    DenseMultiomialModel(nfeats, nlabels, c = 0)

    Args:
        nfeats: Maximum number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).
        c: Margin constant (default: 0).

    Multinomial linear classifier backed by fixed-size contiguous arrays of
    weights.

    This class provides the simplest form of a multinomial linear classifier. At
    construction time, clients specify the maximum number of unique features
    and labels. During training and inference, users pass features as iterables
    of non-negative integers and labels as non-negative integers.
    """

    cdef unique_ptr[cc.DenseMultinomialModel] _model

    def __init__(self, size_t nfeats, size_t nlabels, int c = 0):
        self._model = make_unique[cc.DenseMultinomialModel](nfeats, nlabels, c)

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    @classmethod
    def read(cls, filename):
        """
        DenseMultinomialModel.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef DenseMultinomialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            cc.DenseMultinomialModel.Read(tobytes(filename), addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef bool train(self, feats, size_t label) except *:
        """
        train(feats, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature vector and a non-negative integer label for the
            observation.

        Args:
            features: An iterable of non-negative integer features for the
                observation.
            label: The non-negative integer label for the observation.

        Returns:
            Whether the instance was correctly classified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[size_t] fvector = feats
        return self._model.get().Train(fvector, label)

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Average()

    cpdef size_t predict(self, feats):
        """
        predict(feats)

        Predicts the label for a feature vector.

        Args:
            features: An iterable of non-negative integer features for the
                observation.

        Returns:
            The prediction.
        """
        cdef vector[size_t] fvector = feats
        return self._model.get().Predict(fvector)


cdef class SparseDenseMultinomialModel(object):

    """
    SparseDenseMultinomialModel(nfeats, nlabels, c = 0)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).
        c: Margin constant (default: 0).

    Multinomial linear classifier backed by an outer hash table containing
    fixed-size arrays of weights.

    This class provides a more flexible multinomial linear classifier than
    DenseMultinomialModel, but more restrictive (and probably more
    performant) than SparseMultinomialModel. At construction time, clients
    specify a hint for the number of features and a maximum number of labels.
    The former is used to compute the initial size for the outer hash table;
    Like SparseMultinomialModel, this is not a hard constraint. The latter
    is used to compute the max size for the inner array of weights; like
    DenseMultinomialModel but unlike SparseMultinomialModel, this _is_
    a hard constraint. During training and inference, users pass features as
    iterables of strings and labels as non-negative integers.
    """

    cdef unique_ptr[cc.SparseDenseMultinomialModel] _model

    def __init__(self, size_t nfeats, size_t nlabels, int c = 0):
        self._model = make_unique[cc.SparseDenseMultinomialModel](
            nfeats, nlabels, c
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    @classmethod
    def read(cls, filename):
        """
        SparseDenseMultinomialModel.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseDenseMultinomialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(cc.SparseDenseMultinomialModel.Read(
            tobytes(filename),
            addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef bool train(self, feats, size_t label) except *:
        """
        train(feats, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature vector and a string label for the observation.

        Args:
            feats: An iterable of string features for the observation.
            label: The non-negative integer label for the observation.

        Returns:
            Whether the instance was correctly classified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[string] fvector = tobytevector(feats)
        return self._model.get().Train(fvector, label)

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Average()

    cpdef size_t predict(self, feats):
        """
        predict(feats)

        Predicts the label for a feature vector.

        Args:
            features: An iterable of non-negative integer features values for
                the observation.

        Returns:
            The prediction. 
        """
        cdef vector[string] fvector = tobytevector(feats)
        return self._model.get().Predict(fvector)


cdef class SparseDenseMultinomialSequentialModel:

    """
    SparseDenseMultinomialSequentialModel(nfeats, nlabels, order, c = 0)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).
        order: Model order (e.g., 1 implies bigram model).
        c: Margin constant (default: 0).

    Multinomial linear classifier backed by an outer hash table containing
    fixed-size arrays of weights and greedy sequential decoding.
    """

    cdef unique_ptr[cc.SparseDenseMultinomialSequentialModel] _model

    def __init__(self,
                 size_t nfeats,
                 size_t nlabels,
                 size_t order,
                 int c = 0):
        self._model = make_unique[cc.SparseDenseMultinomialSequentialModel](
            nfeats, nlabels, order, c
        )

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseDenseMultinomialSequentialModel.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseDenseMultinomialSequentialModel result \
            = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            cc.SparseDenseMultinomialSequentialModel.Read(tobytes(filename),
                                                          order,
                                                          addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef size_t train(self, efeats, labels) except *:
        """
        train(efeats, labels)

        Trains model using a single labeled sequence.

        This method trains the internal model on a single labeled sequence, in 
        which each observation consists of a feature vector and a integer value.

        Args:
            efeats: An iterable of string emission features for each
                observation.
            labels: An iterable of integral labels for each observation.

        Returns:
            The number of observations in the sequence correctly classsified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        return self._model.get().Train(fvectors, labels)

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self):
        """
        average()
    
        Average the weights in the model.
        """
        self._model.get().Average()

    cpdef vector[size_t] predict(self, efeats):
        """
        predict(efeats) 

        Predicts the labels for a sequence.

        Args:
            efeats: An iterable of string emission features for each
                observation.

        Returns:
            A list of the predicted labels.
        """
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[size_t] yhats
        self._model.get().Predict(fvectors, addr(yhats))
        return yhats


cdef class SparseMultinomialModel(object):

    """
    SparseMultinomialModel(nfeats, nlabels, c = 0)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Hint for the number of unique labels (i.e., classes).
        c: Margin constant (default: 0).

    Multinomial linear classifier backed by a nested hash tables of weights.

    This class provides a more flexible multinomial linear classifier than
    DenseMultinomialModel. At construction time, clients specify an
    hint for the number of features and labels, which is used to compute the
    initial sizes for the nested hash tables; unlike DenseBinomialModel,
    neither are hard constraints. During training and inference, users pass
    features as iterables of strings and labels as strings.
    """

    cdef unique_ptr[cc.SparseMultinomialModel] _model

    def __init__(self, size_t nfeats, size_t nlabels, int c = 0):
        self._model = make_unique[cc.SparseMultinomialModel](nfeats, nlabels, c)

    def __repr__(self):
        return f"<{self.__class__.__name__} at 0x{id(self):x}>"

    @classmethod
    def read(cls, filename):
        """
        SparseMutinomialModel.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseMultinomialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(cc.SparseMultinomialModel.Read(tobytes(filename),
                                                           addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef bool train(self, feats, label) except *:
        """
        train(feats, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature vector and a string label for the observation.

        Args:
            features: An iterable of string features for the observation.
            label: The string label for the observation.

        Returns:
            A boolean indicating whether the instance as already correctly
                labeled; this can be used to compute a epoch's resubstitution
                accuracy.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[string] fvector = tobytevector(feats)
        return self._model.get().Train(fvector, tobytes(label))

    @property
    def averaged(self):
        return self._model.get().Averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Averaged()

    cpdef string predict(self, feats):
        """
        predict(feats)

        Predicts the label for a feature vector.

        Args:
            feats: An iterable of string features for the observation.

        Returns:
             String prediction.
        """
        cdef vector[string] fvector = tobytevector(feats)
        return self._model.get().Predict(fvector)


cdef class SparseMultinomialSequentialModel:

    """
    SparseMultinomialSequentialModel(nfeats, nlabels, order, c = 0)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Hint for the number of unique labels (i.e., classes).
        order: Model order (e.g., 1 implies bigram model).
        c: Margin constant (default: 0).

    Multinomial linear classifier backed by a nested hash tables of weights and
    greedy sequential decoding.
    """

    cdef unique_ptr[cc.SparseMultinomialSequentialModel] _model

    def __init__(self, size_t nfeats, size_t nlabels, size_t order, int c = 0):
        self._model = make_unique[cc.SparseMultinomialSequentialModel](
            nfeats, nlabels, order, c
        )

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseMultinomialSequentialModel.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseMultinomialSequentialModel result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(cc.SparseMultinomialSequentialModel.Read(
            tobytes(filename), order, addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError(f"Read failed: {filename}")
        return (result, metadata)

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Write failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError(f"Write failed: {filename}")

    cpdef size_t train(self, efeats, labels) except *:
        """
        train(efeats, labels)

        Trains model using a single labeled sequence.

        This method trains the internal model on a single labeled sequence, in
        which each observation consists of a feature vector and a string label.

        Args:
            efeats: An iterable of string emission features for each
                observation.
            labels: An iterable of string labels for each observation.

        Returns:
            The number of observations in the sequence correctly classsified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[string] ys = tobytevector(labels)
        return self._model.get().Train(fvectors, ys)

    @property
    def averaged(self):
        return self._model.get().Averaged()
    
    cpdef void average(self):
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOPError: Model already averaged.
        """ 
        if self._model.get().Averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.get().Average()

    cpdef vector[string] predict(self, efeats):
        """
        predict(efeats)
    
        Predicts the labels for a sequence.

        Args:
            efeats: An iterable of string emission features for each
                observation.

        Returns:
            A list of the predicted labels.
        """
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[string] yhats
        self._model.get().Predict(fvectors, addr(yhats))
        return yhats
