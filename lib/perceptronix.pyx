#cython: language_level=3
"""Python interface to Perceptronix.

Each instance is initially an averaging model; calling the instance method
`average` "finalizes" the weights (by calling the averaging constructor
of the unaveraged class), disabling further training and freeing the
averaging model's memory.
"""

from cython.operator cimport address as addr
from cython.operator cimport dereference as deref

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string


# Helper for converting strings.


cdef string tobytes(data) except *:
    if isinstance(data, bytes):
        return data
    try:
        return data.encode("utf8")
    except Exception:
        raise ValueError("Cannot encode {!r} as a bytestring".format(data))


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


cdef class DenseBinomialClassifier(object):

    """
    DenseBinomialClassifier(nfeats)

    Args:
        nfeats: Maximum number of unique features.

    Binomial linear classifier backed by a fixed-size contiguous array of
    weights.

    This class provides the simplest form of a binomial linear classifier. At
    construction time, clients specify the maximum number of unique features.
    During training and inference, users pass features as iterables of
    non-negative integers.
    """

    cdef unique_ptr[DenseBinomialModel] _model

    def __init__(self, size_t nfeats):
        self._model.reset(new DenseBinomialModel(nfeats))

    def __repr__(self):
        return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

    @classmethod
    def read(cls, filename):
        """
        DenseBinomialClassifier.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef DenseBinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(DenseBinomialModel.Read(tobytes(filename),
                                                    addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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


cdef class SparseBinomialClassifier(object):

    """
    SparseBinomialClassifier(nfeats)

    Args:
        nfeats: Hint for the number of unique features.

    Binomial linear classifier backed by a hash table of weights.

    This class provides a more flexible binomial linear classifier than
    DenseBinomialClassifier. At construction time, clients specify a hint for
    the number of features, which is used to compute the initial size for the
    hash table; unlike DenseBinomialClassifier, however, this is not a hard
    constraint. During training and inference, users pass features as iterables
    of strings.
    """

    cdef unique_ptr[SparseBinomialModel] _model

    def __init__(self, size_t nfeats):
        self._model.reset(new SparseBinomialModel(nfeats))

    def __repr__(self):
        return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

    @classmethod
    def read(cls, filename):
        """
        SparseBinomialClassifier.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseBinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(SparseBinomialModel.Read(tobytes(filename),
                                                     addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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


cdef class SparseBinomialSequentialClassifier:

    """
    SparseBinomialSequentialClassifier(nfeats, order)

    Args:
        nfeats: Hint for the number of unique features.
        order: Model order (e.g., 1 implies bigram model).
        
    Binomial linear classifier backed by a hash table of weights and greedy
    sequential decoding.
    """

    cdef unique_ptr[SparseBinomialSequentialModel] _model

    def __init__(self, size_t nfeats, size_t order):
        self._model.reset(new SparseBinomialSequentialModel(nfeats, order))

    def __repr__(self):
        return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseBinomialSequentialClassifier.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order (e.g., 1 implies bigram model).

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseBinomialSequentialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            SparseBinomialSequentialModel.Read(tobytes(filename),
                                               order,
                                               addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata) except *:
        """
                write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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


cdef class DenseMultinomialClassifier(object):

    """
    DenseMultiomialClassifier(nfeats, nlabels)

    Args:
        nfeats: Maximum number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).

    Multinomial linear classifier backed by fixed-size contiguous arrays of
    weights.

    This class provides the simplest form of a multinomial linear classifier. At
    construction time, clients specify the maximum number of unique features
    and labels. During training and inference, users pass features as iterables
    of non-negative integers and labels as non-negative integers.
    """

    cdef unique_ptr[DenseMultinomialModel] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._model.reset(new DenseMultinomialModel(nlabels, nfeats))

    def __repr__(self):
        return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

    @classmethod
    def read(cls, filename):
        """
        DenseMultinomialClassifier.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef DenseMultinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            DenseMultinomialModel.Read(tobytes(filename), addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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


cdef class SparseDenseMultinomialClassifier(object):

    """
    SparseDenseMultinomialClassifier(nfeats, nlabels)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).

    Multinomial linear classifier backed by an outer hash table containing
    fixed-size arrays of weights.

    This class provides a more flexible multinomial linear classifier than
    DenseMultinomialClassifier, but more restrictive (and probably more
    performant) than SparseMultinomialClassifier. At construction time, clients
    specify a hint for the number of features and a maximum number of labels.
    The former is used to compute the initial size for the outer hash table;
    Like SparseMultinomialClassifier, this is not a hard constraint. The latter
    is used to compute the max size for the inner array of weights; like
    DenseMultinomialClassifier but unlike SparseMultinomialClassifier, this _is_
    a hard constraint. During training and inference, users pass features as
    iterables of strings and labels as non-negative integers.
    """

    cdef unique_ptr[SparseDenseMultinomialModel] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._model.reset(new SparseDenseMultinomialModel(nlabels, nfeats)) 

    def __repr__(self):
        return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

    cdef bool _averaged(self):
        return self._amodel.get() == NULL

    @property
    def averaged(self):
        return self._averaged()

    cpdef void average(self) except *:
        """
        average()

        Average the weights in the model.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        self._model.reset(new SparseDenseMultinomialPerceptron(
            self._amodel.get()))
        self._amodel.reset()

    @classmethod
    def read(cls, filename):
        """
        SparseDenseMultinomialClassifier.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseDenseMultinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(SparseDenseMultinomialModel.Read(tobytes(filename),
                                                             addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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
            Whetehr the instance was correctly classified.

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

cdef class SparseDenseMultinomialSequentialClassifier:

    """
    SparseDenseMultinomialSequentialClassifier(nfeats, nlabels, order)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).
        order: Model order (e.g., 1 implies bigram model).

    Multinomial linear classifier backed by an outer hash table containing
    fixed-size arrays of weights and greedy sequential decoding.
    """

    cdef unique_ptr[SparseDenseMultinomialSequentialModel] _model

    def __init__(self, size_t nfeats, size_t nlabels, size_t order):
        self._model.reset(new SparseDenseMultinomialSequentialModel(
            nfeats, nlabels, order))

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseDenseMultinomialSequentialClassifier.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseDenseMultinomialSequentialClassifier result \
            = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            SparseDenseMultinomialSequentialModel.Read(tobytes(filename),
                                                       order,
                                                       addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata) except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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


cdef class SparseMultinomialClassifier(object):

    """
    SparseMultinomialClassifier(nfeats, nlabels)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Hint for the number of unique labels (i.e., classes).

    Multinomial linear classifier backed by a nested hash tables of weights.

    This class provides a more flexible multinomial linear classifier than
    DenseMultinomialClassifier. At construction time, clients specify an
    hint for the number of features and labels, which is used to compute the
    initial sizes for the nested hash tables; unlike DenseBinomialClassifier,
    neither are hard constraints. During training and inference, users pass
    features as iterables of strings and labels as strings.
    """

    cdef unique_ptr[SparseMultinomialModel] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._model.reset(new SparseMultinomialModel(nlabels, nfeats))

    def __repr__(self):
        return "<{} at 0x{:x}>".format(self.__class__.__name__, id(self))

    @classmethod
    def read(cls, filename):
        """
        SparseMutinomialClassifier.read(filename)

        Creates instance by deserializing model on disk.

        Args:
            filename: Path to source file.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseMultinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(SparseMultinomialModel.Read(tobytes(filename),
                                                        addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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


cdef class SparseMultinomialSequentialClassifier:

    """
    SparseMultinomialSequentialClassifier(nfeats, nlabels, order)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Hint for the number of unique labels (i.e., classes).
        order: Model order (e.g., 1 implies bigram model).

    Multinomial linear classifier backed by a nested hash tables of weights and
    greedy sequential decoding.
    """

    cdef unique_ptr[SparseMultinomialSequentialModel] _model

    def __init__(self, size_t nfeats, size_t nlabels, size_t order):
        self._model.reset(new SparseMultinomialSequentialModel(nfeats,
                                                               nlabels,
                                                               order))

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseMultinomialSequentialClassifier.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        cdef SparseMultinomialSequentialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(SparseMultinomialSequentialModel.Read(
            tobytes(filename), order, addr(metadata)))
        if result._model.get() == NULL:
            raise PerceptronixIOError("Read failed: {}".format(filename))
        return (result, metadata.decode("utf8"))

    cpdef void write(self, filename, metadata=b"") except *:
        """
        write(filename, metadata)

        Serializes model to disk.

        Args:
            filename: Path to sink file.
            metadata: Optional metadata string.

        Raises:
            PerceptronixOpError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._model.get().Averaged():
            raise PerceptronixOpError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

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
