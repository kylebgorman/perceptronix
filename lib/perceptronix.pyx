"""Python interface to Perceptronix, providing wrappers for:

* DenseBinomial(Averaging)Perceptron: 
    - DenseBinomialClassifier
* SparseBinomial(Averaging)Perceptron: 
    - SparseBinomialClassifier
    - SparseBinomialSequenceClassifier
* DenseMultinomial(Averaging)Perceptron: 
    - DenseMultinomialClassifier
* SparseDenseMultinomial(Averaging)Perceptron:
    - SparseDenseMultinomialClassifier
    - SparseDenseMultinomialSequenceClassifier
* SparseMultinomial(Averaging)Perceptron: 
    - SparseMultinomialClassifier
    - SparseMultinomialSequenceClassifier

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

    cdef unique_ptr[DenseBinomialAveragingPerceptron] _amodel
    cdef unique_ptr[DenseBinomialPerceptron] _model

    def __init__(self, size_t nfeats):
        self._amodel.reset(new DenseBinomialAveragingPerceptron(nfeats))

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
        self._model.reset(new DenseBinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        result._model.reset(DenseBinomialPerceptron.Read(tobytes(filename),
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
        if not self._averaged():
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
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[size_t] fvector = feats
        return self._amodel.get().Train(fvector, label)

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
        if self._averaged():
            return self._model.get().Predict(fvector)
        else:
            return self._amodel.get().Predict(fvector)


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

    cdef unique_ptr[SparseBinomialAveragingPerceptron] _amodel
    cdef unique_ptr[SparseBinomialPerceptron] _model

    def __init__(self, size_t nfeats):
        self._amodel.reset(new SparseBinomialAveragingPerceptron(nfeats))

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
        self._model.reset(new SparseBinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        result._model.reset(SparseBinomialPerceptron.Read(tobytes(filename),
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
        if not self._averaged():
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
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[string] fvector = tobytevector(feats)
        return self._amodel.get().Train(fvector, label)

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
        if self._averaged():
            return self._model.get().Predict(fvector)
        else:
            return self._amodel.get().Predict(fvector)


cdef class SparseBinomialSequentialClassifier(SparseBinomialClassifier):

    """
    SparseBinomialSequentialClassifier(nfeats, order)

    Args:
        nfeats: Hint for the number of unique features.
        order: Model order (e.g., 1 implies bigram model).
        
    Binomial linear classifier backed by a hash table of weights and greedy
    sequential decoding.
    """

    cdef unique_ptr[SparseTransitionFunctor[bool]] _tf
    cdef unique_ptr[SparseBinomialAveragingDecoder] _adecoder
    cdef unique_ptr[SparseBinomialDecoder] _decoder

    cdef void _init_transition_functor(self, size_t order):
        self._tf.reset(new SparseTransitionFunctor[bool](order))

    cdef void _average_decoder(self):
        self._decoder.reset(new SparseBinomialDecoder(deref(self._model),
                                                      deref(self._tf)))
        self._adecoder.reset()

    def __init__(self, size_t nfeats, size_t order):
        super(SparseBinomialSequentialClassifier, self).__init__(nfeats)
        self._init_transition_functor(order)
        self._adecoder.reset(new SparseBinomialAveragingDecoder(
            self._amodel.get(), deref(self._tf)))

    cpdef void average(self):
        """
        average()
        
        Average the weights in the model.
        """
        super(SparseBinomialSequentialClassifier, self).average()
        self._average_decoder()

    @classmethod
    def read(cls, filename, size_t order):
        """
        SparseBinomialSequentialClassifier.read(filename, order)

        Creates instance by deserialization model on disk.

        Args:
            filename: Path to source file.
            order: Model order.

        Returns:
            A tuple of the deserialized model and the metadata string.

        Raises:
            PerceptronixIOError: Read failed.
        """
        (classifier, metadata) = \
            super(SparseBinomialSequentialClassifier, cls).read(filename)
        classifier._init_transition_functor(order)
        classifier._average_decoder()
        return (classifier, metadata)

    cpdef size_t train_sequence(self, efeats, labels) except *:
        """
        train_sequence(efeats, labels)

        Trains model using a single labeled sequence.

        This method trains the internal model on a single labeled sequence, in
        which each observation consists of a feature vector and a boolean label.

        Args:
            efeats: An iterable of string emission features for each observation.
            labels: An iterable of boolean labels for each observation.

        Returns:
            The number of observations in the sequence correctly classsified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        return self._adecoder.get().Train(fvectors, labels)

    cpdef vector[bool] predict_sequence(self, efeats):
        """
        predict_sequence(efeats)

        Predicts the labels for a sequence.

        Args:
            efeats: An iterable of string emission features for each observation.
        
        Returns:
            A list of the predicted labels.
        """
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[bool] yhats
        if self._averaged():
            self._decoder.get().Predict(fvectors, addr(yhats))
        else:
            self._adecoder.get().Predict(fvectors, addr(yhats))
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

    cdef unique_ptr[DenseMultinomialAveragingPerceptron] _amodel
    cdef unique_ptr[DenseMultinomialPerceptron] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._amodel.reset(new DenseMultinomialAveragingPerceptron(nlabels,
                                                                   nfeats))

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
        self._model.reset(new DenseMultinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
            DenseMultinomialPerceptron.Read(tobytes(filename), addr(metadata))
        )
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
        return self._amodel.get().Train(fvector, label)

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
        if self._averaged():
            return self._model.get().Predict(fvector)
        else:
            return self._amodel.get().Predict(fvector)


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

    cdef unique_ptr[SparseDenseMultinomialAveragingPerceptron] _amodel
    cdef unique_ptr[SparseDenseMultinomialPerceptron] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._amodel.reset(new SparseDenseMultinomialAveragingPerceptron(nlabels,
                                                                         nfeats))

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
        self._model.reset(new SparseDenseMultinomialPerceptron(self._amodel.get()))
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
        result._model.reset(SparseDenseMultinomialPerceptron.Read(
            tobytes(filename), addr(metadata)))
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
        if not self._averaged():
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
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[string] fvector = tobytevector(feats)
        return self._amodel.get().Train(fvector, label)

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
        if self._averaged():
            return self._model.get().Predict(fvector)
        else:
            return self._amodel.get().Predict(fvector)


cdef class SparseDenseMultinomialSequentialClassifier(
    SparseDenseMultinomialClassifier):

    """
    SparseDenseMultinomialSequentialClassifier(nfeats, nlabels, order)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).
        order: Model order (e.g., 1 implies bigram model).

    Multinomial linear classifier backed by an outer hash table containing
    fixed-size arrays of weights and greedy sequential decoding.
    """

    cdef unique_ptr[SparseTransitionFunctor[size_t]] _tf
    cdef unique_ptr[SparseDenseMultinomialAveragingDecoder] _adecoder
    cdef unique_ptr[SparseDenseMultinomialDecoder] _decoder

    cdef void _init_transition_functor(self, size_t order):
        self._tf.reset(new SparseTransitionFunctor[size_t](order))

    cdef void _average_decoder(self):
        self._decoder.reset(new SparseDenseMultinomialDecoder(
            deref(self._model), deref(self._tf)))
        self._adecoder.reset()

    def __init__(self, size_t nfeats, size_t nlabels, size_t order):
        super(SparseDenseMultinomialSequentialClassifier, self).__init__(
            nfeats, nlabels)
        self._init_transition_functor(order)
        self._adecoder.reset(new SparseDenseMultinomialAveragingDecoder(
            self._amodel.get(), deref(self._tf)))

    cpdef void average(self):
        """
        average()
    
        Average the weights in the model.
        """
        super(SparseDenseMultinomialSequentialClassifier, self).average()
        self._average_decoder()

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
        cdef SparseDenseMultinomialSequentialClassifier classifier
        (classifier, metadata) = \
            super(SparseDenseMultinomialSequentialClassifier, cls).read(
                filename)
        classifier._init_transition_functor(order)
        classifier._average_decoder()
        return (classifier, metadata)

    cpdef size_t train_sequence(self, efeats, labels) except *:
        """
        train_sequence(efeats, labels)

        Trains model using a single labeled sequence.

        This method trains the internal model on a single labeled sequence, in 
        which each observation consists of a feature vector and a integer value.

        Args:
            efeats: An iterable of string emission features for each observation.
            labels: An iterable of integral labels for each observation.

        Returns:
            The number of observations in the sequence correctly classsified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        return self._adecoder.get().Train(fvectors, labels)

    cpdef vector[size_t] predict_sequence(self, efeats):
        """
        predict_sequence(efeats) 

        Predicts the labels for a sequence.

        Args:
            efeats: An iterable of string emission features for each observation.

        Returns:
            A list of the predicted labels.
        """
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[size_t] yhats
        if self._averaged():
            self._decoder.get().Predict(fvectors, addr(yhats))
        else:
            self._adecoder.get().Predict(fvectors, addr(yhats))
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

    cdef unique_ptr[SparseMultinomialAveragingPerceptron] _amodel
    cdef unique_ptr[SparseMultinomialPerceptron] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._amodel.reset(new SparseMultinomialAveragingPerceptron(nlabels,
                                                                    nfeats))

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
        self._model.reset(new SparseMultinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        result._model.reset(SparseMultinomialPerceptron.Read(tobytes(filename),
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
        if not self._averaged():
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
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[string] fvector = tobytevector(feats)
        return self._amodel.get().Train(fvector, tobytes(label))

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
        if self._averaged():
            return self._model.get().Predict(fvector)
        else:
            return self._amodel.get().Predict(fvector)


cdef class SparseMultinomialSequentialClassifier(
    SparseMultinomialClassifier):

    """
    SparseMultinomialSequentialClassifier(nfeats, nlabels, order)

    Args:
        nfeats: Hint for the number of unique features.
        nlabels: Hint for the number of unique labels (i.e., classes).
        order: Model order (e.g., 1 implies bigram model).

    Multinomial linear classifier backed by a nested hash tables of weights and
    greedy sequential decoding.
    """

    cdef unique_ptr[SparseTransitionFunctor[string]] _tf
    cdef unique_ptr[SparseMultinomialAveragingDecoder] _adecoder
    cdef unique_ptr[SparseMultinomialDecoder] _decoder

    cdef void _init_transition_functor(self, size_t order):
        self._tf.reset(new SparseTransitionFunctor[string](order))

    cdef void _average_decoder(self):
        self._decoder.reset(new SparseMultinomialDecoder(deref(self._model),
                                                         deref(self._tf)))
        self._adecoder.reset()

    def __init__(self, size_t nfeats, size_t nlabels, size_t order):
        super(SparseMultinomialSequentialClassifier, self).__init__(nfeats,
                                                                    nlabels)
        self._init_transition_functor(order)
        self._adecoder.reset(new SparseMultinomialAveragingDecoder(
            self._amodel.get(), deref(self._tf)))

    cpdef void average(self):
        """
        average()

        Average the weights in the model.
        """ 
        super(SparseMultinomialSequentialClassifier, self).average()
        self._average_decoder()

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
        cdef SparseMultinomialSequentialClassifier classifier
        (classifier, metadata) = \
            super(SparseMultinomialSequentialClassifier, cls).read(filename)
        classifier._init_transition_functor(order)
        classifier._average_decoder()
        return (classifier, metadata)

    cpdef size_t train_sequence(self, efeats, labels) except *:
        """
        train_sequence(efeats, labels)

        Trains model using a single labeled sequence.

        This method trains the internal model on a single labeled sequence, in
        which each observation consists of a feature vector and a string label.

        Args:
            efeats: An iterable of string emission features for each observation.
            labels: An iterable of string labels for each observation.

        Returns:
            The number of observations in the sequence correctly classsified.

        Raises:
            PerceptronixOpError: Model already averaged.
        """
        if self._averaged():
            raise PerceptronixOpError("Model already averaged")
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[string] ys = tobytevector(labels)
        return self._adecoder.get().Train(fvectors, ys)

    cpdef vector[string] predict_sequence(self, efeats):
        """
        predict_sequence(efeats)
    
        Predicts the labels for a sequence.

        Args:
            efeats: An iterable of string emission features for each observation.

        Returns:
            A list of the predicted labels.
        """
        cdef vector[vector[string]] fvectors = tobytevectors(efeats)
        cdef vector[string] yhats
        if self._averaged():
            self._decoder.get().Predict(fvectors, addr(yhats))
        else:
            self._adecoder.get().Predict(fvectors, addr(yhats))
        return yhats
