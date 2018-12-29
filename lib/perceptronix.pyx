"""Python interface to Perceptronix, providing wrappers for:

* DenseBinomial(Averaged)Perceptron: `DenseBinomialClassifier`
* SparseBinomial(Averaged)Perceptron: `SparseBinomialClassifier`
* DenseMultinomial(Averaged)Perceptron: `DenseMultinomialClassifier`
* SparseDenseMultinomial(Averaged)Perceptron:
            `SparseDenseMultinomialClassifier`
* SparseMultinomial(Averaged)Perceptron: `SparseMultinomialClassifier`

Each instance is initially an averaging model; calling the instance method
`average` "finalizes" the weights (by calling the averaging constructor
of the unaveraged class), disabling further training and freeing the
averaged model's memory.
"""

from cython.operator cimport address as addr

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


# Custom exception classes.



class PerceptronixIOError(IOError):

    pass


class PerceptronixOperationError(RuntimeError):

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

    cdef unique_ptr[DenseBinomialAveragedPerceptron] _amodel
    cdef unique_ptr[DenseBinomialPerceptron] _model

    def __init__(self, size_t nfeats):
        self._amodel.reset(new DenseBinomialAveragedPerceptron(nfeats))

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

        This method is applied to a model constructed using the __init__
        constructor to average all weights in the model. As a result:

        * Further training with `train` is disallowed.
        * Prediction with `predict` becomes faster and should generalizes
          better.
        * Serialization with (`write`) becomes possible.
        * The memory footprint of the model shrinks to approximately one third 
          of the pre-averaged model.

        Therefore, this method should be called when switching from training to
        inference or serialization.

        Raises:
            PerceptronixOperationError: Model already averaged.

        This method cannot be invoked on an instance created by deserialization
        with `read`, as such models are already averaged.
        """
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        self._model.reset(new DenseBinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        cdef DenseBinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            DenseBinomialPerceptron.Read(tobytes(filename), addr(metadata))
        )
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
            PerceptronixOperationError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._averaged():
            raise PerceptronixOperationError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

    cpdef bool train(self, features, bool label) except *:
        """
        train(features, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature bundle and a boolean label for the observation.

        Args:
            features: An iterable of non-negative integer feature values for the
                observation.
            label: The boolean label for the observation.

        Returns:
            A boolean indicating whether the instance as already correctly
                labeled; this can be used to compute a epoch's resubstitution
                accuracy.

        Raises:
            PerceptronixOperationError: Model already averaged.
        """
        cdef vector[size_t] fb = features
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        return self._amodel.get().Train(fb, label)

    cpdef bool predict(self, features):
        """
        predict(features)

        Predicts the label for a feature bundle.

        Args:
            features: An iterable of non-negative integer feature values for the
                observation.

        Returns:
             Boolean prediction.
        """
        cdef vector[size_t] fb = features
        if self._averaged():
            return self._model.get().Predict(fb)
        else:
            return self._amodel.get().Predict(fb)


cdef class SparseBinomialClassifier(object):

    """
    SparseBinomialClassifier(nfeats)

    Args:
        nfeats: Estimated number of unique features.

    Binomial linear classifier backed by a hash table of weights.

    This class provides a more flexible binomial linear classifier than
    DenseBinomialClassifier. At construction time, clients specify an estimated
    number of features, which is used to compute the initial size for the hash
    table; unlike DenseBinomialClassifier, though, this is not a hard
    constraint. During training and inference, users pass features as iterables
    of strings.
    """

    cdef unique_ptr[SparseBinomialAveragedPerceptron] _amodel
    cdef unique_ptr[SparseBinomialPerceptron] _model

    def __init__(self, size_t nlabels):
        self._amodel.reset(new SparseBinomialAveragedPerceptron(nlabels))

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

        This method is applied to a model constructed using the __init__
        constructor to average all weights in the model. As a result:

        * Further training with `train` is disallowed.
        * Prediction with `predict` becomes faster and should generalizes
          better.
        * Serialization with (`write`) becomes possible.

        Raises:
            PerceptronixOperationError: Model already averaged.

        This method cannot be invoked on an instance created by deserialization
        with `read`, as such models are already averaged.
        """
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        self._model.reset(new SparseBinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        cdef SparseBinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            SparseBinomialPerceptron.Read(tobytes(filename), addr(metadata))
        )
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
            PerceptronixOperationError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._averaged():
            raise PerceptronixOperationError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

    cpdef bool train(self, features, bool label) except *:
        """
        train(features, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature bundle and a boolean label for the observation.

        Args:
            features: An iterable of string feature values for the observation.
            label: The boolean label for the observation.

        Returns:
            A boolean indicating whether the instance as already correctly
                labeled; this can be used to compute a epoch's resubstitution
                accuracy.

        Raises:
            PerceptronixOperationError: Model already averaged.
        """
        cdef vector[string] fb = [tobytes(feat) for feat in features]
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        return self._amodel.get().Train(fb, label)

    cpdef bool predict(self, features):
        """
        predict(features)

        Predicts the label for a feature bundle.

        Args:
            features: An iterable of string feature values for the observation.

        Returns:
             Boolean prediction.
        """
        cdef vector[string] fb = [tobytes(feat) for feat in features]
        if self._averaged():
            return self._model.get().Predict(fb)
        else:
            return self._amodel.get().Predict(fb)


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

    cdef unique_ptr[DenseMultinomialAveragedPerceptron] _amodel
    cdef unique_ptr[DenseMultinomialPerceptron] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._amodel.reset(new DenseMultinomialAveragedPerceptron(nlabels,
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

        This method is applied to a model constructed using the __init__
        constructor to average all weights in the model. As a result:

        * Further training with `train` is disallowed.
        * Prediction with `predict` becomes faster and should generalizes
          better.
        * Serialization with (`write`) becomes possible.
        * The memory footprint of the model shrinks to approximately one third
          of the pre-averaged model.

        Therefore, this method should be called when switching from training to
        inference or serialization.

        Raises:
            PerceptronixOperationError: Model already averaged.

        This method cannot be invoked on an instance created by deserialization
        with `read`, as such models are already averaged.
        """
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        self._model.reset(new DenseMultinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
            PerceptronixOperationError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._averaged():
            raise PerceptronixOperationError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

    cpdef bool train(self, features, size_t label) except *:
        """
        train(features, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature bundle and a non-negative integer label for the
            observation.

        Args:
            features: An iterable of non-negative integer feature values for the
                observation.
            label: The non-negative integer label for the observation.

        Returns:
            A boolean indicating whether the instance as already correctly
                labeled; this can be used to compute a epoch's resubstitution
                accuracy.

        Raises:
            PerceptronixOperationError: Model already averaged.
        """
        cdef vector[size_t] fb = features
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        return self._amodel.get().Train(fb, label)

    cpdef size_t predict(self, features):
        """
        predict(features)

        Predicts the label for a feature bundle.

        Args:
            features: An iterable of non-negative integer feature values for the
                observation.

        Returns:
            Non-negative integer prediction.
        """
        cdef vector[size_t] fb = features
        if self._averaged():
            return self._model.get().Predict(fb)
        else:
            return self._amodel.get().Predict(fb)


cdef class SparseDenseMultinomialClassifier(object):

    """
    SparseDenseMultinomialClassifier(nfeats, nlabels)

    Args:
        nfeats: Estimated number of unique features.
        nlabels: Maximum number of unique labels (i.e., classes).

    Multinomial linear classifier backed by an outer hash table containing
    fixed-size arrays of weights.

    This class provides a more flexible multinomial linear classifier than
    DenseMultinomialClassifier, but more restrictive (and probably more
    performant) than SparseMultinomialClassifier. At construction time, clients
    specify an estimated number of features and a maximum number of labels.
    The former is used to compute the initial size for the outer hash table;
    Like SparseMultinomialClassifier, this is not a hard constraint. The latter
    is used to compute the max size for the inner array of weights; like
    DenseMultinomialClassifier but unlike SparseMultinomialClassifier, this _is_
    a hard constraint. During training and inference, users pass features as
    iterables of strings and labels as non-negative integers.
    """

    cdef unique_ptr[SparseDenseMultinomialAveragedPerceptron] _amodel
    cdef unique_ptr[SparseDenseMultinomialPerceptron] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._amodel.reset(new SparseDenseMultinomialAveragedPerceptron(nlabels,
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

        This method is applied to a model constructed using the __init__
        constructor to average all weights in the model. As a result:

        * Further training with `train` is disallowed.
        * Prediction with `predict` becomes faster and should generalizes
          better.
        * Serialization with (`write`) becomes possible.
        * The memory footprint of the model shrinks to approximately one third
          of the pre-averaged model.

        Therefore, this method should be called when switching from training to
        inference or serialization.

        Raises:
            PerceptronixOperationError: Model already averaged.

        This method cannot be invoked on an instance created by deserialization
        with `read`, as such models are already averaged.
        """
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        self._model.reset(new SparseDenseMultinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        cdef SparseDenseMultinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            SparseDenseMultinomialPerceptron.Read(
                tobytes(filename),
                addr(metadata),
            )
        )
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
            PerceptronixOperationError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._averaged():
            raise PerceptronixOperationError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

    cpdef bool train(self, features, size_t label) except *:
        """
        train(features, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature bundle and a string label for the observation.

        Args:
            features: An iterable of string feature values for the observation.
            label: The non-negative integer label for the observation.

        Returns:
            A boolean indicating whether the instance as already correctly
                labeled; this can be used to compute a epoch's resubstitution
                accuracy.

        Raises:
            PerceptronixOperationError: Model already averaged.
        """
        cdef vector[string] fb = [tobytes(feat) for feat in features]
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        return self._amodel.get().Train(fb, label)

    cpdef size_t predict(self, features):
        """
        predict(features)

        Predicts the label for a feature bundle.

        Args:
            features: An iterable of non-negative integer features values for
                the observation.

        Returns:
             Non-negative integer prediction.
        """
        cdef vector[string] fb = [tobytes(feat) for feat in features]
        if self._averaged():
            return self._model.get().Predict(fb)
        else:
            return self._amodel.get().Predict(fb)


cdef class SparseMultinomialClassifier(object):

    """
    SparseMultinomialClassifier(nfeats, nlabels)

    Args:
        nfeats: Estimated number of unique features.
        nlabels: Estimated number of unique labels (i.e., classes).

    Multinomial linear classifier backed by a nested hash tables of weights.

    This class provides a more flexible multinomial linear classifier than
    DenseMultinomialClassifier. At construction time, clients specify an
    estimated number of features and labels, which is used to compute the
    initial sizes for the nested hash tables; unlike DenseBinomialClassifier,
    neither are hard constraints. During training and inference, users pass
    features as iterables of strings and labels as strings.
    """

    cdef unique_ptr[SparseMultinomialAveragedPerceptron] _amodel
    cdef unique_ptr[SparseMultinomialPerceptron] _model

    def __init__(self, size_t nlabels, size_t nfeats):
        self._amodel.reset(new SparseMultinomialAveragedPerceptron(nlabels,
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

        This method is applied to a model constructed using the __init__
        constructor to average all weights in the model. As a result:

        * Further training with `train` is disallowed.
        * Prediction with `predict` becomes faster and should generalizes
          better.
        * Serialization with (`write`) becomes possible.
        * The memory footprint of the model shrinks to approximately one third 
          of the pre-averaged model.

        Therefore, this method should be called when switching from training to
        inference or serialization.

        Raises:
            PerceptronixOperationError: Model already averaged.

        This method cannot be invoked on an instance created by deserialization
        with `read`, as such models are already averaged.
        """
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        self._model.reset(new SparseMultinomialPerceptron(self._amodel.get()))
        self._amodel.reset()

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
        cdef SparseMultinomialClassifier result = cls.__new__(cls)
        cdef string metadata
        result._model.reset(
            SparseMultinomialPerceptron.Read(
                tobytes(filename),
                addr(metadata),
            )
        )
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
            PerceptronixOperationError: Must average model first.
            PerceptronixIOError: Read failed.
        """
        if not self._averaged():
            raise PerceptronixOperationError("Must average model first")
        if not self._model.get().Write(tobytes(filename), tobytes(metadata)):
            raise PerceptronixIOError("Write failed: {}".format(filename))

    cpdef bool train(self, features, label) except *:
        """
        train(features, label)

        Trains model using a single labeled observation.

        This method trains the internal model on a single labeled observation,
        consisting of a feature bundle and a string label for the observation.

        Args:
            features: An iterable of string feature values for the observation.
            label: The string label for the observation.

        Returns:
            A boolean indicating whether the instance as already correctly
                labeled; this can be used to compute a epoch's resubstitution
                accuracy.

        Raises:
            PerceptronixOperationError: Model already averaged.
        """
        cdef vector[string] fb = [tobytes(feat) for feat in features]
        if self._averaged():
            raise PerceptronixOperationError("Model already averaged")
        return self._amodel.get().Train(fb, tobytes(label))

    cpdef string predict(self, features):
        """
        predict(features)

        Predicts the label for a feature bundle.

        Args:
            features: An iterable of string feature values for the observation.

        Returns:
             String prediction.
        """
        cdef vector[string] fb = [tobytes(feat) for feat in features]
        if self._averaged():
            return self._model.get().Predict(fb)
        else:
            return self._amodel.get().Predict(fb)
