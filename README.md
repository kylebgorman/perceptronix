Perceptronix: Linear model template library for C++11

This library provides C++ classes for linear classifiers for
binary-valued (i.e., nominal, "one-hot") features, such as encountered
in natural language processing. Models are trained from labeled examples
using the [perceptron learning
algorithm](https://en.wikipedia.org/wiki/Perceptron#Learning_algorithm)
with or without [weight
averaging](http://cseweb.ucsd.edu/~yfreund/papers/LargeMarginsUsingPerceptron.pdf).
The former produces an informal form of L1 normalization (with all else
held equal, it favors zero weights over non-zero weights, as weights are
initialized to zero and a large margin is not part of the objective)
whereas the latter produces an informal form of L2 normalization (with
all else held equal, it prefers weights with smaller magnitudes, as
weights are initialized to zero). Both binomial and multinomial
classifiers are supported. Weights may be stored either in sparse
(string-keyed hash table) or dense (integer-keyed array) containers.
Models may be serialized and deserialized using [protocol
buffers](https://github.com/google/protobuf).

API
===

C++
---

Classes are available in both averaged and unaveraged forms. The
suggested workflow is to train an averaged model, and then use the
averaged weights to initialize an immutable unaveraged model. E.g.:

      perceptronix::SparseDenseMultinomialModel model(nfeats, nlabels);
      // ... training ...
      model.Average();
      // ... inference ...

Note that in the example above, `avgmodel` requires approximately three
times as much memory as `model`. Furthermore, inference is significantly
faster with an constant `model`, and this conversion is required for
model serialization.

The major classes are:

-   `DenseBinomialModel`: A binomial classifier using a dense weight
    table; users must set the maximum number of unique features at
    construction time.
-   `SparseBinomialModel`: A binomial classifier using a
    dynamically-sized sparse weight table.
-   `DenseMultinomialModel`: A multinomial classifier using dense weight
    tables; users must set the maximum number of unique features and
    labels at construction time.
-   `SparseDenseMultinomialModel`: A multinomial classifier using a
    dynamically-sized hash table for the outer table and dense arrays
    for the inner table.
-   `SparseMultinomialModel`: A multinomial classifier using
    dynamically-sized hash tables for both outer and inner tables.

There are also HMM-like classes for greedy sequential decoding:

-   `SparseBinomialSequentialModel`
-   `SparseDenseMultinomialSequentialModel`
-   `SparseMultinomialSequentialModel`

Python
------

Each of the above classifiers has a correspondent type in the Python
API. However, these automate the averaging process. Take a look at
`applications/sentence_tokenizer/sentence_tokenizer.py` for a worked
example.

Design
======

Weights
-------

Weight averaging is done using a [lazy
strategy](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python#averaging-the-weights).
An averaged weight consists of:

-   The true weight (`weight_`), used for inference *during training*,
    and updated using the perceptron learning rule
-   The summed weight (`summed_weight_`), which may or may not be
    "fresh"
-   A timestamp (`time_`) indicating when the averaged weight was last
    updated

At `time`, the time elapsed since the summed weight was last "freshened"
is given by `time - time_`.

The summed weight is "freshened" by adding to it the true weight
multiplied by the time elapsed since timestamp.

The weight is averaged by freshening the summed weight then dividing it
by the current time.

Tables
------

The header `table.h` wraps the two types of weight containers so that
they provide a common interface for access, mutation, and iteration.

The resulting classes are then used as template template parameters (no,
that's not a typo) to the classifiers themselves.

Virtual dispatch is not used anywhere, so there is no performance
penalty for this polymorphism.

Labels and features
-------------------

Labels and features are to either either unsigned integers (for dense
tables) or strings (for sparse tables), and therefore are passed by
value in both cases. In the latter case, it is assumed that the strings
are short so that we can take advantage of "short string" optimizations.

There is a built-in bias term, so users do not need to provide one.

For dense tables, passing a feature integer greater than or equal to
`nfeats` is Undefined Behavior.

For sparse tables, the empty string is reserved for internal use, and
attempting to use it as either a label or feature is Undefined Behavior.

Performance
-----------

Instead of the naïve strategy of treating labels as the outer indices
and features as the inner indices, this library uses outer feature
indices and inner label indices in the multinomial case. This allow
sparse multinomial models to perform particularly well in the very
common case that most features have zero weights for all labels; the
cost of inference for an "irrelevant" feature---one which has zero
weights for all labels, or one which has a non-zero weight for some
labels but not the one under consideration---is merely that of a hash
table miss.

TODO: Kyle will add an actual big-O analysis here.

When you have sparse features, but you know the maximum number of labels
at construction time (and this number is small, say, less than 64),
consider using `SparseDenseMultinomial(Averaging)Perceptron` rather than
`SparseMultinomialPerceptron`. This model uses `std::valarray` for inner
weight tables, which should allow the "dot product" during scoring to
use fast vectorized arithmetic operations.

If you do use `SparseMultinomialPerceptron`, consider tuning the
`nlabels` constructor argument, which is used as the default size for
inner tables.

Dependencies
============

The library depends on C++11 and protobuf 3.0 or greater. It should
compile with `g++` or `clang++`. Users will also need the protobuf
compilation tool `protoc`, which is sometimes distributed separately
from protobuf itself.

The Python wrapper and applications are written for Python 3.7.
Compiling the wrapper requires Cython, and the applications require the
`nlup` and `regex` Python libraries, both of which are available via
pip.

Author
======

[Kyle Gorman](kylebgorman@gmail.com)

License
=======

See `LICENSE`.
