from setuptools import Extension
from setuptools import setup

from Cython.Build import cythonize


extension = Extension(
    "perceptronix",
    sources=[
        "src/linear_model.pb.cc",
        "src/binomial_perceptron.cc",
        "src/multinomial_perceptron.cc",
        "src/perceptronix.pyx",
    ],
    libraries=["protobuf", "pthread"],
    language="c++",
    extra_compile_args=["-std=c++11", "-funsigned-char"],
)


__version__ = "0.7"


setup(
    name="perceptronix",
    version=__version__,
    author="Kyle Gorman",
    author_email="kylebgorman@gmail.com",
    description="Perceptron classifiers",
    keywords=[
        "computational linguistics",
        "morphology",
        "natural language processing",
        "phonology",
        "phonetics",
        "speech",
        "language",
    ],
    license="Apache 2.0",
    python_requires=">=3.6",
    zip_safe=False,
    install_requires=["Cython >= 0.29"],
    ext_modules=cythonize([extension]),
)
