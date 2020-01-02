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

setup(
    name="Perceptronix",
    name="perceptronix",
    version="0.7",
    author="Kyle Gorman",
    author_email="kylebgorman@gmail.com",
    install_requires=["Cython >= 0.29"],
    ext_modules=cythonize([extension]),
)
