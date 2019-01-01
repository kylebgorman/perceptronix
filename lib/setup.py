from setuptools import Extension
from setuptools import setup

from Cython.Build import cythonize


extension = Extension(
    "perceptronix",
    sources=[
        "linear_model.pb.cc",
        "binomial_perceptron.cc",
        "multinomial_perceptron.cc",
        "perceptronix.pyx",
    ],
    libraries=["protobuf"],
    language="c++",
    extra_compile_args=["-std=c++11", "-funsigned-char"],
)

setup(
    name="Perceptronix",
    version="0.3",
    author="Kyle Gorman",
    author_email="kylebgorman@gmail.com",
    install_requires=["Cython >= 0.29"],
    ext_modules=cythonize(
        [extension], compiler_directives={"language_level": 3}
    ),
)
