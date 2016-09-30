from setuptools import Extension
from setuptools import setup

from Cython.Build import cythonize


extension = Extension("perceptronix",
                      sources=["perceptronix.pyx",
                               "binomial_perceptron.cc",
                               "linmod.pb.cc",
                               "multinomial_perceptron.cc"],
                      libraries=["protobuf"],
                      language="c++",
                      extra_compile_args=["-std=c++11"])

setup(name="Perceptronix",
      version="0.1",
      author="Kyle Gorman",
      author_email="kylebgorman@gmail.com",
      install_requires=["Cython >= 0.23"],
      ext_modules=cythonize([extension]))
