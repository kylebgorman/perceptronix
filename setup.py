from setuptools import Extension
from setuptools import setup

from Cython.Build import cythonize


extension = Extension(
    "_perceptronix",
    sources=[
        "extensions/_perceptronix.pyx",
        "extensions/binomial_perceptron.cc",
        "extensions/linear_model.pb.cc",
        "extensions/multinomial_perceptron.cc",
    ],
    libraries=["protobuf", "pthread"],
    language="c++",
    extra_compile_args=["-std=c++17", "-funsigned-char"],
)


__version__ = "0.7.5"


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
    install_requires=["Cython >= 0.29"],
    ext_modules=cythonize([extension]),
    packages=["perceptronix"],
    package_data={"perceptronix": ["__init__.pyi", "py.typed"]},
    zip_safe=False,
)
