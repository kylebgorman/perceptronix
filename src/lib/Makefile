all: unittest python

CXXFLAGS += -std=c++11 -Wall -O2

linmod.pb.h: linmod.proto
	protoc --cpp_out=. linmod.proto

linmod.pb.cc: linmod.pb.h

unittest: binomial_perceptron.h binomial_perceptron.cc        \
		  multinomial_perceptron.h multinomial_perceptron.cc  \
          linmod.pb.h linmod.pb.cc table.h weight.h  		  \
		  unittest.cc
	g++ $(CXXFLAGS) $(LDFLAGS) binomial_perceptron.cc \
        multinomial_perceptron.cc linmod.pb.cc unittest.cc \
        -lprotobuf -lpthread -o unittest

test: unittest
	./unittest
	protoc --decode=perceptronix.DenseBinomialPerceptron_pb linmod.proto < db.pb
	protoc --decode=perceptronix.SparseBinomialPerceptron_pb linmod.proto < sb.pb
	protoc --decode=perceptronix.DenseMultinomialPerceptron_pb linmod.proto < dm.pb
	protoc --decode=perceptronix.SparseDenseMultinomialPerceptron_pb linmod.proto < sdm.pb
	protoc --decode=perceptronix.SparseMultinomialPerceptron_pb linmod.proto < sm.pb

python: binomial_perceptron.h binomial_perceptron.cc linmod.pb.h linmod.pb.cc multinomial_perceptron.h multinomial_perceptron.cc table.h weight.h
	python setup.py build

clean:
	$(RM) -r *.pb *.pb.cc *.pb.h unittest

clean_python:
	$(RM) -r build dist perceptronix.cpp Perceptronix.egg-info

.PHONY: test clean
