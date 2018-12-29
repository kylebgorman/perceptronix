all: unittest python

CXXFLAGS += -std=c++11 -Wall -O2

linear_model.pb.h: linear_model.proto
	protoc --cpp_out=. linear_model.proto

linear_model.pb.cc: linear_model.pb.h

unittest: binomial_perceptron.h binomial_perceptron.cc        \
		  multinomial_perceptron.h multinomial_perceptron.cc  \
          linear_model.pb.h linear_model.pb.cc table.h weight.h  		  \
		  unittest.cc
	g++ $(CXXFLAGS) $(LDFLAGS) binomial_perceptron.cc \
        multinomial_perceptron.cc linear_model.pb.cc unittest.cc \
        -lprotobuf -lpthread -o unittest

test: unittest
	./unittest
	protoc --decode=perceptronix.DenseBinomialPerceptron_pb linear_model.proto < db.pb
	protoc --decode=perceptronix.SparseBinomialPerceptron_pb linear_model.proto < sb.pb
	protoc --decode=perceptronix.DenseMultinomialPerceptron_pb linear_model.proto < dm.pb
	protoc --decode=perceptronix.SparseDenseMultinomialPerceptron_pb linear_model.proto < sdm.pb
	protoc --decode=perceptronix.SparseMultinomialPerceptron_pb linear_model.proto < sm.pb

python: binomial_perceptron.h binomial_perceptron.cc linear_model.pb.h linear_model.pb.cc multinomial_perceptron.h multinomial_perceptron.cc table.h weight.h
	python setup.py build

clean:
	$(RM) -r *.pb *.pb.cc *.pb.h unittest \
	build dist perceptronix.cpp Perceptronix.egg-info

.PHONY: test clean
