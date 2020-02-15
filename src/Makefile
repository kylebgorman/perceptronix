all: unittest

CXXFLAGS += -std=c++14 -Wall -O3

linear_model.pb.h: linear_model.proto
	protoc --cpp_out=. linear_model.proto

linear_model.pb.cc: linear_model.pb.h

linear_model.pb.o: linear_model.pb.h linear_model.pb.cc
	g++ $(CXXFLAGS) -c linear_model.pb.cc

binomial_perceptron.o: binomial_perceptron.h binomial_perceptron.cc \
                       linear_model.pb.h linear_model.pb.cc table.h weight.h
	g++ $(CXXFLAGS) -c binomial_perceptron.cc

multinomial_perceptron.o: multinomial_perceptron.h multinomial_perceptron.cc \
                          linear_model.pb.h linear_model.pb.cc table.h weight.h
	g++ $(CXXFLAGS) -c multinomial_perceptron.cc

unittest.o: unittest.cc binomial_model.h multinomial_model.h decoder.h
	g++ $(CXXFLAGS) -c unittest.cc

unittest: unittest.o \
          linear_model.pb.o binomial_perceptron.o multinomial_perceptron.o
	g++ $(CXXFLAGS) $(LDFLAGS) \
                unittest.o \
                linear_model.pb.o \
                binomial_perceptron.o \
                multinomial_perceptron.o \
                -lprotobuf -lpthread -o unittest

test: unittest
	./unittest
	protoc \
            --decode=perceptronix.DenseBinomialPerceptronProto \
            linear_model.proto \
             < db.pb
	protoc \
             --decode=perceptronix.SparseBinomialPerceptronProto \
             linear_model.proto \
             < sb.pb
	protoc \
            --decode=perceptronix.DenseMultinomialPerceptronProto \
            linear_model.proto \
            < dm.pb
	protoc \
             --decode=perceptronix.SparseDenseMultinomialPerceptronProto \
             linear_model.proto \
             < sdm.pb
	protoc \
             --decode=perceptronix.SparseMultinomialPerceptronProto \
             linear_model.proto \
             < sm.pb

clean:
	$(RM) -r *.o *.pb unittest

.PHONY: test clean
