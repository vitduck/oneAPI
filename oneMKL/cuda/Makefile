CXX		 = nvcc
CXXFLAGS = -O3 -lcublas 

all: sgemm

sgemm.o: sgemm.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

getopt.o: ../utils/getopt.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

sgemm: sgemm.o getopt.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o sgemm-cublas.x $^

clean:
	rm -f *.o
