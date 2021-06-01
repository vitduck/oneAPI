CXX		 = nvcc
CXXFLAGS = -O3 -lcublas 

all: sgemm

cublas.o: cublas.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@

getopt.o: ../utils/getopt.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

sgemm: cublas.o getopt.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o sgemm_cublas.x $^

clean:
	rm -f *.o
