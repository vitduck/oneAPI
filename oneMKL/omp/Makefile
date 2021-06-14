CXX		 = icx
CXXFLAGS = -fiopenmp -fopenmp-targets=spir64 -mllvm -vpo-paropt-use-raw-dev-ptr -fsycl -DMKL_ILP64 -m64
LDFLAGS  = -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lOpenCL -lpthread -ldl -lm

.PHONY: all

all: sgemm

getopt.o: ../utils/getopt.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

onemkl.o: sgemm-openmp.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

sgemm: getopt.o sgemm-openmp.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o sgemm-openmp.x $^

clean:
	rm -f *.o
