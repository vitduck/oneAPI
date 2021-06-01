CXX		 = dpcpp

CXXFLAGS = -O3 \
           -std=c++17 \
	       -fsycl \
		   -I/home/u66264/apps/build/oneMKL/include
		    		   
LDFLAGS = -L/home/u66264/apps/build/oneMKL/lib

.PHONY: all 

all: sgemm

getopt.o: ../utils/getopt.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

onemkl.o: onemkl.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

sgemm: getopt.o onemkl.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o sgemm_onemkl.x $^ -lonemkl

clean:
	rm -f *.o
