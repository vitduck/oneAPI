CXX		 = clang++

CXXFLAGS := -O3 \
            -std=c++17  \
		    -fsycl \
		    -fsycl-unnamed-lambda \
		    -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
		    -I/scratch/optpar01/apps/build/oneMKL/include \
            -I/scratch/optpar01/apps/build/llvm/unstable/include \
		   
LDFLAGS  = -L/scratch/optpar01/apps/build/oneMKL/lib 

.PHONY: all 

all: sgemm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

sgemm: getopt.o onemkl.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o sgemm_onemkl.x $^ -lonemkl

clean:
	rm -f *.o
