CXX		 = clang++

CXXFLAGS := -O3 \
            -std=c++17  \
		    -fsycl \
		    -fsycl-unnamed-lambda \
		    -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
		    -I/scratch/optpar01/apps/build/oneMKL/include \
            -I/scratch/optpar01/apps/build/llvm/unstable/include \
		   
LDFLAGS  = -L/scratch/optpar01/apps/build/oneMKL/lib 

.PHONY:all 

all:gemm

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

gemm: onemkl.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o gemm_onemkl.x onemkl.o -lonemkl

clean:
	rm -f *.o
