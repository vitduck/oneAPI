CXX		 = dpcpp
CXXFLAGS = -O2

TARGET  = intel
PROGRAM = sgemm-onemkl.x

# devcloud 
ifeq ($(TARGET), intel )
	CXXFLAGS += -fsycl \
				-I/home/u66264/apps/build/oneMKL/include

	LDFLAGS = -L/home/u66264/apps/build/oneMKL/lib
endif

ifeq ($(TARGET), nvidia)
	ARCH     = V100
	PROGRAM  = sgemm-cublas_$(ARCH).x
	CXX      = clang++
	CXXFLAGS = -fsycl \
		       -fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
		       -I/scratch/optpar01/apps/build/oneMKL/include 

	LDFLAGS = -L/scratch/optpar01/apps/build/oneMKL/lib 

ifeq ($(ARCH), V100)
	CXXFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_70"
else ifeq ($(ARCH), K40)
	CXXFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_35"
endif

endif

.PHONY: all 

all: sgemm

getopt.o: ../utils/getopt.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

onemkl.o: sgemm-onemkl.cpp 
	$(CXX) $(CXXFLAGS) -c $< -o $@

sgemm: getopt.o sgemm-onemkl.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROGRAM) $^ -lonemkl

clean:
	rm -f *.o
