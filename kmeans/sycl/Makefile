CXX      = dpcpp
CXXFLAGS = -O3

TARGET  = intel
PROGRAM = kmeans-$(TARGET).x

# for nvidia devices
ifeq ($(TARGET), nvidia)
	ARCH      = V100
	CXX       = clang++
	PROGRAM   = kmeans-sycl_$(ARCH).x
	LLVM_ROOT = /scratch/optpar01/apps/build/llvm/2021-06-08/
	DPCT_ROOT = /scratch/optpar01/apps/build/oneAPI/dpcpp-ct/2021.2.0/
	CXXFLAGS += -I$(LLVM_ROOT)/include \
				-I$(DPCT_ROOT)/include \
				-fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda-sycldevice

ifeq ($(ARCH), V100)
        CXXFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_70"
else ifeq ($(ARCH), K40)
        CXXFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_35"
endif

endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -c

kmeans: cluster.o getopt.o read_input.o rmse.o kmeans.cpp
	$(CXX) $(CXXFLAGS) $^ -o $(PROGRAM)

clean:
	rm -f *.o
