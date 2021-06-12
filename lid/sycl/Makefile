CXX      = dpcpp
CXXFLAGS = -O2 -DBLOCK_SIZE=$(WG_SIZE)

TARGET   = intel
WG_SIZE  = 128
PROGRAM  = lid-$(TARGET)-wg_$(WG_SIZE).x

# for nvidia devices
ifeq ($(TARGET), nvidia)
	ARCH      = V100
	PROGRAM   = lid-sycl_$(ARCH)-wg_$(WG_SIZE).x
	CXX       = clang++
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

SOURCE = main.cpp
OBJ    = $(SOURCE:.cpp=.o)

lid: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(PROGRAM)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ)
