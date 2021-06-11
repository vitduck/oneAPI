CXX      = dpcpp
CXXFLAGS = -O2 -DBLOCK_SIZE=$(WG_SIZE)

TARGET   = intel
WG_SIZE  = 128

PROGRAM  = lid-$(TARGET)-wg_$(WG_SIZE).x

# bug in devcloud with default_selector/cpu
ifeq ($(TARGET), intel)
CXXFLAGS += -DDEVCLOUD
endif

# for nvidia devices
ifeq ($(TARGET), nvidia)
PROGRAM   = lid-sycl_$(ARCH)-wgsize_$(WG_SIZE).x
CXX       = clang++
LLVM_ROOT = /scratch/optpar01/apps/build/llvm/2021-06-08/
CXXFLAGS += -I$(LLVM_ROOT)/include \
		    -fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda-sycldevice

# fat bin is not yet supported 
ARCH = V100

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
