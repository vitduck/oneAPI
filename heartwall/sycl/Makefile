CXX      = dpcpp
CXXFLAGS = -O2 -DRD_WG_SIZE=$(WG_SIZE)

TARGET   = intel
WG_SIZE  = 256
PROGRAM  = heartwall-$(TARGET)-wg_$(WG_SIZE).x

ifdef OUTPUT
	CXXFLAGS += -DOUTPUT
endif

# for nvidia devices
ifeq ($(TARGET), nvidia)
	ARCH      = V100
	PROGRAM   = heartwall-sycl_$(ARCH)-wg_$(WG_SIZE).x
	CXX       = clang++
	LLVM_ROOT = /scratch/optpar01/apps/build/llvm/2021-06-08/
	DPCT_ROOT = /scratch/optpar01/apps/build/oneAPI/dpcpp-ct/2021.2.0/
	CXXFLAGS += -I$(LLVM_ROOT)/include \
				-I$(DPCT_ROOT)/include \
				-I/scratch/optpar01/apps/build/oneAPI/dpcpp-ct/2021.2.0/include \
				-fsycl -fsycl-unnamed-lambda -fsycl-targets=nvptx64-nvidia-cuda-sycldevice

ifeq ($(ARCH), V100)
	CXXFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_70"
else ifeq ($(ARCH), K40)
	CXXFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_35"
endif
endif

heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CXX) $(CXXFLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -o $(PROGRAM)

main.o: main.cpp kernel.cpp define.c
	$(CXX) $(CXXFLAGS) main.cpp -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

clean:
	rm -f *.o AVI/*.o result.txt
