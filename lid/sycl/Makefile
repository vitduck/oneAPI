CC        := clang++
LLVM_ROOT := /scratch/optpar01/apps/build/llvm/unstable/

SOURCE  := main.cpp
OBJ     := $(SOURCE:.cpp=.o)

CFLAGS := -std=c++17 \
	     -I${LLVM_ROOT}/include \
	     -fsycl \
         -fsycl-unnamed-lambda \
         -fsycl-targets=nvptx64-nvidia-cuda-sycldevice 

BUILD   := K40
WG_SIZE := 64

ifeq ($(BUILD), K40) 
	BIN     = lid_k40.x
	CFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_35"
endif 
	
ifeq ($(BUILD), V100) 
	BIN     = lid_v100.x
	CFLAGS += -Xsycl-target-backend "--cuda-gpu-arch=sm_70"
endif 

%.o: %.cpp 
	$(CC) $(CFLAGS) -DBLOCK_SIZE=$(WG_SIZE) -c $< -o $@

lid: $(OBJ) 
	$(CC) $(CFLAGS) $(OBJ) -o $(BIN) $(LDFLAGS)

clean:
	rm -rf $(OBJ) $(BIN) *.dat
