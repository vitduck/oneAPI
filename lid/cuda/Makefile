CC      = nvcc
SAMPLES = /scratch/optpar01/apps/build/cuda-samples/
WG_SIZE = 128

CFLAGS  = -O2 \
		  -DBLOCK_SIZE=$(WG_SIZE) \
		  -I$(SAMPLES)/Common \
		  -gencode=arch=compute_35,code=sm_35 \
		  -gencode=arch=compute_70,code=sm_70

SOURCE  = main.cu
OBJ     = $(SOURCE:.cu=.o)
PROGRAM = lid-cuda-wg_$(WG_SIZE).x

lid: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(PROGRAM)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ)
