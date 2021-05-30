CC     = nvcc
CFLAGS = -std=c++14 \
         -gencode=arch=compute_35,code=sm_35 \
         -gencode=arch=compute_70,code=sm_70 

SOURCE = main.cu
OBJ    = $(SOURCE:.cu=.o)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

lid.x: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

clean:
	rm -rf $(OBJ)
