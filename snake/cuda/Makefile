CC     = nvcc
CFLAGS = -DNuints=8 \
         -O3 \
         -std=c++14 \
		 -gencode=arch=compute_35,code=sm_35 \
         -gencode=arch=compute_70,code=sm_70 

OBJ = main.o

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

snake: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o snake.x

clean:
	rm -rf $(OBJ)

