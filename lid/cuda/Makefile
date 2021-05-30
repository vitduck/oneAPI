CC        = nvcc

SOURCE = main.cu
OBJ    = $(SOURCE:.cu=.o)

CFLAGS := -std=c++14 -Xcompiler -Wall \
          -gencode=arch=compute_35,code=sm_35 \
          -gencode=arch=compute_70,code=sm_70 

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

lid.x: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@

clean:
	rm -rf $(OBJ) lid.x *.dat
