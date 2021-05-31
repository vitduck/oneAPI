CC     = nvcc
CFLAGS = -O3 \
         -std=c++14 \
         -gencode=arch=compute_35,code=sm_35 \
         -gencode=arch=compute_70,code=sm_70 

%.o: %.cpp 
	$(CC) $(CFLAGS) $< -c

%.o: %.cu
	$(CC) $(CFLAGS) $< -c

kmeans: cluster.o getopt.o read_input.o rmse.o kmeans.cpp
	$(CC) $(CFLAGS) $^ -o kmeans.x $(LDFLAGS)

clean:
	rm -f *.o 
