CC      = clang++
CFLAGS  = -O2 -fsycl -I/scratch/optpar01/apps/build/oneDNN/2.5/include 
LDFLAGS = -L/scratch/optpar01/apps/build/oneDNN/2.5/lib64/ -ldnnl

sigmoid: 
	$(CC) $(CFLAGS) sigmoid.cpp -o sigmoid.x $(LDFLAGS)

clean: 
	rm -f *.x
