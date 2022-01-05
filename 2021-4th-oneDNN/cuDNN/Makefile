CC      = nvcc
CFLAGS  = -O2
LDFLAGS = -lcudnn

buffer: 
	$(CC) $(CFLAGS) sigmoid.cpp -o buffer.x $(LDFLAGS)

usm: 
	$(CC) $(CFLAGS) sigmoid_usm.cpp -o usm.x $(LDFLAGS)

clean: 
	rm *.x
