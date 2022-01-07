CC      = clang++
CFLAGS  = -O2 -fsycl -I/scratch/optpar01/apps/build/oneDNN/2.5/include 
LDFLAGS = -L/scratch/optpar01/apps/build/oneDNN/2.5/lib64/ -ldnnl

OPENCV_CFLAGS  = -I/scratch/optpar01/apps/build/opencv/4.5.5/include/opencv4
OPENCV_LDFLAGS = -L/scratch/optpar01/apps/build/opencv/4.5.5/lib64 -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

all: edge sigmoid 

edge: 
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) conv.cpp -o edge.x $(LDFLAGS) $(OPENCV_LDFLAGS)

sigmoid: 
	$(CC) $(CFLAGS) sigmoid.cpp -o sigmoid.x $(LDFLAGS)

clean: 
	rm -f *.x
