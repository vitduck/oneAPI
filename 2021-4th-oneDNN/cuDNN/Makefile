CC      = nvcc
CFLAGS  = -O2
LDFLAGS = -lcudnn

OPENCV_CFLAGS  = -I/scratch/optpar01/apps/build/opencv/4.5.5/include/opencv4
OPENCV_LDFLAGS = -L/scratch/optpar01/apps/build/opencv/4.5.5/lib64 -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

all: edge buffer usm

edge: 
	$(CC) $(CFLAGS) $(OPENCV_CFLAGS) conv.cu -o edge.x $(LDFLAGS) $(OPENCV_LDFLAGS)

buffer: 
	$(CC) $(CFLAGS) sigmoid.cu -o sigmoid_buffer.x $(LDFLAGS)

usm: 
	$(CC) $(CFLAGS) sigmoid_usm.cu -o sigmoid_usm.x $(LDFLAGS)

clean: 
	rm *.x
