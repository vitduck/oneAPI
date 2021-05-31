CC      = nvcc
CFLAGS := -O3 \
	      -std=c++14 \
		  -gencode=arch=compute_35,code=sm_35 \
          -gencode=arch=compute_70,code=sm_70 \
	      -DOUTPUT

all: heartwall

./main.o: ./main.h ./main.cu
	$(CC) ./main.cu -I./util/timer/ -I./util/file/ -c -o ./main.o 

./kernel/kernel.o: ./kernel/kernel.h ./kernel/kernel.cu
	$(CC) $(CFLAGS) ./kernel/kernel.cu -c -o ./kernel/kernel.o 

./util/avi/avilib.o: ./util/avi/avilib.h ./util/avi/avilib.c
	$(CC) $(CFLAGS) ./util/avi/avilib.c -I./util/avi -c -o ./util/avi/avilib.o 

./util/avi/avimod.o: ./util/avi/avilib.h ./util/avi/avimod.c
	$(CC) $(CFLAGS) ./util/avi/avimod.c -I./util/avi -c -o ./util/avi/avimod.o 

./util/file/file.o:	./util/file/file.h ./util/file/file.c
	$(CC) $(CFLAGS) ./util/file/file.c -I./util/file -I./util/timer	-c -o ./util/file/file.o 

./util/timer/timer.o: ./util/timer/timer.h ./util/timer/timer.c 
	$(CC) $(CFLAGS) ./util/timer/timer.c -I./util/timer -c -o ./util/timer/timer.o 

heartwall: ./main.o ./kernel/kernel.o ./util/avi/avilib.o ./util/avi/avimod.o ./util/file/file.o ./util/timer/timer.o 
	$(CC) $(CFLAGS) ./main.o ./kernel/kernel.o ./util/avi/avilib.o ./util/avi/avimod.o ./util/file/file.o ./util/timer/timer.o -o heartwall

clean:
	rm -rf *.o ./kernel/*.o ./util/avi/*.o ./util/file/*.o ./util/timer/*.o 
