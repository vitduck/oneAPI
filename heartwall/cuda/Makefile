CC      = nvcc
CFLAGS  = -O2 -DRD_WG_SIZE=$(WG_SIZE)
PROGRAM = heartwall-cuda-wg_$(WG_SIZE).x

WG_SIZE = 256

ifdef OUTPUT
	CFLAGS += -DOUTPUT
endif

heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) $(CFLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -o $(PROGRAM)

main.o: main.cu kernel.cu define.c
	$(CC) $(CFLAGS) main.cu -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

clean:
	rm -f *.o AVI/*.o result.txt
