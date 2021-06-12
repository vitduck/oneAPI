CC     = gcc 
CFLAGS = -O2 -fopenmp -lm

ifdef OUTPUT
	CFLAGS += -DOUTPUT
endif

heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o
	$(CC) $(CFLAGS) main.o ./AVI/avilib.o ./AVI/avimod.o -o heartwall-omp.x

main.o: main.c define.c kernel.c
	$(CC) main.c -I./AVI -c

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

clean:
	rm -f *.o AVI/*.o result.txt
