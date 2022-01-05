CC     = dpcpp
CFLAGS = -DNuints=8 \
         -O3 \
         -std=c++14 
		 
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

snake: main.o
	$(CC) $(CFLAGS) main.o -o snake.x

clean:
	rm -rf *.o
