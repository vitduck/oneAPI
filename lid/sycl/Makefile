CC     = dpcpp
CFLAGS = -std=c++14

SOURCE = main.cpp
OBJ    = $(SOURCE:.cpp=.o)

WG_SIZE  = 64
BIN     := lid-wg_$(WG_SIZE).x

%.o: %.cpp 
	$(CC) $(CFLAGS) -DBLOCK_SIZE=$(WG_SIZE) -c $< -o $@

lid: $(OBJ) 
	$(CC) $(CFLAGS) $(OBJ) -o $(BIN)

clean:
	rm -rf $(OBJ)
