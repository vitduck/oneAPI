#include <iostream>
#include <cstring>
#include <sstream>
#include <unistd.h>

// GNU getopt
void parseArguments(int argc, char *argv[]) { 
    extern int SIZE;  
    extern int LOOP;    

    int cmd; 
    char* opt_size = 0; 
    char* opt_loop = 0; 
    
    while ( (cmd = getopt(argc, argv, "s:n:")) != -1 ) { 
        switch (cmd) { 
            case 's': 
                std::stringstream(optarg) >> SIZE; 
                break; 
            case 'n': 
                std::stringstream(optarg) >> LOOP; 
                break; 
        } 
    } 
} 
