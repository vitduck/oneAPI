// generate random number between [0,1)
template <typename T> inline T normalized_random_number() { 
    return (T)rand() / ((T)(RAND_MAX)+(T)(1));
}

// generate random matrix
template <typename T> void random_matrix(T *matrix, int m, int n) { 
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i*m+j] = normalized_random_number<T>(); 
} 

// zero matrix
template <typename T> void zero_matrix(T *matrix, int m, int n) { 
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            matrix[i*m+j] = (T) 0.0; 
} 

// print matrix for debug
template <typename T> void print_matrix(T *matrix, std::string name, int m, int n) { 
    std::cout << name << " =" << std::endl; 

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrix[i*m+j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// GNU getopt
void parseArguments(int argc, char *argv[]); 
