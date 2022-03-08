#include "Matrix.h"
#include "profile.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <random>

void generateMatrix(const std::string& filename, size_t row, size_t col) {
    std::ofstream file(filename);
    if(file.is_open() == false)
        throw std::runtime_error("Can't open file");

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(-10000, 10000);

    file << row << ' ' << col << '\n';
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            file << dist(rng);

            if(j != col - 1)
                file << ' ';
        }
        file << '\n';
    }

    file.close();
}

template <typename T>
std::vector<std::vector<T>> readMatrix(const std::string& filename) {
    std::ifstream in(filename);

    if(in.is_open() == false)
        throw std::runtime_error("Can't open file");

    size_t r;
    size_t c;
    in >> r >> c;

    std::vector<std::vector<T>> v(r, std::vector<T>(c));

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            in >> v [i][j];
        }
    }

    return v;
}

int main() {
    std::string filename("../input.txt");

    size_t row_size = 1024;
    size_t col_size = 1024;

    Matrix<double> matrix;
    Matrix<double> inverse_matrix;
    Matrix<double> inverse_matrix_;

    for(uint8_t i = 0; i < 4; ++i, row_size *= 2, col_size *= 2) {
        generateMatrix(filename, row_size, col_size);
        matrix = Matrix(readMatrix<double>(filename));

        std::cout << "Matrix dimension: " << row_size << ' ' << col_size << '\n';

        {
            LOG_DURATION("Single thread")
            inverse_matrix = matrix.inverseEMethod();
        }
        {
            LOG_DURATION("Omp multithreading thread")
            inverse_matrix_ = matrix.inverseEMethod_();
        }

        std::cout << ((inverse_matrix == inverse_matrix_) ? "all fine\n" : "multithreading got diff answer\n");
    }


    return 0;
}