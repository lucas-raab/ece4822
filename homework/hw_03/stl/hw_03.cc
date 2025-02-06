#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using Matrix = std::vector<std::vector<float>>;

Matrix genmat(long nrows, long ncols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 5);

    Matrix mat(nrows, std::vector<float>(ncols));
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            mat[i][j] = dis(gen);
        }
    }
    return mat;
}

Matrix matrix_multiply(const Matrix& mat1, const Matrix& mat2) {
    long nrows = mat1.size();
    long ncols = mat2[0].size();
    long inner = mat2.size();

    Matrix result(nrows, std::vector<float>(ncols, 0.0f));

    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            for (int k = 0; k < inner; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "ERROR, p01 requires 3 input arguments\n";
        return -1;
    }

    long nrows = std::stol(argv[1]);
    long ncols = std::stol(argv[2]);
    long niter = std::stol(argv[3]);

    Matrix mat1 = genmat(nrows, ncols);
    Matrix mat2 = genmat(ncols, nrows);
    Matrix mat3 = genmat(nrows, nrows);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niter; ++i) {
        mat3 = matrix_multiply(mat1, mat2);
    }
    auto end = std::chrono::high_resolution_clock::now();


    
    
    std::chrono::duration<double> diff = end - start;

    printf("%f,%f\n",diff/niter,nrows);
    


    return 0;
}
