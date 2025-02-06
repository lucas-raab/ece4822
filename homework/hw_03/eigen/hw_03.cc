#include <iostream>
#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

bool genmat(float*** mat, long nrows, long ncols) {
    if (mat == NULL) return false;
    *mat = (float**)malloc(nrows * sizeof(float*));
    for (int i = 0; i < nrows; i++) {
        (*mat)[i] = (float*)malloc(ncols * sizeof(float));
        for (int j = 0; j < ncols; j++) {
            (*mat)[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / 5);
        }
    }
    return true;
}

// Convert dynamically allocated 2D array to Eigen matrix
Eigen::MatrixXf convert_to_eigen_matrix(float** mat, long nrows, long ncols) {
    Eigen::MatrixXf eigen_mat(nrows, ncols);
    for (int i = 0; i < nrows; i++) {
        for (int j = 0; j < ncols; j++) {
            eigen_mat(i, j) = mat[i][j];
        }
    }
    return eigen_mat;
}

int main(int argc, char** argv) {
    srand(static_cast<unsigned int>(time(NULL)));
    if (argc != 4) {
        std::cerr << "ERROR, p01 requires 3 input arguments\n";
        return -1;
    }

    char* out;
    long nrows = strtol(argv[1], &out, 10);
    if (argv[1] == out) {
        std::cerr << "ERROR, nrows not a number\n";
        return -1;
    }

    long ncols = strtol(argv[2], &out, 10);
    if (argv[2] == out) {
        std::cerr << "ERROR, ncols not a number\n";
        return -1;
    }

    long niter = strtol(argv[3], &out, 10);
    if (argv[3] == out) {
        std::cerr << "ERROR, niter not a number\n";
        return -1;
    }



    float **mat1, **mat2, **mat3;
    genmat(&mat1, nrows, ncols);
    genmat(&mat2, ncols, nrows);
    genmat(&mat3, nrows, nrows);

    // Convert to Eigen matrices
    Eigen::MatrixXf eigen_mat1 = convert_to_eigen_matrix(mat1, nrows, ncols);
    Eigen::MatrixXf eigen_mat2 = convert_to_eigen_matrix(mat2, ncols, nrows);
    Eigen::MatrixXf eigen_mat3 = convert_to_eigen_matrix(mat3, nrows, nrows);

    // Perform matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < niter; ++i) {
        eigen_mat3.noalias() = eigen_mat1 * eigen_mat2;
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    printf("%f;%f\n",diff/niter,nrows);

    // Free dynamically allocated memory
    for (int i = 0; i < nrows; i++) free(mat1[i]);
    free(mat1);
    for (int i = 0; i < ncols; i++) free(mat2[i]);
    free(mat2);
    for (int i = 0; i < nrows; i++) free(mat3[i]);
    free(mat3);

    return 0;
}
