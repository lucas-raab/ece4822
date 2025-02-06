#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

void generate_random_matrix(float *matrix, long nrows, long ncols);
void print_matrix(const float *matrix, long nrows, long ncols);
long next_power_of_two(long n);
void pad_matrix(const float *matrix, float *padded_matrix, long original_size, long padded_size);
void unpad_matrix(const float *padded_matrix, float *result_matrix, long original_size, long padded_size);
void matrix_multiply(const long size, const long row_stride_X, const float X[], const long row_stride_Y, const float Y[], const long row_stride_Z, float Z[]);
void matrix_add(const long size, const long row_stride_A, const float A[], const long row_stride_B, const float B[], const long row_stride_result, float result[]);
void matrix_subtract(const long size, const long row_stride_A, const float A[], const long row_stride_B, const float B[], const long row_stride_result, float result[]);
void strassen_multiply(const long size, const long row_stride_X, const float X[], const long row_stride_Y, const float Y[], const long row_stride_Z, float Z[]);

#endif // MATRIX_OPERATIONS_H
