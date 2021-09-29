/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*
*/


#include <iostream>
#include <fstream>
#include <random>
#include <cuda.h>
#include <thrust/scan.h>
#include "d_vars.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef F_CUTILS_H_
#define F_CUTILS_H_

void generateProblem(float *cost_matrix, int SP, int N, int costrange);

void cudaSafeCall(cudaError_t error, const char *message);
void printMemoryUsage(float memory);

void readFile (const char *filename);
void calculateLinearDims (dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateSquareDims (dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize);
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize);
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, float total_time, int *stepcounts, float *steptimes, const char *logpath, int N);
void exclusiveSumScan (int *arr, int size);
void exclusiveSumScan (long *arr, int size);
int reduceSUM(int *arr, int size);
long reduceSUM(long *arr, int size);
float reduceSUM(float *arr, int size);
bool reduceOR(bool *arr, int size);
float reduceMIN(float *arr, int size);

void printDebugArray(int *d_array, int size, const char *name);
void printDebugMatrix(float *d_matrix, int rowsize, int colsize, const char *name);
void printDebugMatrix(int *d_matrix, int rowsize, int colsize, const char *name);
void printHostArray(int *h_array, int size, const char *name);
void printHostArray(float *h_array, int size, const char *name);
void printDebugArray(long *d_array, int size, const char *name);
void printDebugArray(float *d_array, int size, const char *name);
void printHostArray(long *h_array, int size, const char *name);
void printHostMatrix(float *h_matrix, int rowsize, int colsize, const char *name);
void printHostMatrix(int *h_matrix, int rowsize, int colsize, const char *name);

void printHostArray_tofile(float *h_array, int size, const char *name, std::ofstream &out);
void printHostMatrix_tofile(float *h_matrix, int rowsize, int colsize, const char *name, std::ofstream &out);

void readFile(float *cost_matrix, const char *filename);

void split(int *array, int val, int size);
void split(long *array, long val, int size);

__global__ void kernel_memSet(int *array, int val, int size);
__global__ void kernel_memSet(float *array, float val, int size);
__global__ void kernel_memSet(long *array, long val, int size);


#endif /* F_CUTILS_H_ */
