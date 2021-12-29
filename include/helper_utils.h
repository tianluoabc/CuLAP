/*
 * Developed by Ketan Date
 */

#ifndef _HELPER_UTILS_H
#define _HELPER_UTILS_H

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <thrust/scan.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "variables.h"
#include "sfmt.h"
#include <cmath>

void cudaSafeCall(cudaError_t error, const char *message);
void printMemoryUsage(double memory);

void readFile(const char *filename);
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize);
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize);

void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath, int N);
void exclusiveSumScan(int *array, int size);
void exclusiveSumScan(long *array, int size);
int reduceSUM(int *array, int size);
long reduceSUM(long *array, int size);
bool reduceOR(bool *array, int size);
void printDebugArray(int *d_array, int size, const char *name);
void printDebugArray(double *d_array, int size, const char *name);
void printDebugMatrix(double *d_matrix, int rowsize, int colsize, const char *name);

void printHostArray(int *h_array, int size, const char *name);
void printHostArray(double *h_array, int size, const char *name);
void printDebugArray(long *d_array, int size, const char *name);
void printHostArray(long *h_array, int size, const char *name);

void generateProblem(double *cost_matrix, int SP, int N, int costrange);
void readFile(double *cost_matrix, const char *filename, int spcount);

double reduceMIN(double *array, int size);

#endif
