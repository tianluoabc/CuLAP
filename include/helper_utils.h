/*
 * Developed by Ketan Date
 */

#ifndef _HELPER_UTILS_H
#define _HELPER_UTILS_H

#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "structures.h"
#include "variables.h"
#include "sfmt.h"

void cudaSafeCall(cudaError_t error, char *message);
void printMemoryUsage(void);
void initialize(void);
void initDeviceVertices(void);
void initDeviceEdges(void);
void finalize(void);
void readFile(const char *filename);
void generateProblem(int problemsize, int costrange);
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size);
void printLog(int prno, int repetition, int costrange, int obj_val, int init_assignments, int total_time, int *stepcounts, float *steptimes, const char *logpath);

#endif