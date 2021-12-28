/*
 * Developed by Ketan Date
 */

#ifndef _EXCLUSIVE_SCAN_PADDED
#define _EXCLUSIVE_SCAN_PADDED

#include <cuda.h>
#include "helper_utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "variables.h"

int recursiveMin (int *d_in, int size);
int recursiveSum (int *d_in, int size);

__global__ void kernel_parallelMin (int *d_in, int *d_block_min, int size);
__global__ void kernel_parallelSum (int *d_in, int *d_block_sum, int size);

#endif