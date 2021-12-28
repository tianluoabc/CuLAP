/*
 * Developed by Ketan Date
 */

#ifndef _EXCLUSIVE_SCAN
#define _EXCLUSIVE_SCAN

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_utils.h"
#include "structures.h"
#include "variables.h"

int recursiveScan (int *d_in, int size);

__global__ void kernel_exclusiveScan (int *d_in, int *d_block_sum, int size);
__global__ void kernel_uniformUpdate (int *d_in, int *d_block_sum, int size);

#endif