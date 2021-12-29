/*
 * Created by Ketan Date
 */

#ifndef _FUNCTIONS_STEP_0_H
#define _FUNCTIONS_STEP_0_H

#include "cuda.h"
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "helper_utils.h"

void initialReduction(Matrix *d_costs, int N, unsigned int devid);

__global__ void kernel_rowReduction(double *d_costs, double *d_row_duals, int N);
__global__ void kernel_columnReduction(double *d_costs, double *d_col_duals, int N);

#endif
