/*
 * Created by Ketan Date
 */

#ifndef _FUNCTIONS_STEP_1_H
#define _FUNCTIONS_STEP_1_H

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "helper_utils.h"

void computeInitialAssignments(Matrix *d_costs, Vertices *d_vertices_dev, int N, unsigned int devid);

__global__ void kernel_computeInitialAssignments(double *d_costs, double *d_row_duals, double *d_col_duals, int* d_row_assignments, int* d_col_assignments, int *d_row_lock, int *d_col_lock, int N) ;


#endif
