/*
 * Created by Ketan Date
 */

#ifndef _FUNCTIONS_STEP_5_H
#define _FUNCTIONS_STEP_5_H

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "variables.h"
#include "structures.h"
#include "helper_utils.h"


void computeTheta(double &h_device_min, Matrix *d_costs_dev, Vertices *d_vertices_dev, VertexData *d_col_data_dev, int N, unsigned int devid);

__global__ void kernel_dualUpdate_2(double d_min_val, double *d_row_duals, double *d_col_duals, int *d_row_cover, int *d_col_cover, int row_start, int row_count, int N);

#endif
