/*
 * Created by Ketan Date
 */

#ifndef _FUNCTIONS_STEP_2_H
#define _FUNCTIONS_STEP_2_H

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "variables.h"
#include "structures.h"
#include "helper_utils.h"

void initializeStep2(Vertices h_vertices_dev, Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, int N, unsigned int devid);
int computeRowCovers(Vertices *d_vertices_dev, int N, unsigned int devid);
void updateRowCovers(Vertices *d_vertices_dev, int *h_row_covers, int N, unsigned int devid);

__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers, int row_count);

#endif
