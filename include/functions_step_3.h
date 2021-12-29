/*
 * Created by Ketan Date
 */

#ifndef _FUNCTIONS_STEP_3_H
#define _FUNCTIONS_STEP_3_H

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "variables.h"
#include "helper_utils.h"

void executeZeroCover(Matrix *d_costs_dev, Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, bool *h_flag, int N, unsigned int devid);

void compactRowVertices(VertexData *d_row_data_dev, Array &d_vertices_csr_out, Array &d_vertices_csr_in, unsigned int devid);

void coverZeroAndExpand(Matrix *d_costs_dev, Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, Array &d_vertices_csr_in, bool *h_flag, int N, unsigned int devid);
void computeAdjacency(void);


__global__ void kernel_rowInitialization(int *d_vertex_ids, int *d_visited, int *d_covers, int row_start, int row_count);
__global__ void kernel_offsetRowPointers(long *d_ptrs, long d_offset, long g_offset, int row_count);
__global__ void kernel_vertexPredicateConstructionCSR(Predicates d_vertex_predicates, Array d_vertices_csr_in, int *d_visited);
__global__ void kernel_vertexScatterCSR(int *d_vertex_ids_csr, int *d_vertex_ids, int *d_visited, Predicates d_vertex_predicates);
__global__ void kernel_coverAndExpand (bool *d_flag, Array d_vertices_csr_in, Matrix d_costs, Vertices d_vertices, VertexData d_row_data, VertexData d_col_data, int N);

__device__ void __traverse(Matrix d_costs, Vertices d_vertices, bool *d_flag, int *d_row_parents, int *d_col_parents, int *d_row_visited, int *d_col_visited, double *d_slacks, int *d_start_ptr, int *d_end_ptr, int colid, int N);

#endif
