/*
 * Created by Ketan Date
 */

#ifndef _FUNCTIONS_STEP_4_H
#define _FUNCTIONS_STEP_4_H

#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "structures.h"
#include "variables.h"
#include "helper_utils.h"

void reversePass(VertexData *d_row_data_dev, VertexData *d_col_data_dev, int N, unsigned int devid);
void augmentationPass(Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, int N,unsigned int devid);


__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, int *d_visited, int offset, int size);
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int offset, int size);
__global__ void kernel_reverseTraversal(Array d_col_vertices, VertexData d_row_data, VertexData d_col_data);
__global__ void kernel_augmentation(int *d_row_assignments, int *d_col_assignments, Array d_row_vertices, VertexData d_row_data, VertexData d_col_data);

__device__ void __reverse_traversal(int *d_row_visited, int *d_row_children, int *d_col_children, int *d_row_parents, int *d_col_parents, int init_colid);
__device__ void __augment(int *d_row_assignments, int *d_col_assignments, int *d_row_children, int *d_col_children, int init_rowid);

#endif
