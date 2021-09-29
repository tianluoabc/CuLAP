/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*
*/


#include <cuda.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "d_structs.h"
#include "f_cutils.h"

#ifndef F_CULAP_H_
#define F_CULAP_H_

void initialReduction(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

void computeInitialAssignments(Matrix &d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

int computeRowCovers(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);

void executeZeroCover(Matrix &d_costs_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, bool *h_flag, int SP, int N, unsigned int devid);
void compactRowVertices(CompactEdges &d_rows_csr_dev, VertexData &d_row_data_dev, long &M, int SP, int N, unsigned int devid);
void coverZeroAndExpand(Matrix &d_costs_dev, CompactEdges &d_rows_csr_dev, Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, bool *h_flag, int SP, int N, unsigned int devid);

void reversePass(VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);
void augmentationPass(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);

void dualUpdate(Vertices &d_vertices_dev, VertexData &d_row_data_dev, VertexData &d_col_data_dev, int SP, int N, unsigned int devid);

void calcObjValDual(float *d_obj_val, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);

void calcObjValPrimal(float *d_obj_val, float *d_costs, Vertices &d_vertices_dev, int SP, int N, unsigned int devid);


__global__ void kernel_rowReduction(float *d_costs, float *d_row_duals, float *d_col_duals, int SP, int N);
__global__ void kernel_columnReduction(float *d_costs, float *d_row_duals, float *d_col_duals, int SP, int N);

__global__ void kernel_computeInitialAssignments(float *d_costs, float *d_row_duals, float *d_col_duals, int* d_row_assignments, int* d_col_assignments, int *d_row_lock, int *d_col_lock, int SP, int N);

__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers,  int *d_row_visited, int SP, int N);

__global__ void kernel_rowPredicateConstructionCSR(Predicates d_row_predicates_csr, int *d_row_visited, int SP, int N);
__global__ void kernel_rowScatterCSR(CompactEdges d_row_vertices_csr, Predicates d_row_predicates_csr, long M, int SP, int N);
__global__ void kernel_coverAndExpand(bool *d_flag, CompactEdges d_row_vertices_csr, Matrix d_costs, Vertices d_vertices, VertexData d_row_data, VertexData d_col_data, int SP, int N);


__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, int *d_visited, int size);
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int size);
__global__ void kernel_reverseTraversal(Array d_col_vertices, VertexData d_row_data, VertexData d_col_data);
__global__ void kernel_augmentation(int *d_row_assignments, int *d_col_assignments, Array d_row_vertices, VertexData d_row_data, VertexData d_col_data, int N);

__global__ void kernel_dualUpdate_1(float *d_sp_min, float *d_col_slacks, int *d_col_covers, int SP, int N);
__global__ void kernel_dualUpdate_2(float *d_sp_min, float *d_row_duals, float *d_col_duals, float *d_col_slacks, int *d_row_covers, int *d_col_covers, int *d_row_visited, int *d_col_parents, int SP, int N);

__global__ void kernel_calcObjValDual(float *d_obj_val_dual, float *d_row_duals, float *d_col_duals, int SP, int N);
__global__ void kernel_calcObjValPrimal(float *d_obj_val_primal, float *d_costs, int *d_row_assignments, int SP, int N);

__device__ void __traverse(Matrix d_costs, Vertices d_vertices, bool *d_flag, int *d_row_parents, int *d_col_parents, int *d_row_visited, int *d_col_visited, int *d_start_ptr, int *d_end_ptr, int spid, int colid, int SP, int N);

__device__ void __reverse_traversal(int *d_row_visited, int *d_row_children, int *d_col_children, int *d_row_parents, int *d_col_parents, int COLID);
__device__ void __augment(int *d_row_assignments, int *d_col_assignments, int *d_row_children, int *d_col_children, int ROWID, int N);


#endif /* F_CULAP_H_ */
