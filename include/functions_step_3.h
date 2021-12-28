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
#include "variables.h"
#include "structures.h"
#include "helper_utils.h"
#include "exclusive_scan.h"
#include "reduction.h"

void executeZeroCover(int &next);
void compactEdgesCSR (Predicates &d_edge_predicates_csr);
void compactRowVertices( Array &d_vertices_csr_out, Array &d_vertices_csr_in);
void coverZeroAndExpand( Array &d_vertices_csr_out, Array &d_vertices_csr_in, int &h_next);

__global__ void kernel_rowInitialization ( short *d_vertex_ids, short *d_visited, int *d_ptrs,  int *d_covers, int N);
__global__ void kernel_edgePredicateConstructionCSR (Predicates d_edge_predicates_csr, char *d_masks, int N);
__global__ void kernel_edgeScatterCSR (CompactEdges d_edges_csr, Predicates d_edge_predicates_csr, int M, int N );
__global__ void kernel_vertexPredicateConstructionCSR (Predicates d_vertex_predicates, Array d_vertices_csr_in, short *d_visited);
__global__ void kernel_vertexScatterCSR (short *d_vertex_ids_csr, short *d_vertex_ids, Predicates d_vertex_predicates);
__global__ void kernel_vertexAllocationConstructionCSR (Predicates d_vertex_allocations, Array d_vertices_csr_in, int * d_ptrs);
__global__ void kernel_coverAndExpand (int *d_next, Array d_vertices_csr_out, Array d_vertices_csr_in, Predicates d_vertex_allocations, CompactEdges d_edges_csr, CompactEdges d_edges_csc, Vertices d_vertices, Edges d_edges, int N);

__device__ void __update_covers (Vertices d_vertices, int *d_next, short *d_row_visited, short *d_col_visited, short *new_frontier, short *d_start_ptr, short *d_end_ptr, char *d_masks, short vertexid, int N);


#endif