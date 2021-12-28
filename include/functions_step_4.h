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
#include "variables.h"
#include "structures.h"
#include "helper_utils.h"
#include "exclusive_scan.h"
#include "reduction.h"

void compactEdgesCSC (Predicates &d_edge_predicates_csc);
void forwardPass (VertexData &d_row_data, VertexData &d_col_data);
void reversePass (VertexData &d_row_data, VertexData &d_col_data);
void augmentationPass (VertexData &d_row_data, VertexData &d_col_data);
void compactColumnVertices( Array &d_vertices_csc_out, Array &d_vertices_csc_in);
void traverseAndExpand( Array &d_vertices_csc_out, Array &d_vertices_csc_in, VertexData &d_row_data, VertexData &d_col_data);

__global__ void kernel_colInitialization ( short *d_vertex_ids, short *d_visited, int *d_ptrs, int N);
__global__ void kernel_edgePredicateConstructionCSC (Predicates d_edge_predicates_csc, char *d_masks, int N);
__global__ void kernel_edgeScatterCSC (CompactEdges d_edges_csc, Predicates d_edge_predicates_csc, int M, int N );
__global__ void kernel_vertexPredicateConstructionCSC (Predicates d_vertex_predicates, Array d_vertices_csc_in, short *d_visited);
__global__ void kernel_vertexScatterCSC (short *d_vertex_ids_csc, short *d_vertex_ids, Predicates d_vertex_predicates);
__global__ void kernel_vertexAllocationConstructionCSC (Predicates d_vertex_allocations, Array d_vertices_csc_in, int * d_ptrs);
__global__ void kernel_forwardTraversal (Array d_vertices_csc_out, Array d_vertices_csc_in, Predicates d_vertex_allocations, VertexData d_row_data, VertexData d_col_data, CompactEdges d_edges_csr, CompactEdges d_edges_csc, Vertices d_vertices);
__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, short *d_visited, int size);
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int size);
__global__ void kernel_reverseTraversal (Array d_row_vertices, CompactEdges d_edges_csc, VertexData d_row_data, VertexData d_col_data);
__global__ void kernel_augmentation(char *d_masks, Array d_col_vertices, VertexData d_row_data, VertexData d_col_data, int N);

__device__ void __forward_traversal (short *d_row_parents, short *d_col_parents, short *d_row_visited, short *d_col_visited, short *new_frontier, short *d_start_ptr, short *d_end_ptr, Vertices d_vertices, short parent_col_id);
__device__ void __reverse_traversal (short *d_col_visited, short *d_row_children, short *d_col_children, short *d_row_parents, short *d_col_parents, int init_rowid);
__device__ void __augment (char *d_masks, short *d_row_children, short *d_col_children, int init_colid, int N);

#endif