/*
 * Created by Ketan Date
 */

#include "include/functions_step_3.h"

// Function for compacting the edges in row major format.
void compactEdgesCSR(Predicates &d_edge_predicates_csr)
{
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;
	calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, N2);

	kernel_edgePredicateConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_edge_predicates_csr, d_edges.masks, N); // construct predicate matrix for edges.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgePredicateConstructionCSR execution");

	//	M = recursiveScan(d_edge_predicates_csr.addresses, d_edge_predicates_csr.size); // construct scatter addresses for edges using exclusive scan.

	thrust::device_ptr<int> ptr(d_edge_predicates_csr.addresses);
	M = thrust::reduce(ptr, ptr + N2);			// calculate total number of edges.
	thrust::exclusive_scan(ptr, ptr + N2, ptr); // exclusive scan for calculating the scatter addresses.

	cudaSafeCall(cudaMalloc(&d_edges_csr.neighbors, M * sizeof(short)), "Error in cudaMalloc d_edges_csr.neighbors");

	kernel_edgeScatterCSR<<<blocks_per_grid, threads_per_block>>>(d_edges_csr, d_edge_predicates_csr, M, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgeScatterCSR execution");
}

// Function for executing recursive zero cover. Returns the next step (Step 4 or Step 5) depending on the presence of uncovered zeros.
void executeZeroCover(int &next)
{
	next = 5;
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	Array d_vertices_csr1, d_vertices_csr2;

	d_vertices_csr1.size = N;
	cudaSafeCall(cudaMalloc(&d_vertices_csr1.elements, N * sizeof(short)), "Error in cudaMalloc d_vertices_csr1.elements"); // compact vertices initialized to the row ids.

	kernel_rowInitialization<<<blocks_per_grid, threads_per_block>>>(d_vertices_csr1.elements, d_edges_csr.is_visited, d_edges_csr.ptrs, d_vertices.row_covers, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexInitialization execution");

	while (true)
	{
		compactRowVertices(d_vertices_csr2, d_vertices_csr1); // compact the current vertex frontier.
		cudaSafeCall(cudaFree(d_vertices_csr1.elements), "Error in cudaFree d_vertices_csr1.elements");
		if (d_vertices_csr2.size == 0)
			break;

		coverZeroAndExpand(d_vertices_csr1, d_vertices_csr2, next); // traverse the frontier, cover zeros and expand.
		cudaSafeCall(cudaFree(d_vertices_csr2.elements), "Error in cudaFree d_vertices_csr2.elements");
		if (d_vertices_csr1.size == 0)
			break;
	}
}

//Function for compacting row vertices. Used in Step 3 (minimum zero cover).
void compactRowVertices(Array &d_vertices_csr_out, Array &d_vertices_csr_in)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	Predicates d_vertex_predicates;

	d_vertex_predicates.size = d_vertices_csr_in.size;
	cudaSafeCall(cudaMalloc(&d_vertex_predicates.predicates, d_vertex_predicates.size * sizeof(bool)), "Error in cudaMalloc d_vertex_predicates.predicates");
	cudaSafeCall(cudaMalloc(&d_vertex_predicates.addresses, d_vertex_predicates.size * sizeof(int)), "Error in cudaMalloc d_vertex_predicates.addresses");
	cudaSafeCall(cudaMemset(d_vertex_predicates.predicates, false, d_vertex_predicates.size * sizeof(bool)), "Error in cudaMemset d_vertex_predicates.predicates");
	cudaSafeCall(cudaMemset(d_vertex_predicates.addresses, 0, d_vertex_predicates.size * sizeof(int)), "Error in cudaMemset d_vertex_predicates.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);
	if (total_blocks > MAX_GRIDSIZE)
		calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);

	kernel_vertexPredicateConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_vertex_predicates, d_vertices_csr_in, d_edges_csr.is_visited);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexPredicateConstruction execution");

	//	d_vertices_csr_out.size = recursiveScan (d_vertex_predicates.addresses, d_vertex_predicates.size);

	thrust::device_ptr<int> ptr(d_vertex_predicates.addresses);
	d_vertices_csr_out.size = thrust::reduce(ptr, ptr + d_vertex_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_predicates.size, ptr);			   // exclusive scan for calculating the scatter addresses.

	if (d_vertices_csr_out.size > 0)
	{
		cudaSafeCall(cudaMalloc(&d_vertices_csr_out.elements, d_vertices_csr_out.size * sizeof(short)), "Error in cudaMalloc d_vertices_csr_out.elements");

		kernel_vertexScatterCSR<<<blocks_per_grid, threads_per_block>>>(d_vertices_csr_out.elements, d_vertices_csr_in.elements, d_vertex_predicates);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexScatter execution");
	}

	cudaSafeCall(cudaFree(d_vertex_predicates.predicates), "Error in cudaFree d_vertex_predicates.predicates");
	cudaSafeCall(cudaFree(d_vertex_predicates.addresses), "Error in cudaFree d_vertex_predicates.addresses");
}

// Function for covering the zeros in uncovered rows and expanding the frontier.
void coverZeroAndExpand(Array &d_vertices_csr_out, Array &d_vertices_csr_in, int &h_next)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	int *d_next;
	cudaSafeCall(cudaMalloc(&d_next, sizeof(int)), "Error in cudaMalloc d_next");
	cudaSafeCall(cudaMemcpy(d_next, &h_next, sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_next");

	Predicates d_vertex_allocations;
	d_vertex_allocations.size = d_vertices_csr_in.size;
	cudaSafeCall(cudaMalloc(&d_vertex_allocations.predicates, d_vertex_allocations.size * sizeof(bool)), "Error in cudaMalloc d_vertex_allocations.predicates");
	cudaSafeCall(cudaMalloc(&d_vertex_allocations.addresses, d_vertex_allocations.size * sizeof(int)), "Error in cudaMalloc d_vertex_allocations.addresses");

	cudaSafeCall(cudaMemset(d_vertex_allocations.predicates, false, d_vertex_allocations.size * sizeof(bool)), "Error in cudaMemset d_vertex_allocations.predicates");
	cudaSafeCall(cudaMemset(d_vertex_allocations.addresses, 0, d_vertex_allocations.size * sizeof(int)), "Error in cudaMemset d_vertex_allocations.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);
	if (total_blocks > MAX_GRIDSIZE)
		calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csr_in.size);

	kernel_vertexAllocationConstructionCSR<<<blocks_per_grid, threads_per_block>>>(d_vertex_allocations, d_vertices_csr_in, d_edges_csr.ptrs);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexAllocationConstruction execution");

	//	d_vertices_csr_out.size = recursiveScan(d_vertex_allocations.addresses, d_vertex_allocations.size);

	thrust::device_ptr<int> ptr(d_vertex_allocations.addresses);
	d_vertices_csr_out.size = thrust::reduce(ptr, ptr + d_vertex_allocations.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_allocations.size, ptr);				// exclusive scan for calculating the scatter addresses.

	if (d_vertices_csr_out.size > 0)
	{
		cudaSafeCall(cudaMalloc(&d_vertices_csr_out.elements, d_vertices_csr_out.size * sizeof(short)), "Error in cudaMalloc d_vertices_csr_out.elements");

		kernel_coverAndExpand<<<blocks_per_grid, threads_per_block>>>(d_next, d_vertices_csr_out, d_vertices_csr_in, d_vertex_allocations, d_edges_csr, d_edges_csc, d_vertices, d_edges, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_coverAndExpand execution");
		cudaSafeCall(cudaMemcpy(&h_next, d_next, sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_next");
	}

	cudaSafeCall(cudaFree(d_next), "Error in cudaFree d_next");
	cudaSafeCall(cudaFree(d_vertex_allocations.predicates), "Error in cudaFree d_vertex_allocations.predicates");
	cudaSafeCall(cudaFree(d_vertex_allocations.addresses), "Error in cudaFree d_vertex_allocations.addresses");
}

// Kernel for initializing the row or column vertices, later used for recursive frontier update (in Step 3).
__global__ void kernel_rowInitialization(short *d_vertex_ids, short *d_visited, int *d_ptrs, int *d_covers, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int cover = (id < N) ? d_covers[id] : 0;
	int size = (id < N) ? (d_ptrs[id + 1] - d_ptrs[id]) : 0;

	if (id < N)
	{
		d_vertex_ids[id] = (short)id;
		d_visited[id] = (size == 0) ? VISITED : ((cover == 0) ? ACTIVE : DORMANT);
	}
}

// Kernel for populating the predicate matrix for edges in row major format.
__global__ void kernel_edgePredicateConstructionCSR(Predicates d_edge_predicates_csr, char *d_masks, int N)
{
	int rowid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int id = rowid * N + colid;

	// Copy the matrix into shared memory
	char mask = (rowid < N && colid < N) ? d_masks[id] : NORMAL;
	bool predicate = (mask == ZERO || mask == PRIME);
	int addr = predicate ? 1 : 0;

	if (rowid < N && colid < N)
	{
		d_edge_predicates_csr.predicates[id] = predicate; // Copy the predicate matrix back to global memory
		d_edge_predicates_csr.addresses[id] = addr;
	}
}

// Kernel for scattering the edges based on the scatter addresses.
__global__ void kernel_edgeScatterCSR(CompactEdges d_edges_csr, Predicates d_edge_predicates_csr, int M, int N)
{
	int rowid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int id = rowid * N + colid;

	// Copy the matrix into shared memory
	bool predicate = (rowid < N && colid < N) ? d_edge_predicates_csr.predicates[id] : false;
	int compid = (rowid < N && colid < N) ? d_edge_predicates_csr.addresses[id] : -1;

	if (rowid < N && colid < N)
	{
		if (predicate)
		{
			d_edges_csr.neighbors[compid] = (short)colid;
		}
		if (colid == 0)
		{
			d_edges_csr.ptrs[rowid] = compid;
			d_edges_csr.ptrs[N] = M; // extra pointer for the total number of edges. necessary for calculating number of edges in each row.
		}
	}
}

// Kernel for calculating predicates for row vertices.
__global__ void kernel_vertexPredicateConstructionCSR(Predicates d_vertex_predicates, Array d_vertices_csr_in, short *d_visited)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	int size = d_vertices_csr_in.size;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	// Copy the matrix into shared memory.
	short vertexid = (id < size) ? d_vertices_csr_in.elements[id] : -1;
	short visited = (id < size && vertexid != -1) ? d_visited[vertexid] : DORMANT;

	bool predicate = (visited == ACTIVE); // If vertex is not visited then it is added to frontier queue.
	int addr = predicate ? 1 : 0;

	if (id < size)
	{
		d_vertex_predicates.predicates[id] = predicate;
		d_vertex_predicates.addresses[id] = addr;
	}
}

// Kernel for scattering the vertices based on the scatter addresses.
__global__ void kernel_vertexScatterCSR(short *d_vertex_ids_csr, short *d_vertex_ids, Predicates d_vertex_predicates)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	int size = d_vertex_predicates.size;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	short vertexid = (id < size) ? d_vertex_ids[id] : -1;
	bool predicate = (id < size) ? d_vertex_predicates.predicates[id] : false;
	int compid = (predicate) ? d_vertex_predicates.addresses[id] : -1; // compaction id.

	if (id < size)
	{
		if (predicate)
			d_vertex_ids_csr[compid] = vertexid;
	}
}

// Kernel for calculating allocations for new frontier (new unvisited rows).
__global__ void kernel_vertexAllocationConstructionCSR(Predicates d_vertex_allocations, Array d_vertices_csr_in, int *d_ptrs)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	int size = d_vertices_csr_in.size;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	short vertexid = (id < size) ? d_vertices_csr_in.elements[id] : -1;
	int st_ptr = (vertexid != -1) ? d_ptrs[vertexid] : -1;
	int end_ptr = (vertexid != -1) ? d_ptrs[vertexid + 1] : -1;
	int allocation_size = end_ptr - st_ptr;
	bool predicate = (allocation_size > 0);

	if (id < size)
	{
		d_vertex_allocations.predicates[id] = predicate;
		d_vertex_allocations.addresses[id] = allocation_size;
	}
}

// Kernel for finding the minimum zero cover.
__global__ void kernel_coverAndExpand(int *d_next, Array d_vertices_csr_out, Array d_vertices_csr_in, Predicates d_vertex_allocations, CompactEdges d_edges_csr, CompactEdges d_edges_csc, Vertices d_vertices, Edges d_edges, int N)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	int in_size = d_vertices_csr_in.size;
	int out_size = d_vertices_csr_out.size;

	// Load values into local memory
	short vertexid = (id < in_size) ? d_vertices_csr_in.elements[id] : -1;
	int alloc_id = (id < in_size) ? d_vertex_allocations.addresses[id] : -1; // allocation id.
	int alloc_id_nxt = (id < in_size - 1) ? d_vertex_allocations.addresses[id + 1] : out_size;
	int allocation_size = (id < in_size) ? (alloc_id_nxt - alloc_id) : 0;

	int st_ptr = (vertexid != -1) ? d_edges_csr.ptrs[vertexid] : -1;
	short *allocation_start = (alloc_id != -1) ? &d_vertices_csr_out.elements[alloc_id] : NULL;
	short *neighbor_start = (st_ptr != -1) ? &d_edges_csr.neighbors[st_ptr] : NULL;
	short *neighbor_end = (neighbor_start != NULL) ? neighbor_start + allocation_size : NULL;

	if (id < in_size)
	{
		__update_covers(d_vertices, d_next, d_edges_csr.is_visited, d_edges_csc.is_visited, allocation_start, neighbor_start, neighbor_end, d_edges.masks, vertexid, N);
		d_edges_csr.is_visited[vertexid] = VISITED;
	}
}

// Device function for traversing the neighbors from start pointer to end pointer and updating the covers.
// The function sets d_next to 4 if there are uncovered zeros, indicating the requirement of Step 4 execution.
__device__ void __update_covers(Vertices d_vertices, int *d_next, short *d_row_visited, short *d_col_visited, short *new_frontier, short *d_start_ptr, short *d_end_ptr, char *d_masks, short vertexid, int N)
{
	short *ptr1 = d_start_ptr;
	short *ptr2 = new_frontier;

	while (ptr1 != d_end_ptr)
	{
		short colid = *ptr1;
		short rowid = d_vertices.col_assignments[colid];

		if (d_vertices.col_covers[colid] == 0) // if the column is already covered, it should not be included in next frontier expansion.
		{
			d_masks[vertexid * N + colid] = PRIME;
			if (rowid != -1)
			{
				d_vertices.row_covers[rowid] = 0;
				d_vertices.col_covers[colid] = 1;
				if (d_row_visited[rowid] == DORMANT)
					d_row_visited[rowid] = ACTIVE;
			}
			else // activate the column for maximum matching step (Step 4).
			{
				d_col_visited[colid] = ACTIVE;
				*d_next = 4;
			}

			*ptr2 = rowid;
		}
		else
		{
			*ptr2 = -1;
		}
		ptr1++;
		ptr2++;
	}
}