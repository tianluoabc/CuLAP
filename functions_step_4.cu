/*
 * Created by Ketan Date
 */

#include "include/functions_step_4.h"

// Function for compacting the edges in column major format.
void compactEdgesCSC(Predicates &d_edge_predicates_csc)
{
	int K = 16;
	dim3 blocks_per_grid_1;
	dim3 threads_per_block_1;

	int valuex = (int)ceil((double)(N) / K);
	int valuey = (int)ceil((double)(N) / K);

	threads_per_block_1.x = K;
	threads_per_block_1.y = K;
	blocks_per_grid_1.x = valuex;
	blocks_per_grid_1.y = valuey;

	kernel_edgePredicateConstructionCSC<<<blocks_per_grid_1, threads_per_block_1>>>(d_edge_predicates_csc, d_edges.masks, N); // construct predicate matrix for edges.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgePredicateConstructionCSC execution");

	//	M = recursiveScan(d_edge_predicates_csc.addresses, d_edge_predicates_csc.size); // construct scatter addresses for edges using exclusive scan.

	thrust::device_ptr<int> ptr(d_edge_predicates_csc.addresses);
	M = thrust::reduce(ptr, ptr + N2);			// calculate total number of edges.
	thrust::exclusive_scan(ptr, ptr + N2, ptr); // exclusive scan for calculating the scatter addresses.

	cudaSafeCall(cudaMalloc(&d_edges_csc.neighbors, M * sizeof(int)), "Error in cudaMalloc d_edges_csc.neighbors");

	kernel_edgeScatterCSC<<<blocks_per_grid_1, threads_per_block_1>>>(d_edges_csc, d_edge_predicates_csc, M, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_edgeScatterCSC execution");
}

// Function for executing forward pass of the maximum matching.
void forwardPass(VertexData &d_row_data, VertexData &d_col_data)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	Array d_vertices_csc1, d_vertices_csc2;

	d_vertices_csc1.size = N;
	cudaSafeCall(cudaMalloc(&d_vertices_csc1.elements, N * sizeof(short)), "Error in cudaMalloc d_vertices_csc1.elements"); // compact vertices initialized to the column ids.

	kernel_colInitialization<<<blocks_per_grid, threads_per_block>>>(d_vertices_csc1.elements, d_edges_csc.is_visited, d_edges_csc.ptrs, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexInitialization execution");

	while (true)
	{
		compactColumnVertices(d_vertices_csc2, d_vertices_csc1); // compact the current vertex frontier.
		cudaSafeCall(cudaFree(d_vertices_csc1.elements), "Error in cudaFree d_vertices_csc1.elements");
		if (d_vertices_csc2.size == 0)
			break;

		traverseAndExpand(d_vertices_csc1, d_vertices_csc2, d_row_data, d_col_data); // traverse the frontier, find augmenting path and expand.
		cudaSafeCall(cudaFree(d_vertices_csc2.elements), "Error in cudaFree d_vertices_csc2.elements");
		if (d_vertices_csc1.size == 0)
			break;
	}
}

// Function for executing reverse pass of the maximum matching.
void reversePass(VertexData &d_row_data, VertexData &d_col_data)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	Array d_row_ids_csr;
	Predicates d_row_predicates; // predicates for compacting the rowids eligible for the reverse pass.

	d_row_predicates.size = N;
	cudaSafeCall(cudaMalloc(&d_row_predicates.predicates, N * sizeof(bool)), "Error in cudaMalloc d_row_predicates.predicates");
	cudaSafeCall(cudaMalloc(&d_row_predicates.addresses, N * sizeof(int)), "Error in cudaMalloc d_row_predicates.addresses");
	cudaSafeCall(cudaMemset(d_row_predicates.predicates, false, N * sizeof(bool)), "Error in cudaMemset d_row_predicates.predicates");
	cudaSafeCall(cudaMemset(d_row_predicates.addresses, 0, N * sizeof(int)), "Error in cudaMemset d_row_predicates.addresses");

	// compact the reverse pass row vertices.
	kernel_augmentPredicateConstruction<<<blocks_per_grid, threads_per_block>>>(d_row_predicates, d_edges_csr.is_visited, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentPredicateConstruction execution");

	//	d_row_ids_csr.size = recursiveScan(d_row_predicates.addresses, N);

	thrust::device_ptr<int> ptr(d_row_predicates.addresses);
	d_row_ids_csr.size = thrust::reduce(ptr, ptr + d_row_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_row_predicates.size, ptr);		   // exclusive scan for calculating the scatter addresses.

	if (d_row_ids_csr.size > 0)
	{
		int total_blocks_1 = 0;
		dim3 blocks_per_grid_1;
		dim3 threads_per_block_1;
		calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1, d_row_ids_csr.size);

		cudaSafeCall(cudaMalloc(&d_row_ids_csr.elements, d_row_ids_csr.size * sizeof(short)), "Error in cudaMalloc d_row_ids_csr.elements");

		kernel_augmentScatter<<<blocks_per_grid, threads_per_block>>>(d_row_ids_csr, d_row_predicates, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentScatter execution");

		kernel_reverseTraversal<<<blocks_per_grid_1, threads_per_block_1>>>(d_row_ids_csr, d_edges_csc, d_row_data, d_col_data);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_reverseTraversal execution");

		cudaSafeCall(cudaFree(d_row_ids_csr.elements), "Error in cudaFree d_row_ids_csr.elements");
	}

	cudaSafeCall(cudaFree(d_row_predicates.predicates), "Error in cudaFree d_row_predicates.predicates");
	cudaSafeCall(cudaFree(d_row_predicates.addresses), "Error in cudaFree d_row_predicates.addresses");
}

// Function for executing augmentation pass of the maximum matching.
void augmentationPass(VertexData &d_row_data, VertexData &d_col_data)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	Array d_col_ids_csr;
	Predicates d_col_predicates; // predicates for compacting the colids eligible for the augmentation pass.

	d_col_predicates.size = N;
	cudaSafeCall(cudaMalloc(&d_col_predicates.predicates, N * sizeof(bool)), "Error in cudaMalloc d_col_predicates.predicates");
	cudaSafeCall(cudaMalloc(&d_col_predicates.addresses, N * sizeof(int)), "Error in cudaMalloc d_col_predicates.addresses");
	cudaSafeCall(cudaMemset(d_col_predicates.predicates, false, N * sizeof(bool)), "Error in cudaMemset d_col_predicates.predicates");
	cudaSafeCall(cudaMemset(d_col_predicates.addresses, 0, N * sizeof(int)), "Error in cudaMemset d_col_predicates.addresses");

	// compact the reverse pass row vertices.
	kernel_augmentPredicateConstruction<<<blocks_per_grid, threads_per_block>>>(d_col_predicates, d_edges_csc.is_visited, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentPredicateConstruction execution");

	//	d_col_ids_csr.size = recursiveScan(d_col_predicates.addresses, N);

	thrust::device_ptr<int> ptr(d_col_predicates.addresses);
	d_col_ids_csr.size = thrust::reduce(ptr, ptr + d_col_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_col_predicates.size, ptr);		   // exclusive scan for calculating the scatter addresses.

	if (d_col_ids_csr.size > 0)
	{
		int total_blocks_1 = 0;
		dim3 blocks_per_grid_1;
		dim3 threads_per_block_1;
		calculateLinearDims(blocks_per_grid_1, threads_per_block_1, total_blocks_1, d_col_ids_csr.size);

		cudaSafeCall(cudaMalloc(&d_col_ids_csr.elements, d_col_ids_csr.size * sizeof(short)), "Error in cudaMalloc d_col_ids_csr.elements");

		kernel_augmentScatter<<<blocks_per_grid, threads_per_block>>>(d_col_ids_csr, d_col_predicates, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentScatter execution");

		kernel_augmentation<<<blocks_per_grid_1, threads_per_block_1>>>(d_edges.masks, d_col_ids_csr, d_row_data, d_col_data, N);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_augmentation execution");

		cudaSafeCall(cudaFree(d_col_ids_csr.elements), "Error in cudaFree d_col_ids_csr.elements");
	}

	cudaSafeCall(cudaFree(d_col_predicates.predicates), "Error in cudaFree d_col_predicates.predicates");
	cudaSafeCall(cudaFree(d_col_predicates.addresses), "Error in cudaFree d_col_predicates.addresses");
}

//Function for compacting column vertices. Used in Step 4 (maximum matching).
void compactColumnVertices(Array &d_vertices_csc_out, Array &d_vertices_csc_in)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	Predicates d_vertex_predicates;

	d_vertex_predicates.size = d_vertices_csc_in.size;
	cudaSafeCall(cudaMalloc(&d_vertex_predicates.predicates, d_vertex_predicates.size * sizeof(bool)), "Error in cudaMalloc d_vertex_predicates.predicates");
	cudaSafeCall(cudaMalloc(&d_vertex_predicates.addresses, d_vertex_predicates.size * sizeof(int)), "Error in cudaMalloc d_vertex_predicates.addresses");
	cudaSafeCall(cudaMemset(d_vertex_predicates.predicates, false, d_vertex_predicates.size * sizeof(bool)), "Error in cudaMemset d_vertex_predicates.predicates");
	cudaSafeCall(cudaMemset(d_vertex_predicates.addresses, 0, d_vertex_predicates.size * sizeof(int)), "Error in cudaMemset d_vertex_predicates.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csc_in.size);
	if (total_blocks > MAX_GRIDSIZE)
		calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csc_in.size);

	kernel_vertexPredicateConstructionCSC<<<blocks_per_grid, threads_per_block>>>(d_vertex_predicates, d_vertices_csc_in, d_edges_csc.is_visited);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexPredicateConstruction execution");

	//	d_vertices_csc_out.size = recursiveScan (d_vertex_predicates.addresses, d_vertex_predicates.size);

	thrust::device_ptr<int> ptr(d_vertex_predicates.addresses);
	d_vertices_csc_out.size = thrust::reduce(ptr, ptr + d_vertex_predicates.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_predicates.size, ptr);			   // exclusive scan for calculating the scatter addresses.

	if (d_vertices_csc_out.size > 0)
	{
		cudaSafeCall(cudaMalloc(&d_vertices_csc_out.elements, d_vertices_csc_out.size * sizeof(short)), "Error in cudaMalloc d_vertices_csc_out.elements");

		kernel_vertexScatterCSC<<<blocks_per_grid, threads_per_block>>>(d_vertices_csc_out.elements, d_vertices_csc_in.elements, d_vertex_predicates);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexScatter execution");
	}

	cudaSafeCall(cudaFree(d_vertex_predicates.predicates), "Error in cudaFree d_vertex_predicates.predicates");
	cudaSafeCall(cudaFree(d_vertex_predicates.addresses), "Error in cudaFree d_vertex_predicates.addresses");
}

void traverseAndExpand(Array &d_vertices_csc_out, Array &d_vertices_csc_in, VertexData &d_row_data, VertexData &d_col_data)
{
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	Predicates d_vertex_allocations;
	d_vertex_allocations.size = d_vertices_csc_in.size;
	cudaSafeCall(cudaMalloc(&d_vertex_allocations.predicates, d_vertex_allocations.size * sizeof(bool)), "Error in cudaMalloc d_vertex_allocations.predicates");
	cudaSafeCall(cudaMalloc(&d_vertex_allocations.addresses, d_vertex_allocations.size * sizeof(int)), "Error in cudaMalloc d_vertex_allocations.addresses");

	cudaSafeCall(cudaMemset(d_vertex_allocations.predicates, false, d_vertex_allocations.size * sizeof(bool)), "Error in cudaMemset d_vertex_allocations.predicates");
	cudaSafeCall(cudaMemset(d_vertex_allocations.addresses, 0, d_vertex_allocations.size * sizeof(int)), "Error in cudaMemset d_vertex_allocations.addresses");

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csc_in.size);
	if (total_blocks > MAX_GRIDSIZE)
		calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, d_vertices_csc_in.size);

	kernel_vertexAllocationConstructionCSC<<<blocks_per_grid, threads_per_block>>>(d_vertex_allocations, d_vertices_csc_in, d_edges_csc.ptrs);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_vertexAllocationConstructionCSC execution");

	//	d_vertices_csc_out.size = recursiveScan(d_vertex_allocations.addresses, d_vertex_allocations.size);

	thrust::device_ptr<int> ptr(d_vertex_allocations.addresses);
	d_vertices_csc_out.size = thrust::reduce(ptr, ptr + d_vertex_allocations.size); // calculate total number of vertices.
	thrust::exclusive_scan(ptr, ptr + d_vertex_allocations.size, ptr);				// exclusive scan for calculating the scatter addresses.

	if (d_vertices_csc_out.size > 0)
	{
		cudaSafeCall(cudaMalloc(&d_vertices_csc_out.elements, d_vertices_csc_out.size * sizeof(short)), "Error in cudaMalloc d_vertices_csc_out.elements");

		kernel_forwardTraversal<<<blocks_per_grid, threads_per_block>>>(d_vertices_csc_out, d_vertices_csc_in, d_vertex_allocations, d_row_data, d_col_data, d_edges_csr, d_edges_csc, d_vertices);
		cudaSafeCall(cudaGetLastError(), "Error in kernel_coverAndExpand execution");
	}

	cudaSafeCall(cudaFree(d_vertex_allocations.predicates), "Error in cudaFree d_vertex_allocations.predicates");
	cudaSafeCall(cudaFree(d_vertex_allocations.addresses), "Error in cudaFree d_vertex_allocations.addresses");
}

// Kernel for initializing the column vertices, later used for recursive frontier update (in Step 4).
__global__ void kernel_colInitialization(short *d_vertex_ids, short *d_visited, int *d_ptrs, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int size = (id < N) ? (d_ptrs[id + 1] - d_ptrs[id]) : 0;
	int status = (id < N) ? d_visited[id] : DORMANT;

	if (id < N)
	{
		d_vertex_ids[id] = (short)id;
		d_visited[id] = (size == 0) ? VISITED : status;
	}
}

// Kernel for populating the predicate matrix for edges in column major format. Principles of tiled matrix transpose are used.
__global__ void kernel_edgePredicateConstructionCSC(Predicates d_edge_predicates_csc, char *d_masks, int N)
{
	int colid_b = blockIdx.x * blockDim.x;
	int rowid_b = blockIdx.y * blockDim.y;
	int colid_t = threadIdx.x;
	int rowid_t = threadIdx.y;
	int colid = colid_b + colid_t;
	int rowid = rowid_b + rowid_t;
	int id = rowid * N + colid;

	int tr_colid_b = rowid_b;
	int tr_rowid_b = colid_b;
	int tr_colid = tr_colid_b + colid_t;
	int tr_rowid = tr_rowid_b + rowid_t;
	int tr_id = tr_rowid * N + tr_colid;

	__shared__ bool s_predicates[16][16];
	__shared__ int s_addr[16][16];

	// Construct the transposed predicate matrix into shared memory.
	s_predicates[colid_t][rowid_t] = ((rowid < N) && (colid < N) && (d_masks[id] == PRIME));
	s_addr[colid_t][rowid_t] = s_predicates[colid_t][rowid_t] ? 1 : 0;
	__syncthreads();

	// Copy the matrices back to global memory
	if (tr_rowid < N && tr_colid < N)
	{
		d_edge_predicates_csc.predicates[tr_id] = s_predicates[rowid_t][colid_t];
		d_edge_predicates_csc.addresses[tr_id] = s_addr[rowid_t][colid_t];
	}
}

// Kernel for scattering the edges based on the scatter addresses. The predicate matrix is the transpose of CSR predicates.
__global__ void kernel_edgeScatterCSC(CompactEdges d_edges_csc, Predicates d_edge_predicates_csc, int M, int N)
{
	int colid = blockIdx.y * blockDim.y + threadIdx.y; // column major.
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	int id = colid * N + rowid; // ID is in row major format.

	// Copy the matrix into shared memory
	bool predicate = (rowid < N && colid < N) ? d_edge_predicates_csc.predicates[id] : false;
	int compid = (rowid < N && colid < N) ? d_edge_predicates_csc.addresses[id] : -1;

	if (rowid < N && colid < N)
	{
		if (predicate)
		{
			d_edges_csc.neighbors[compid] = rowid;
		}
		if (rowid == 0)
		{
			d_edges_csc.ptrs[colid] = compid;
			d_edges_csc.ptrs[N] = M; // extra pointer for the total number of edges. necessary for calculating number of edges in each column.
		}
	}
}

// Kernel for calculating predicates for column vertices.
__global__ void kernel_vertexPredicateConstructionCSC(Predicates d_vertex_predicates, Array d_vertices_csc_in, short *d_visited)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	int size = d_vertices_csc_in.size;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	// Copy the matrix into shared memory.
	short vertexid = (id < size) ? d_vertices_csc_in.elements[id] : -1;
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
__global__ void kernel_vertexScatterCSC(short *d_vertex_ids_csc, short *d_vertex_ids, Predicates d_vertex_predicates)
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
			d_vertex_ids_csc[compid] = vertexid;
	}
}

// Kernel for calculating allocations for new frontier (new unvisited columns).
__global__ void kernel_vertexAllocationConstructionCSC(Predicates d_vertex_allocations, Array d_vertices_csc_in, int *d_ptrs)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	int size = d_vertices_csc_in.size;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	short vertexid = (id < size) ? d_vertices_csc_in.elements[id] : -1;
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

// Kernel for executing the forward pass of the maximum matching algorithm.
__global__ void kernel_forwardTraversal(Array d_vertices_csc_out, Array d_vertices_csc_in, Predicates d_vertex_allocations, VertexData d_row_data, VertexData d_col_data, CompactEdges d_edges_csr, CompactEdges d_edges_csc, Vertices d_vertices)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int id = blockid * TCOUNT + tid;

	int in_size = d_vertices_csc_in.size;
	int out_size = d_vertices_csc_out.size;

	// Load values into local memory
	short vertexid = (id < in_size) ? d_vertices_csc_in.elements[id] : -1;
	int alloc_id = (id < in_size) ? d_vertex_allocations.addresses[id] : -1; // allocation id.
	int alloc_id_nxt = (id < in_size - 1) ? d_vertex_allocations.addresses[id + 1] : out_size;
	int allocation_size = (id < in_size) ? (alloc_id_nxt - alloc_id) : 0;

	int st_ptr = (vertexid != -1) ? d_edges_csc.ptrs[vertexid] : -1;
	short *allocation_start = (alloc_id != -1) ? &d_vertices_csc_out.elements[alloc_id] : NULL;
	short *neighbor_start = (st_ptr != -1) ? &d_edges_csc.neighbors[st_ptr] : NULL;
	short *neighbor_end = (neighbor_start != NULL) ? neighbor_start + allocation_size : NULL;

	if (id < in_size)
	{
		__forward_traversal(d_row_data.parents, d_col_data.parents, d_edges_csr.is_visited, d_edges_csc.is_visited, allocation_start, neighbor_start, neighbor_end, d_vertices, vertexid);
		d_edges_csc.is_visited[vertexid] = VISITED;
	}
}

// Kernel for constructing the predicates for reverse pass or augmentation candidates.
__global__ void kernel_augmentPredicateConstruction(Predicates d_predicates, short *d_visited, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the matrix into shared memory.
	short visited = (id < size) ? d_visited[id] : DORMANT;
	bool predicate = (visited == REVERSE || visited == AUGMENT);
	int addr = predicate ? 1 : 0;

	if (id < size)
	{
		d_predicates.predicates[id] = predicate;
		d_predicates.addresses[id] = addr;
	}
}

// Kernel for scattering the vertices based on the scatter addresses.
__global__ void kernel_augmentScatter(Array d_vertex_ids, Predicates d_predicates, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	bool predicate = (id < size) ? d_predicates.predicates[id] : false;
	int compid = (predicate) ? d_predicates.addresses[id] : -1; // compaction id.

	if (id < size)
	{
		if (predicate)
			d_vertex_ids.elements[compid] = (short)id;
	}
}

// Kernel for executing the reverse pass of the maximum matching algorithm.
__global__ void kernel_reverseTraversal(Array d_row_vertices, CompactEdges d_edges_csc, VertexData d_row_data, VertexData d_col_data)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_row_vertices.size;

	int rowid = (id < size) ? d_row_vertices.elements[id] : -1;

	if (id < size)
	{
		__reverse_traversal(d_edges_csc.is_visited, d_row_data.children, d_col_data.children, d_row_data.parents, d_col_data.parents, rowid);
	}
}

// Kernel for executing the augmentation pass of the maximum matching algorithm.
__global__ void kernel_augmentation(char *d_masks, Array d_col_vertices, VertexData d_row_data, VertexData d_col_data, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int size = d_col_vertices.size;

	int colid = (id < size) ? d_col_vertices.elements[id] : -1;

	if (id < size)
	{
		__augment(d_masks, d_row_data.children, d_col_data.children, colid, N);
	}
}

// Device function for traversing the neighbors from start pointer to end pointer and finding an augmenting path.
// Unassigned rows are marked for the reverse pass.
__device__ void __forward_traversal(short *d_row_parents, short *d_col_parents, short *d_row_visited, short *d_col_visited, short *new_frontier, short *d_start_ptr, short *d_end_ptr, Vertices d_vertices, short parent_col_id)
{
	short *ptr1 = d_start_ptr;
	short *ptr2 = new_frontier;

	while (ptr1 != d_end_ptr)
	{
		short rowid = *ptr1;
		short colid = d_vertices.row_assignments[rowid];

		if (d_vertices.row_covers[rowid] == 0 && d_row_visited[rowid] == DORMANT) // if the row is already covered or visited, it should not be included in next frontier.
		{
			d_row_visited[rowid] = VISITED;
			d_row_parents[rowid] = parent_col_id;
			if (colid != -1)
			{
				d_col_parents[colid] = rowid;
				if (d_col_visited[colid] == DORMANT)
					d_col_visited[colid] = ACTIVE;
			}
			else // activate the row for reverse pass.
			{
				d_row_visited[rowid] = REVERSE;
			}

			*ptr2 = colid;
		}
		else
		{
			*ptr2 = -1;
		}
		ptr1++;
		ptr2++;
	}
}

// Device function for traversing an alternating path from unassigned row to unassigned column.
__device__ void __reverse_traversal(short *d_col_visited, short *d_row_children, short *d_col_children, short *d_row_parents, short *d_col_parents, int init_rowid)
{
	short cur_rowid = (short)init_rowid;
	short cur_colid = -1;

	while (cur_rowid != -1)
	{
		d_row_children[cur_rowid] = cur_colid;

		cur_colid = d_row_parents[cur_rowid];

		d_col_children[cur_colid] = cur_rowid;
		cur_rowid = d_col_parents[cur_colid];
	}
	d_col_visited[cur_colid] = AUGMENT;
}

// Device function for augmenting the alternating path from unassigned column to unassigned row.
__device__ void __augment(char *d_masks, short *d_row_children, short *d_col_children, int init_colid, int N)
{
	short cur_rowid = -1;
	short cur_colid = (short)init_colid;

	while (true)
	{
		cur_rowid = d_col_children[cur_colid];
		d_masks[cur_rowid * N + cur_colid] = STAR;

		cur_colid = d_row_children[cur_rowid];
		if (cur_colid != -1)
		{
			d_masks[cur_rowid * N + cur_colid] = NORMAL;
		}
		else
		{
			break;
		}
	}
}