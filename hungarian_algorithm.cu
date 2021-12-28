/*
 * Created by Ketan Date
 */

#include "include/hungarian_algorithm.h"

cudaEvent_t start, stop;
int initial_assignment_count;
int h_obj_val;
int *counts;
float *times;

// Executes Hungarian algorithm on the input cost matrix. Returns minimum cost.
int solve(int *stepcounts, float *steptimes, int &init_assignments)
{
	int step = 0;
	int total_count = 0;
	bool done = false;
	h_obj_val = 0;
	initial_assignment_count = 0;

	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);

	counts = stepcounts;
	times = steptimes;

	while (!done)
	{
		total_count++;
		switch (step)
		{
		case 0:
			counts[0]++;
			step = hungarianStep0(true);
			break;
		case 1:
			counts[1]++;
			step = hungarianStep1(true);
			break;
		case 2:
			counts[2]++;
			step = hungarianStep2(true);
			break;
		case 3:
			counts[3]++;
			step = hungarianStep3(true);
			break;
		case 4:
			counts[4]++;
			step = hungarianStep4(true);
			break;
		case 5:
			counts[5]++;
			step = hungarianStep5(true);
			break;
		case 6:
			counts[6]++;
			step = hungarianStep6(true);
			break;
		case 100:
			done = true;
			break;
		}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	init_assignments = initial_assignment_count;

	return h_obj_val;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int hungarianStep0(bool count_time)
{
	float time;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	cudaEventRecord(start, 0);

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	kernel_rowReduction<<<blocks_per_grid, threads_per_block>>>(d_edges.costs, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_rowReduction execution");
	kernel_columnReduction<<<blocks_per_grid, threads_per_block>>>(d_edges.costs, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_colReduction execution");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (count_time)
		times[0] += time;

	return 1;
}

// Function for calculating initial zeros by subtracting row and column minima from each element.
int hungarianStep1(bool count_time)
{
	cudaEvent_t start1, stop1;
	float time;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	cudaEventCreate(&start1, 0);
	cudaEventCreate(&stop1, 0);

	cudaEventRecord(start1, 0);

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	int *d_row_lock, *d_col_lock;
	cudaSafeCall(cudaMalloc(&d_row_lock, N * sizeof(int)), "Error in cudaMalloc d_row_lock");
	cudaSafeCall(cudaMalloc(&d_col_lock, N * sizeof(int)), "Error in cudaMalloc d_col_lock");
	cudaSafeCall(cudaMemset(d_row_lock, 0, N * sizeof(int)), "Error in cudaMemset d_row_lock");
	cudaSafeCall(cudaMemset(d_col_lock, 0, N * sizeof(int)), "Error in cudaMemset d_col_lock");

	kernel_computeInitialAssignments<<<blocks_per_grid, threads_per_block>>>(d_edges.masks, d_edges.costs, d_row_lock, d_col_lock, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeInitialAssignments execution");

	int next = 2;
	while (true)
	{

		initial_assignment_count = 0;

		if ((next = hungarianStep2(false)) == 6)
			break;

		if ((next = hungarianStep3(false)) == 5)
			break;

		hungarianStep4(false);
	}

	cudaSafeCall(cudaFree(d_row_lock), "Error in cudaFree d_row_lock");
	cudaSafeCall(cudaFree(d_col_lock), "Error in cudaFree d_col_lock");

	cudaEventRecord(stop1, 0);
	cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&time, start1, stop1);

	cudaEventDestroy(start1);
	cudaEventDestroy(stop1);

	if (count_time)
		times[1] += time;

	return next;
}

// Function for checking optimality and constructing predicates and covers.
int hungarianStep2(bool count_time)
{
	int next = 3;
	float time;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	cudaEventRecord(start, 0);

	calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, N2);

	cudaSafeCall(cudaMemset(d_vertices.row_assignments, -1, N * sizeof(short)), "Error in cudaMemset d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices.col_assignments, -1, N * sizeof(short)), "Error in cudaMemset d_col_assignment");
	cudaSafeCall(cudaMemset(d_vertices.row_covers, 0, N * sizeof(int)), "Error in cudaMemset d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices.col_covers, 0, N * sizeof(int)), "Error in cudaMemset d_col_covers");

	kernel_populateAssignments<<<blocks_per_grid, threads_per_block>>>(d_vertices.row_assignments, d_vertices.col_assignments, d_vertices.row_covers, d_edges.masks, d_edges.costs, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_populateAssignments execution");

	int cover_count = recursiveSum(d_vertices.row_covers, N); //count number of covered rows using parallel sum.

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (initial_assignment_count == 0)
		initial_assignment_count = cover_count;

	if (cover_count == N)
		next = 6;

	if (count_time)
		times[2] += time;

	return next;
}

int hungarianStep3(bool count_time)
{
	int next;
	float time;
	Predicates d_edge_predicates_csr;

	cudaEventRecord(start, 0);

#ifdef LIGHT // Light memory version. Copy reduced costs to host and delete.
	cudaSafeCall(cudaMemcpy(h_red_costs.elements, d_edges.costs, N2 * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_edges.costs");
	cudaSafeCall(cudaFree(d_edges.costs), "Error in cudaFree d_edges.costs");
#endif

	d_edge_predicates_csr.size = N2;
	cudaSafeCall(cudaMalloc(&d_edge_predicates_csr.predicates, N2 * sizeof(bool)), "Error in cudaMalloc d_edge_predicates_csr.predicates");
	cudaSafeCall(cudaMalloc(&d_edge_predicates_csr.addresses, N2 * sizeof(int)), "Error in cudaMalloc d_edge_predicates_csr.addresses");
	cudaSafeCall(cudaMemset(d_edge_predicates_csr.predicates, false, N2 * sizeof(bool)), "Error in cudaMemset d_edge_predicates_csr.predicates");
	cudaSafeCall(cudaMemset(d_edge_predicates_csr.addresses, 0, N2 * sizeof(int)), "Error in cudaMemset d_edge_predicates_csr.addresses");

	cudaSafeCall(cudaMalloc(&d_edges_csr.ptrs, (N + 1) * sizeof(int)), "Error in cudaMalloc d_edges_csr.ptrs");
	cudaSafeCall(cudaMemset(d_edges_csr.ptrs, -1, (N + 1) * sizeof(int)), "Error in cudaMemset d_edges_csr.ptrs");

	cudaSafeCall(cudaMemset(d_edges_csc.is_visited, DORMANT, N * sizeof(short)), "Error in cudaMemset d_edges_csc.is_visited"); // initialize "visited" array for columns. later used in BFS (Step 4).

	compactEdgesCSR(d_edge_predicates_csr); // execute edge compaction in row major format.

	executeZeroCover(next); // execute zero cover algorithm.

	cudaSafeCall(cudaFree(d_edges_csr.neighbors), "Error in cudaFree d_edges_csr.neighbors");
	cudaSafeCall(cudaFree(d_edges_csr.ptrs), "Error in cudaFree d_edges_csr.ptrs");
	cudaSafeCall(cudaFree(d_edge_predicates_csr.predicates), "Error in cudaFree d_edge_predicates_csr.predicates");
	cudaSafeCall(cudaFree(d_edge_predicates_csr.addresses), "Error in cudaFree d_edge_predicates_csr.addresses");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (count_time)
		times[3] += time;

	return next;
}

int hungarianStep4(bool count_time)
{
	float time;
	Predicates d_edge_predicates_csc;
	VertexData d_row_data, d_col_data;

	cudaEventRecord(start, 0);

	d_edge_predicates_csc.size = N2;
	cudaSafeCall(cudaMalloc(&d_edge_predicates_csc.predicates, N2 * sizeof(bool)), "Error in cudaMalloc d_edge_predicates_csc.predicates");
	cudaSafeCall(cudaMalloc(&d_edge_predicates_csc.addresses, N2 * sizeof(int)), "Error in cudaMalloc d_edge_predicates_csc.addresses");
	cudaSafeCall(cudaMemset(d_edge_predicates_csc.predicates, false, N2 * sizeof(bool)), "Error in cudaMemset d_edge_predicates_csc.predicates");
	cudaSafeCall(cudaMemset(d_edge_predicates_csc.addresses, 0, N2 * sizeof(int)), "Error in cudaMemset d_edge_predicates_csc.addresses");

	cudaSafeCall(cudaMalloc(&d_row_data.parents, N * sizeof(short)), "Error in cudaMalloc d_row_data.parents");
	cudaSafeCall(cudaMalloc(&d_row_data.children, N * sizeof(short)), "Error in cudaMalloc d_row_data.children");
	cudaSafeCall(cudaMalloc(&d_col_data.parents, N * sizeof(short)), "Error in cudaMalloc d_col_data.parents");
	cudaSafeCall(cudaMalloc(&d_col_data.children, N * sizeof(short)), "Error in cudaMalloc d_col_data.children");
	cudaSafeCall(cudaMemset(d_row_data.parents, -1, N * sizeof(short)), "Error in cudaMemset d_row_data.parents");
	cudaSafeCall(cudaMemset(d_row_data.children, -1, N * sizeof(short)), "Error in cudaMemset d_row_data.children");
	cudaSafeCall(cudaMemset(d_col_data.parents, -1, N * sizeof(short)), "Error in cudaMemset d_col_data.parents");
	cudaSafeCall(cudaMemset(d_col_data.children, -1, N * sizeof(short)), "Error in cudaMemset d_col_data.children");

	cudaSafeCall(cudaMalloc(&d_edges_csc.ptrs, (N + 1) * sizeof(int)), "Error in cudaMalloc d_edges_csc.ptrs");
	cudaSafeCall(cudaMemset(d_edges_csc.ptrs, -1, (N + 1) * sizeof(int)), "Error in cudaMemset d_edges_csc.ptrs");

	cudaSafeCall(cudaMemset(d_edges_csr.is_visited, DORMANT, N * sizeof(short)), "Error in cudaMemset d_edges_csr.is_visited");

	compactEdgesCSC(d_edge_predicates_csc); // execute edge compaction in column major format.

	forwardPass(d_row_data, d_col_data); // execute forward pass of the maximum matching algorithm.

	reversePass(d_row_data, d_col_data); // execute reverse pass of the maximum matching algorithm.

	augmentationPass(d_row_data, d_col_data); // execute augmentation pass of the maximum matching algorithm.

	cudaSafeCall(cudaFree(d_edges_csc.neighbors), "Error in cudaFree d_edges_csc.neighbors");
	cudaSafeCall(cudaFree(d_edges_csc.ptrs), "Error in cudaFree d_edges_csc.ptrs");
	cudaSafeCall(cudaFree(d_edge_predicates_csc.predicates), "Error in cudaFree d_edge_predicates_csc.predicates");
	cudaSafeCall(cudaFree(d_edge_predicates_csc.addresses), "Error in cudaFree d_edge_predicates_csc.addresses");

	cudaSafeCall(cudaFree(d_row_data.parents), "Error in cudaFree d_row_data.parents");
	cudaSafeCall(cudaFree(d_row_data.children), "Error in cudaFree d_row_data.children");
	cudaSafeCall(cudaFree(d_col_data.parents), "Error in cudaFree d_col_data.parents");
	cudaSafeCall(cudaFree(d_col_data.children), "Error in cudaFree d_col_data.children");

#ifdef LIGHT //Light memory version. Copy reduced cost to device.
	cudaSafeCall(cudaMalloc(&d_edges.costs, N2 * sizeof(int)), "error in cudaMalloc d_edges.costs");
	cudaSafeCall(cudaMemcpy(d_edges.costs, h_red_costs.elements, N2 * sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_red_costs.elements");
#endif

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (count_time)
		times[4] += time;

	return 2;
}

int hungarianStep5(bool count_time)
{
	float time;

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks;

	dim3 blocks_per_grid_1;
	dim3 threads_per_block_1;
	int K = 16;

	int valuex = (int)ceil((double)(N) / K);
	int valuey = (int)ceil((double)(N) / K);

	threads_per_block_1.x = K;
	threads_per_block_1.y = K;
	blocks_per_grid_1.x = valuex;
	blocks_per_grid_1.y = valuey;

	cudaEventRecord(start, 0);

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

#ifdef LIGHT // Light memory version. Copy reduced cost to device.
	cudaSafeCall(cudaMalloc(&d_edges.costs, N2 * sizeof(int)), "error in cudaMalloc d_edges.costs");
	cudaSafeCall(cudaMemcpy(d_edges.costs, h_red_costs.elements, N2 * sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_red_costs.elements");
#endif

	int *d_min;
	cudaSafeCall(cudaMalloc(&d_min, N * sizeof(int)), "Error in cudaMalloc d_min_val");
	cudaSafeCall(cudaMemset(d_min, INF, N * sizeof(int)), "Error in cudaMemset d_min_val");

	cudaSafeCall(cudaMemset(d_edges_csr.is_visited, DORMANT, N * sizeof(short)), "Error in cudaMemset d_edges_csr.is_visited"); // initialize "visited" array for columns. later used in Row Cover (Step 3).

	kernel_dualUpdate_1_nonAtomic<<<blocks_per_grid, threads_per_block>>>(d_min, d_edges.costs, d_vertices.row_covers, d_vertices.col_covers, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_dualUpdate_1 execution");
	int min = recursiveMin(d_min, N);

	kernel_dualUpdate_2<<<blocks_per_grid_1, threads_per_block_1>>>(min, d_edges.masks, d_edges.costs, d_vertices.row_covers, d_vertices.col_covers, d_edges_csr.is_visited, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_dualUpdate_2 execution");

	cudaSafeCall(cudaFree(d_min), "Error in cudaFree d_min_val");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (count_time)
		times[5] += time;

	return 3;
}

int hungarianStep6(bool count_time)
{
	float time;
	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	cudaEventRecord(start, 0);

	calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, N2);

	int *d_obj_val;
	cudaSafeCall(cudaMalloc(&d_obj_val, sizeof(int)), "Error in cudaMalloc d_obj_val");
	cudaSafeCall(cudaMemset(d_obj_val, 0, sizeof(int)), "Error in cudaMemset d_obj_val");

	cudaSafeCall(cudaMemcpy(d_edges.costs, h_costs.elements, N2 * sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_costs"); // reset the costs for final calculation.

	kernel_finalCost<<<blocks_per_grid, threads_per_block>>>(d_obj_val, d_edges.costs, d_edges.masks, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_finalCost execution");

	cudaSafeCall(cudaMemcpy(&h_obj_val, d_obj_val, sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_obj_val");
	cudaSafeCall(cudaFree(d_obj_val), "Error in cudaFree d_obj_val");

	//	printMemoryUsage ();
	//	printf("used = %f MB\n", memory/1024.0/1024.0);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	if (count_time)
		times[6] += time;

	return 100;
}

// Kernel for reducing the rows by subtracting row minimum from each row element.
__global__ void kernel_rowReduction(int *d_costs, int N)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	int min = INF;

	if (rowid < N)
	{
		for (int colid = 0; colid < N; colid++)
		{
			int val = d_costs[rowid * N + colid];
			if (val < min)
			{
				min = val;
			}
		}

		for (int colid = 0; colid < N; colid++)
		{
			d_costs[rowid * N + colid] -= min;
		}
	}
}

// Kernel for reducing the column by subtracting column minimum from each column element.
__global__ void kernel_columnReduction(int *d_costs, int N)
{
	int colid = blockIdx.x * blockDim.x + threadIdx.x;
	int min = INF;

	if (colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			int val = d_costs[rowid * N + colid];
			if (val < min)
			{
				min = val;
			}
		}

		for (int rowid = 0; rowid < N; rowid++)
		{
			d_costs[rowid * N + colid] -= min;
		}
	}
}

// Kernel for calculating initial assignments.
__global__ void kernel_computeInitialAssignments(char *d_masks, int *d_costs, int *d_row_lock, int *d_col_lock, int N)
{
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	if (colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			if (d_col_lock[colid] == 1)
				break;

			if (d_costs[rowid * N + colid] == 0)
			{
				if (atomicCAS(&d_row_lock[rowid], 0, 1) == 0)
				{
					d_masks[rowid * N + colid] = STAR;
					d_col_lock[colid] = 1;
				}
			}
		}
	}
}

// Kernel for populating the assignment arrays and cover arrays.
__global__ void kernel_populateAssignments(short *d_row_assignments, short *d_col_assignments, int *d_row_covers, char *d_masks, int *d_costs, int N)
{
	int rowid = blockIdx.y * blockDim.y + threadIdx.y;
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	int id = rowid * N + colid;

	// Copy the matrix into shared memory

	int cost = (rowid < N && colid < N) ? d_costs[id] : INF;
	char mask = (rowid < N && colid < N) ? d_masks[id] : NORMAL;

	mask = (mask == STAR) ? STAR : ((cost == 0) ? ZERO : NORMAL);

	// Copy the predicate matrix back to global memory
	if (rowid < N && colid < N)
	{
		d_masks[id] = mask;
		if (mask == STAR)
		{
			d_row_assignments[rowid] = (short)colid;
			d_col_assignments[colid] = (short)rowid;
			d_row_covers[rowid] = 1;
		}
	}
}

// Kernel for updating the dual reduced costs in Step 5, without using atomic functions.
__global__ void kernel_dualUpdate_1_nonAtomic(int *d_min_val, int *d_costs, int *d_row_cover, int *d_col_cover, int N)
{
	int colid = blockIdx.x * blockDim.x + threadIdx.x;

	if (colid < N)
	{
		for (int rowid = 0; rowid < N; rowid++)
		{
			int cost = d_costs[rowid * N + colid];
			if (d_row_cover[rowid] == 0 && d_col_cover[colid] == 0)
			{
				if (cost < d_min_val[colid])
					d_min_val[colid] = cost;
			}
		}
	}
}

// Kernel for updating the dual reduced costs in Step 5.
__global__ void kernel_dualUpdate_1(int *d_min_val, int *d_costs, int *d_row_cover, int *d_col_cover, int N)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	int colid = blockIdx.y * blockDim.y + threadIdx.y;

	int id = rowid * N + colid;

	int cost = (rowid < N && colid < N) ? d_costs[id] : INF;
	int row_cover = (rowid < N) ? d_row_cover[rowid] : -1;
	int col_cover = (colid < N) ? d_col_cover[colid] : -1;

	if (rowid < N && colid < N)
	{
		if (row_cover == 0 && col_cover == 0)
			atomicMin(d_min_val, cost);
	}
}

// Kernel for updating the dual reduced costs in Step 5.
__global__ void kernel_dualUpdate_2(int d_min_val, char *d_masks, int *d_costs, int *d_row_cover, int *d_col_cover, short *d_row_visited, int N)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	int colid = blockIdx.y * blockDim.y + threadIdx.y;

	int id = rowid * N + colid;

	int cost = (rowid < N && colid < N) ? d_costs[id] : INF;
	int minval = (rowid < N && colid < N) ? d_min_val : (0 - INF);
	int row_cover = (rowid < N) ? d_row_cover[rowid] : -1;
	int col_cover = (colid < N) ? d_col_cover[colid] : -1;

	if (rowid < N && colid < N)
	{
		if (row_cover == 0 && col_cover == 0)
		{
			d_costs[id] = cost - minval;
			if (cost == minval)
			{
				d_row_visited[rowid] = ACTIVE;
				d_masks[id] = ZERO;
			}
		}

		else if (row_cover == 1 && col_cover == 1)
			d_costs[id] = cost + minval;
	}
}

// Kernel for calculating the optimal assignment cost.
__global__ void kernel_finalCost(int *d_obj_val, int *d_costs, char *d_masks, int N)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;
	int colid = blockIdx.y * blockDim.y + threadIdx.y;

	int id = rowid * N + colid;

	int cost = (rowid < N && colid < N) ? d_costs[id] : INF;
	char mask = (rowid < N && colid < N) ? d_masks[id] : NORMAL;

	if (mask == STAR)
		atomicAdd(d_obj_val, cost);
}