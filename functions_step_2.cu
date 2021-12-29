/*
 * Created by Ketan Date
 */

#include "include/functions_step_2.h"

// Function for initializing all devices for execution of Step 2.
void initializeStep2(Vertices h_vertices, Vertices *d_vertices_dev, VertexData *d_row_data_dev, VertexData *d_col_data_dev, int N, unsigned int devid)
{
	cudaSetDevice(devid);

	cudaSafeCall(cudaMemcpy(h_vertices.row_assignments, d_vertices_dev[devid].row_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_vertices_dev[devid].row_assignments");
	cudaSafeCall(cudaMemcpy(h_vertices.col_assignments, d_vertices_dev[devid].col_assignments, N * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_vertices_dev[devid].col_assignments");

	cudaSafeCall(cudaMemset(d_vertices_dev[devid].row_covers, 0, N * sizeof(int)), "Error in cudaMemset d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices_dev[devid].col_covers, 0, N * sizeof(int)), "Error in cudaMemset d_col_covers");

	cudaSafeCall(cudaMemset(d_row_data_dev[devid].is_visited, DORMANT, N * sizeof(int)), "Error in cudaMemset d_row_data.is_visited");
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].is_visited, DORMANT, N * sizeof(int)), "Error in cudaMemset d_col_data.is_visited"); // initialize "visited" array for columns. later used in BFS (Step 4).
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].slack, INF, N * sizeof(double)), "Error in cudaMemset d_col_data.slack");

	cudaSafeCall(cudaMemset(d_row_data_dev[devid].parents, -1, N * sizeof(int)), "Error in cudaMemset d_row_data.parents");
	cudaSafeCall(cudaMemset(d_row_data_dev[devid].children, -1, N * sizeof(int)), "Error in cudaMemset d_row_data.children");
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].parents, -1, N * sizeof(int)), "Error in cudaMemset d_col_data.parents");
	cudaSafeCall(cudaMemset(d_col_data_dev[devid].children, -1, N * sizeof(int)), "Error in cudaMemset d_col_data.children");
}

// Function for finding row cover on individual devices.
int computeRowCovers(Vertices *d_vertices_dev, int N, unsigned int devid)
{

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks = 0;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);
	kernel_computeRowCovers<<<blocks_per_grid, threads_per_block>>>(d_vertices_dev[devid].row_assignments, d_vertices_dev[devid].row_covers, N); // Kernel execution.
	cudaSafeCall(cudaGetLastError(), "Error in kernel_computeRowCovers execution");

	thrust::device_ptr<int> ptr(d_vertices_dev[devid].row_covers);

	int cover_count = thrust::reduce(ptr, ptr + N);

	return cover_count;
}

// Function for copying row cover array back to each device.
void updateRowCovers(Vertices *d_vertices_dev, int *h_row_covers, int N, unsigned int devid)
{
	cudaSetDevice(devid);
	cudaSafeCall(cudaMemcpy(d_vertices_dev[devid].row_covers, h_row_covers, N * sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_row_covers");
}

// Kernel for populating the assignment arrays and cover arrays.
__global__ void kernel_computeRowCovers(int *d_row_assignments, int *d_row_covers, int row_count)
{
	int rowid = blockIdx.x * blockDim.x + threadIdx.x;

	// Copy the predicate matrix back to global memory
	if (rowid < row_count)
	{
		if (d_row_assignments[rowid] != -1)
		{
			d_row_covers[rowid] = 1;
		}
	}
}
