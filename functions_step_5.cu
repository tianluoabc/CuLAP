/*
 * Created by Ketan Date
 */

#include "include/functions_step_5.h"

void computeTheta(double &h_device_min, Matrix *d_costs_dev, Vertices *d_vertices_dev, VertexData *d_col_data_dev, int N, unsigned int devid)
{

	dim3 blocks_per_grid;
	dim3 threads_per_block;
	int total_blocks;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, N);

	double *temp = new double[N];
	int *temp2 = new int[N];

	///////////////////////////////////////////////////////////////////
	//printDebugArray(d_col_data_dev[devid].slack, N, "Slack", 0);
	///////////////////////////////////////////////////////////////////

	cudaSafeCall(cudaMemcpy(temp, d_col_data_dev[devid].slack, N * sizeof(double), cudaMemcpyDeviceToHost), "Error in cudaMalloc d_device_min");
	cudaSafeCall(cudaMemcpy(temp2, d_vertices_dev[devid].col_covers, N * sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMalloc d_device_min");

	h_device_min = INF;

	for (int j = 0; j < N; j++)
	{
		double slack = temp[j];
		if (temp2[j] == 0)
		{
			h_device_min = slack < h_device_min ? slack : h_device_min;
		}
	}

	h_device_min /= 2;

	kernel_dualUpdate_2<<<blocks_per_grid, threads_per_block>>>(h_device_min, d_costs_dev[devid].row_duals, d_costs_dev[devid].col_duals, d_vertices_dev[devid].row_covers, d_vertices_dev[devid].col_covers, 0, N, N);
	cudaSafeCall(cudaGetLastError(), "Error in kernel_dualUpdate_2 execution");

	delete[] temp;
	delete[] temp2;
}

// Kernel for updating the dual reduced costs in Step 5.
__global__ void kernel_dualUpdate_2(double d_min_val, double *d_row_duals, double *d_col_duals, int *d_row_cover, int *d_col_cover, int row_start, int row_count, int N)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	int row_cover = (id < row_count) ? d_row_cover[id + row_start] : -1;
	int col_cover = (id < N) ? d_col_cover[id] : -1;

	if (id < N)
	{

		if (row_cover == 0) // Row is labeled
		{
			d_row_duals[id] += d_min_val;
		}

		else
		{
			d_row_duals[id] -= d_min_val;
		}

		if (col_cover == 1) // Column is labeled
		{
			d_col_duals[id] -= d_min_val;
		}

		else
		{
			d_col_duals[id] += d_min_val;
		}
	}
}
