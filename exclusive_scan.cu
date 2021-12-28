/*
 * Created by Ketan Date
 */

#include "include/exclusive_scan.h"

// Host code for performing exclusive scan on large array. Scan kernels are executed recursively.
// Returns the sum of all the predicates.
int recursiveScan(int *d_in, int size)
{
	int sum = 0;
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, size);
	if (total_blocks > MAX_GRIDSIZE)
		calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, size);

	int *d_block_sum;
	cudaSafeCall(cudaMalloc(&d_block_sum, total_blocks * sizeof(int)), "Error in cudaMalloc d_block_sum");
	cudaSafeCall(cudaMemset(d_block_sum, 0, total_blocks * sizeof(int)), "Error in cudaMemset d_block_sum");

	kernel_exclusiveScan<<<blocks_per_grid, threads_per_block>>>(d_in, d_block_sum, size);

	if (total_blocks > 1)
	{
		sum = recursiveScan(d_block_sum, total_blocks);
	}

	if (total_blocks == 1)
	{
		cudaSafeCall(cudaMemcpy(&sum, d_block_sum, sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_block_sum");
		cudaSafeCall(cudaMemset(d_block_sum, 0, sizeof(int)), "Error in cudaMemset d_block_sum");
	}

	kernel_uniformUpdate<<<blocks_per_grid, threads_per_block>>>(d_in, d_block_sum, size);

	cudaSafeCall(cudaFree(d_block_sum), "Error in cudaFree d_block_sum");
	d_block_sum = NULL;

	return sum;
}

// Kernel for calculating exclusive sum scan for individual blocks. Blelloch's exclusive scan algorithm.
__global__ void kernel_exclusiveScan(int *d_in, int *d_block_sum, int size)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	const unsigned int H_TCOUNT = TCOUNT >> 1;

	// If there are more than 65535 blocks in a grid, calculate global blockid. Use it to calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int g_tid = blockid * TCOUNT + tid;

	__shared__ int s_temp[TCOUNT];

	// Copy Predicate Matrix into shared memory. Extra elements are set to zero.

	s_temp[tid] = (g_tid < size) ? d_in[g_tid] : 0;
	__syncthreads();

	// Reduce phase

	for (unsigned int s = 1; s < TCOUNT; s <<= 1)
	{
		int index = 2 * s * (TCOUNT - 1 - tid);
		if (index < TCOUNT)
		{
			int tidx2 = (TCOUNT - 1 - index);
			s_temp[tidx2] += s_temp[tidx2 - s];
		}
		__syncthreads();
	}
	__syncthreads();

	// Downsweep phase
	if (tid == 0)
	{
		d_block_sum[blockid] = s_temp[TCOUNT - 1];
		s_temp[TCOUNT - 1] = 0;
	}
	__syncthreads();

	for (unsigned int s = (H_TCOUNT); s > 0; s >>= 1)
	{
		int index = 2 * s * (TCOUNT - 1 - tid);
		if (index < TCOUNT)
		{
			int tidx2 = (TCOUNT - 1 - index);
			int temp = s_temp[tidx2];
			s_temp[tidx2] += s_temp[tidx2 - s];
			s_temp[tidx2 - s] = temp;
		}
		__syncthreads();
	}
	__syncthreads();

	if (g_tid < size)
		d_in[g_tid] = s_temp[tid]; // Copy scatter addresses back to global memory
}

// Kernel for uniformly updating input array by incrementing with the block sums.
__global__ void kernel_uniformUpdate(int *d_in, int *d_block_sum, int size)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;

	// If there are more than 65535 blocks in a grid, calculate global blockid. Use it to calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int g_tid = blockid * TCOUNT + tid;

	__shared__ int s_block_sum;

	// Copy Predicate Matrix and Block sums into shared memory. Extra elements are set to zero. Block sum of previous block is loaded for incrementing scatter addresses.

	int val = (g_tid < size) ? d_in[g_tid] : 0;

	if (tid == 0)
		s_block_sum = d_block_sum[blockid];

	__syncthreads();

	// Increment the scatter addresses by the block sum of previous block.
	val += s_block_sum;

	__syncthreads();

	// Update scatter addresses in global memory.
	if (g_tid < size)
		d_in[g_tid] = val;
}
