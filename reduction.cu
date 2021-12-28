/*
 * Created by Ketan Date
 */

#include "include/reduction.h"

// Host code for performing parallel minimization on large array. Scan kernels are executed recursively.
// Returns the minimum of all the elements.
int recursiveMin(int *d_in, int size)
{
	int min = INF;
	int total_blocks = 0;
	dim3 blocks_per_grid;
	dim3 threads_per_block;

	calculateLinearDims(blocks_per_grid, threads_per_block, total_blocks, size);
	if (total_blocks > MAX_GRIDSIZE)
		calculateSquareDims(blocks_per_grid, threads_per_block, total_blocks, size);

	int *d_block_min;
	cudaSafeCall(cudaMalloc(&d_block_min, total_blocks * sizeof(int)), "Error in cudaMalloc d_block_min");
	cudaSafeCall(cudaMemset(d_block_min, INF, total_blocks * sizeof(int)), "Error in cudaMemset d_block_min");

	kernel_parallelMin<<<blocks_per_grid, threads_per_block>>>(d_in, d_block_min, size);

	if (total_blocks > 1)
	{
		min = recursiveMin(d_block_min, total_blocks);
	}

	if (total_blocks == 1)
	{
		cudaSafeCall(cudaMemcpy(&min, d_block_min, sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_block_min");
	}

	cudaSafeCall(cudaFree(d_block_min), "Error in cudaFree d_block_min");
	d_block_min = NULL;

	return min;
}

// Host code for performing parallel sum on large array. Scan kernels are executed recursively.
// Returns the sum of all the elements.
int recursiveSum(int *d_in, int size)
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

	kernel_parallelSum<<<blocks_per_grid, threads_per_block>>>(d_in, d_block_sum, size);

	if (total_blocks > 1)
	{
		sum = recursiveSum(d_block_sum, total_blocks);
	}

	if (total_blocks == 1)
	{
		cudaSafeCall(cudaMemcpy(&sum, d_block_sum, sizeof(int), cudaMemcpyDeviceToHost), "Error in cudaMemcpy d_block_sum");
	}

	cudaSafeCall(cudaFree(d_block_sum), "Error in cudaFree d_block_sum");
	d_block_sum = NULL;

	return sum;
}

// Kernel for calculating min of all elements.
__global__ void kernel_parallelMin(int *d_in, int *d_block_min, int size)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	const unsigned int H_TCOUNT = TCOUNT >> 1;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int g_tid = blockid * TCOUNT + tid;

	__shared__ int s_pred[TCOUNT];

	// Copy Predicate Matrix into shared memory with sequential addressing. Extra elements are set to zero.
	s_pred[tid] = (g_tid < size) ? d_in[g_tid] : INF;
	__syncthreads();

	// Reduce phase
	for (unsigned int s = H_TCOUNT; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			s_pred[tid] = (s_pred[tid] < s_pred[tid + s]) ? s_pred[tid] : s_pred[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid == 0)
	{
		d_block_min[blockid] = s_pred[0];
	}
}

// Kernel for calculating sum of all elements.
__global__ void kernel_parallelSum(int *d_in, int *d_block_min, int size)
{
	const unsigned int TCOUNT = BLOCKDIMX * BLOCKDIMY;
	const unsigned int H_TCOUNT = TCOUNT >> 1;

	// Calculate golbal threadid.
	int blockid = blockIdx.y * gridDim.x + blockIdx.x;
	int tid = threadIdx.y * blockDim.x + threadIdx.x;
	int g_tid = blockid * TCOUNT + tid;

	__shared__ int s_pred[TCOUNT];

	// Copy Predicate Matrix into shared memory with sequential addressing. Extra elements are set to zero.
	s_pred[tid] = (g_tid < size) ? d_in[g_tid] : 0;
	__syncthreads();

	// Reduce phase
	for (unsigned int s = H_TCOUNT; s > 0; s >>= 1)
	{
		if (tid < s)
		{
			s_pred[tid] += s_pred[tid + s];
		}
		__syncthreads();
	}
	__syncthreads();

	if (tid == 0)
	{
		d_block_min[blockid] = s_pred[0];
	}
}

__device__ void __pad_array(int *out_array, int *in_array, int tid, int s)
{
	unsigned int q = tid << 1;
	out_array[tid] = in_array[q];
	out_array[tid + s] = in_array[q + 1];
}

__device__ void __unpad_array(int *out_array, int *in_array, int tid, int s)
{
	unsigned int q = tid << 1;
	out_array[q] = in_array[tid];
	out_array[q + 1] = in_array[tid + s];
}

__device__ void __copy_array(int *out_array, int *in_array, int tid, int s)
{
	out_array[tid] = in_array[tid];
	out_array[tid + s] = in_array[tid + s];
}
