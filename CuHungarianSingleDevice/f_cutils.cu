/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*
*/

#include "f_cutils.h"

std::default_random_engine generator(SEED);

// Function for generating problem with uniformly distributed integer costs between [0, COSTRANGE].
void generateProblem(float *cost_matrix, int SP, int N, int costrange) {

	long N2 = SP * N * N;

	std::uniform_int_distribution<int> distribution(0, costrange);

	for (long i = 0; i < N2; i++) {
		int val = distribution(generator);
		cost_matrix[i] = (float) val;

	}
}


// Helper function for printing device errors.
void cudaSafeCall(cudaError_t error, const char *message) {
	if (error != cudaSuccess) {
		std::cerr << "Error " << error << ": " << message << ": " << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
}

// Helper function for printing device memory info.
void printMemoryUsage(float memory) {
	size_t free_byte;
	size_t total_byte;

	cudaSafeCall(cudaMemGetInfo(&free_byte, &total_byte), "Error in cudaMemGetInfo");

	float free_db = (float) free_byte;
	float total_db = (float) total_byte;
	float used_db = total_db - free_db;

	if (memory < used_db)
		memory = used_db;

	printf("used = %f MB, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

// Function for calculating grid and block dimensions from the given input size.
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size) {
	threads_per_block.x = BLOCKDIMX * BLOCKDIMY;

	int value = size / threads_per_block.x;
	if (size % threads_per_block.x > 0)
		value++;

	total_blocks = value;
	blocks_per_grid.x = value;

}

// Function for calculating grid and block dimensions from the given input size for square grid.
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size) {
	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int sq_size = (int) ceil(sqrt(size)); 

	int valuex = (int) ceil((float) (sq_size) / BLOCKDIMX);
	int valuey = (int) ceil((float) (sq_size) / BLOCKDIMY);

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

// Function for calculating grid and block dimensions from the given input size for rectangular grid.
void calculateRectangularDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize) {

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int valuex = xsize / threads_per_block.x;
	if (xsize % threads_per_block.x > 0)
		valuex++;

	int valuey = ysize / threads_per_block.y;
	if (ysize % threads_per_block.y > 0)
		valuey++;

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

// Function for calculating grid and block dimensions from the given input size for cubic grid.
void calculateCubicDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int xsize, int ysize, int zsize) {

	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;
	threads_per_block.z = BLOCKDIMZ;

	int valuex = xsize / threads_per_block.x;
	if (xsize % threads_per_block.x > 0)
		valuex++;

	int valuey = ysize / threads_per_block.y;
	if (ysize % threads_per_block.y > 0)
		valuey++;

	int valuez = zsize / threads_per_block.z;
	if (zsize % threads_per_block.z > 0)
		valuez++;

	total_blocks = valuex * valuey * valuez;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
	blocks_per_grid.z = valuez;
}

// Function for printing the output log.
void printLog(int prno, int repetition, int numprocs, int numdev, int costrange, long obj_val, int init_assignments, float total_time, int *stepcounts, float *steptimes, const char *logpath, int N) {
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << numprocs << "\t" << numdev << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t" << stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6] << "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t" << steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t"
			<< steptimes[7] << "\t" << steptimes[8] << "\t" << total_time << std::endl;

	logfile.close();
}

// Function for sequential exclusive scan.
void exclusiveSumScan(int *array, int size) {

	int sum = 0;
	int val = 0;

	for (int i = 0; i <= size; i++) {
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for sequential exclusive scan.
void exclusiveSumScan(long *array, int size) {

	long sum = 0;
	long val = 0;

	for (int i = 0; i <= size; i++) {
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for sequential exclusive scan.
void exclusiveSumScan(long *array, long size) {

	long sum = 0;
	long val = 0;

	for (long i = 0; i <= size; i++) {
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}

// Function for reducing an array (SUM operation)
int reduceSUM(int *array, int size) {
	int val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(long *array, int size) {
	long val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
float reduceSUM(float *array, int size) {
	float val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(long *array, long size) {
	long val = 0;

	for (long i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
long reduceSUM(int *array, long size) {
	long val = 0;

	for (int i = 0; i < size; i++) {
		val += array[i];
	}

	return val;
}

// Function for reducing an array (SUM operation)
float reduceMIN(float *array, int size) {
	float val = INF;

	for (int i = 0; i < size; i++) {
		if (array[i] <= val - EPSILON)
			val = array[i];
	}

	return val;
}

// Function for reducing an array (OR operation)
bool reduceOR(bool *array, int size) {
	bool val = false;

	for (int i = 0; i < size; i++) {
		val = val || array[i];
	}

	return val;
}

void printDebugArray(int *d_array, int size, const char *name) {

	int *h_array = new int[size];

	std::cout << name  << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray(long *d_array, int size, const char *name) {

	long *h_array = new long[size];

	std::cout << name  << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(long), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugArray(float *d_array, int size, const char *name) {

	float *h_array = new float[size];

	std::cout << name << std::endl;
	cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;

	delete[] h_array;
}

void printDebugMatrix(float *d_matrix, int rowsize, int colsize, const char *name) {
	float *h_matrix = new float[rowsize * colsize];

	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] h_matrix;
}

void printDebugMatrix(int *d_matrix, int rowsize, int colsize, const char *name) {
	int *h_matrix = new int[rowsize * colsize];

	std::cout << name << std::endl;
	cudaMemcpy(h_matrix, d_matrix, rowsize * colsize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] h_matrix;
}

void printHostArray(int *h_array, int size, const char *name) {
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostArray(float *h_array, int size, const char *name) {
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostArray_tofile(float *h_array, int size, const char *name, std::ofstream &out) {
	out << name << std::endl;

	for (int i = 0; i < size; i++) {
		out << h_array[i] << "\t";
	}
	out << std::endl;
}

void printHostMatrix(float *h_matrix, int rowsize, int colsize, const char *name) {

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

}
void printHostMatrix_tofile(float *h_matrix, int rowsize, int colsize, const char *name, std::ofstream &out) {

	out << name << std::endl;
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			out << h_matrix[i * colsize + j] << "\t";
		}
		out << std::endl;
	}

	out << std::endl;

}


void printHostMatrix(int *h_matrix, int rowsize, int colsize, const char *name) {

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++) {
		for (int j = 0; j < colsize; j++) {
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

}

void printHostArray(long *h_array, int size, const char *name) {
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++) {
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

// Function for reading specified input file.
void readFile(float *cost_matrix, const char *filename) {
	std::string s = filename;
	std::ifstream myfile(s.c_str());

//	if (myfile == 0) {
//		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
//		exit(-1);
//	}

	while (myfile.is_open() && myfile.good()) {
		int N = 0;
		myfile >> N;

		long N2 = N * N;

		for (long i = 0; i < N2; i++) {
			myfile >> cost_matrix[i];

		}
	}

	myfile.close();

}

// This function splits "val" equally among the elements "array" of length equal to "size."
void split(int *array, int val, int size) {

	int split_val = val / size;
	int overflow = val % size;

	std::fill(array, array + size, split_val);

	if (overflow > 0) {
		for (int i = 0; i < size; i++) {
			array[i]++;
			overflow--;
			if (overflow == 0)
				break;
		}
	}
}

// This function splits "val" equally among the elements "array" of length equal to "size."
void split(long *array, long val, int size) {

	long split_val = val / size;
	long overflow = val % size;

	std::fill(array, array + size, split_val);

	if (overflow > 0) {
		for (int i = 0; i < size; i++) {
			array[i]++;
			overflow--;
			if (overflow == 0)
				break;
		}
	}
}

/*
 *	This is a test kernel.
 */
__global__ void kernel_memSet(int *array, int val, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
	{
		array[id] = val;
	}
}

/*
 *	This is a test kernel.
 */
__global__ void kernel_memSet(float *array, float val, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
	{
		array[id] = val;
	}
}

__global__ void kernel_memSet(long *array, long val, int size)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id < size)
	{
		array[id] = val;
	}
}
