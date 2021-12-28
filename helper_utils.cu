/*
 * Developed by Ketan Date
 */

#include "include/helper_utils.h"

CRandomSFMT randomGenerator(SEED);

int N = 0;
int N2 = 0;
int M = 0;
double memory = 0;
Matrix h_costs;
Matrix h_red_costs;
Vertices d_vertices;
Edges d_edges;
CompactEdges h_edges_csr, h_edges_csc;
CompactEdges d_edges_csr, d_edges_csc;

// Helper function for printing device errors.
void cudaSafeCall(cudaError_t error, char *message)
{
	if (error != cudaSuccess)
	{
		std::cerr << "Error " << error << ": " << message << ": " << cudaGetErrorString(error) << std::endl;
		exit(-1);
	}
}

// Helper function for printing device memory info.
void printMemoryUsage(void)
{
	size_t free_byte;
	size_t total_byte;

	cudaSafeCall(cudaMemGetInfo(&free_byte, &total_byte), "Error in cudaMemGetInfo");

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	if (memory < used_db)
		memory = used_db;

	printf("used = %f MB, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

// Helper function for initializing global variables and arrays.
void initialize(void)
{
	M = 0;
	initDeviceVertices();
	initDeviceEdges();
}

// Function for initializing vertex matrices on device.
void initDeviceVertices(void)
{
	cudaSafeCall(cudaMalloc(&d_vertices.row_assignments, N * sizeof(short)), "error in cudaMalloc d_row_assignment");
	cudaSafeCall(cudaMalloc(&d_vertices.col_assignments, N * sizeof(short)), "error in cudaMalloc d_col_assignment");
	cudaSafeCall(cudaMalloc(&d_vertices.row_covers, N * sizeof(int)), "error in cudaMalloc d_row_covers");
	cudaSafeCall(cudaMalloc(&d_vertices.col_covers, N * sizeof(int)), "error in cudaMalloc d_col_covers");

	cudaSafeCall(cudaMemset(d_vertices.row_assignments, -1, N * sizeof(short)), "Error in cudaMemset d_row_assignment");
	cudaSafeCall(cudaMemset(d_vertices.col_assignments, -1, N * sizeof(short)), "Error in cudaMemset d_col_assignment");
	cudaSafeCall(cudaMemset(d_vertices.row_covers, 0, N * sizeof(int)), "Error in cudaMemset d_row_covers");
	cudaSafeCall(cudaMemset(d_vertices.col_covers, 0, N * sizeof(int)), "Error in cudaMemset d_col_covers");
}

// Function for initializing edge matrices on device.
void initDeviceEdges(void)
{

	cudaSafeCall(cudaMalloc(&d_edges.costs, N2 * sizeof(int)), "error in cudaMalloc d_edges.costs");
	cudaSafeCall(cudaMalloc(&d_edges.masks, N2 * sizeof(char)), "error in cudaMalloc d_edges.masks");
	cudaSafeCall(cudaMemcpy(d_edges.costs, h_costs.elements, N2 * sizeof(int), cudaMemcpyHostToDevice), "Error in cudaMemcpy h_costs");
	cudaSafeCall(cudaMemset(d_edges.masks, NORMAL, N2 * sizeof(char)), "Error in cudaMemset d_edges.masks");

	cudaSafeCall(cudaMalloc(&d_edges_csr.is_visited, N * sizeof(short)), "Error in cudaMalloc d_edges_csr.is_visited");
	cudaSafeCall(cudaMalloc(&d_edges_csc.is_visited, N * sizeof(short)), "Error in cudaMalloc d_edges_csc.is_visited");
}

// Function for freeing device memory.
void finalize(void)
{
	cudaSafeCall(cudaFree(d_vertices.row_assignments), "Error in cudaFree d_row_assignment");
	cudaSafeCall(cudaFree(d_vertices.col_assignments), "Error in cudaFree d_col_assignment");
	cudaSafeCall(cudaFree(d_vertices.row_covers), "Error in cudaFree d_row_covers");
	cudaSafeCall(cudaFree(d_vertices.col_covers), "Error in cudaFree d_col_covers");

	cudaSafeCall(cudaFree(d_edges.costs), "error in cudaFree d_edges.costs");
	cudaSafeCall(cudaFree(d_edges.masks), "error in cudaFree d_edges.masks");

	cudaSafeCall(cudaFree(d_edges_csr.is_visited), "Error in cudaFree d_edges_csr.is_visited");
	cudaSafeCall(cudaFree(d_edges_csc.is_visited), "Error in cudaFree d_edges_csc.is_visited");

	cudaDeviceReset();
}

// Function for reading specified input file.
void readFile(const char *filename)
{
	std::string s = filename;
	std::ifstream myfile(s.c_str());

	if (!myfile.good())
	{
		std::cerr << "Error: input file not found: " << s.c_str() << std::endl;
		exit(-1);
	}

	while (myfile.is_open() && myfile.good())
	{
		myfile >> N;
		N2 = N * N;
		h_costs.rowsize = h_costs.colsize = N;
		h_costs.elements = new int[N2];

		for (int k = 0; k < N2; k++)
		{
			myfile >> h_costs.elements[k];
		}
		myfile.close();
	}
}

// Function for generating problem with uniformly distributed integer costs between [0, COSTRANGE].
void generateProblem(int problemsize, int costrange)
{
	N = problemsize;
	N2 = N * N;
	h_costs.rowsize = h_costs.colsize = N;
	h_costs.elements = new int[N2];

	for (int i = 0; i < N2; i++)
	{
		h_costs.elements[i] = randomGenerator.IRandomX(0, costrange);
	}
}

// Function for calculating grid and block dimensions from the given input size.
void calculateLinearDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size)
{
	threads_per_block.x = BLOCKDIMX * BLOCKDIMY;
	int value = (int)ceil((double)(size) / threads_per_block.x);
	total_blocks = value;
	blocks_per_grid.x = value;
}

// Function for calculating grid and block dimensions from the given input size for square grid.
void calculateSquareDims(dim3 &blocks_per_grid, dim3 &threads_per_block, int &total_blocks, int size)
{
	threads_per_block.x = BLOCKDIMX;
	threads_per_block.y = BLOCKDIMY;

	int sq_size = (int)ceil(sqrt(size));

	int valuex = (int)ceil((double)(sq_size) / BLOCKDIMX);
	int valuey = (int)ceil((double)(sq_size) / BLOCKDIMY);

	total_blocks = valuex * valuey;
	blocks_per_grid.x = valuex;
	blocks_per_grid.y = valuey;
}

// Function for printing the output log.
void printLog(int prno, int repetition, int costrange, int obj_val, int init_assignments, int total_time, int *stepcounts, float *steptimes, const char *logpath)
{
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t"
			<< stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6]
			<< "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t" << steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t" << total_time << std::endl;

	logfile.close();
}