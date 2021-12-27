/*
 * Developed by Ketan Date
 */

#include "include/helper_utils.h"

CRandomSFMT randomGenerator(SEED);

double memory = 0;

// Function for reading specified input file.
void readFile(double *cost_matrix, const char *filename)
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
		int N = 0;
		myfile >> N;

		long N2 = N * N;

		for (long i = 0; i < N2; i++)
		{
			double val = 0;
			myfile >> val;

			cost_matrix[i] = val;
		}
	}

	myfile.close();
}

// Function for generating problem with uniformly distributed integer costs between [0, COSTRANGE].
void generateProblem(int *cost_matrix, int N, int costrange)
{

	long N2 = N * N;

	for (long i = 0; i < N2; i++)
	{
		int val = randomGenerator.IRandomX(0, costrange);
		cost_matrix[i] = val;
	}
}

// Function for printing the output log.
void printLog(int prno, int repetition, int N, int costrange, double obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath)
{
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t"
			<< stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6]
			<< "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t" << steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t" << total_time << std::endl;

	logfile.close();
}

void printHostArray(int *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostArray(double *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

void printHostMatrix(double *h_matrix, int rowsize, int colsize, const char *name)
{

	std::cout << name << std::endl;
	for (int i = 0; i < rowsize; i++)
	{
		for (int j = 0; j < colsize; j++)
		{
			std::cout << h_matrix[i * colsize + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
}

void printHostArray(long *h_array, int size, const char *name)
{
	std::cout << name << std::endl;

	for (int i = 0; i < size; i++)
	{
		std::cout << h_array[i] << "\t";
	}
	std::cout << std::endl;
}

/*
void consoleOut(void) 
{
	int* printCosts = new int[N2];
	char* printMasks = new char[N2];

	cudaSafeCall(cudaMemcpy(printCosts, d_edges.costs, N2 * sizeof(int), cudaMemcpyDeviceToHost), "Error in print h_costs");
	cudaSafeCall(cudaMemcpy(printMasks, d_edges.masks, N2 * sizeof(char), cudaMemcpyDeviceToHost), "Error in print h_masks");

	std::cout << "Cost" << std::endl;

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			std::cout << printCosts[i * N + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	std::cout << "Mask" << std::endl;

	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			std::cout << printMasks[i * N + j] << "\t";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;

	delete[] printCosts;
	delete[] printMasks;
}


void printDebugInfo(void) 
{
	std::cout << std::endl;
	std::cout << "COSTS:" << std::endl;
	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			std::cout << d_edges.costs[i*N+j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "MASKS:" << std::endl;
	for(int i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			std::cout << d_edges.masks[i*N+j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "ROW COVERS:" << std::endl;
	for(int i=0; i<N; i++)
	{
		std::cout << d_vertices.row_covers[i] << "\t";
	}
	std::cout << std::endl;

	std::cout << "COL COVERS:" << std::endl;
	for(int i=0; i<N; i++)
	{
		std::cout << d_vertices.col_covers[i] << "\t";
	}
	std::cout << std::endl;

	std::cout << "ROW ASSGN:" << std::endl;
	for(int i=0; i<N; i++)
	{
		std::cout << d_vertices.row_assignments[i] << "\t";
	}
	std::cout << std::endl;

	std::cout << "COL ASSGN:" << std::endl;
	for(int i=0; i<N; i++)
	{
		std::cout << d_vertices.col_assignments[i] << "\t";
	}
	std::cout << std::endl;
	std::cout << std::endl;

}
*/
