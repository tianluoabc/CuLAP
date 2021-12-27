/*
 * Developed by Ketan Date
 */

#include "include/helper_utils.h"
#include <cstring>

CRandomSFMT randomGenerator(SEED);

int N = 0;
int N2 = 0;
int M = 0;
double memory = 0;
Matrix h_costs, h_red_costs;
Vertices d_vertices;
Edges d_edges;
VertexData d_row_data, d_col_data;
CompactEdges d_edges_csr;

// Function for initializing vertex matrices on device.
void initialize(void)
{
	d_vertices.row_assignments = new int[N];
	d_vertices.col_assignments = new int[N];
	d_vertices.row_covers = new int[N];
	d_vertices.col_covers = new int[N];

	d_edges.costs = new int[N2];
	d_edges.masks = new int[N2];

	d_row_data.parents = new int[N];
	d_row_data.is_visited = new int[N];

	d_col_data.parents = new int[N];
	d_col_data.is_visited = new int[N];

	memset(d_vertices.row_assignments, -1, N * sizeof(int));
	memset(d_vertices.col_assignments, -1, N * sizeof(int));
	memset(d_vertices.row_covers, 0, N * sizeof(int));
	memset(d_vertices.col_covers, 0, N * sizeof(int));

	memcpy(d_edges.costs, h_costs.elements, N2 * sizeof(int));
	memset(d_edges.masks, NORMAL, N2 * sizeof(int));

	memset(d_row_data.parents, -1, N * sizeof(int));
	memset(d_row_data.is_visited, 0, N * sizeof(int));

	memset(d_col_data.parents, -1, N * sizeof(int));
	memset(d_col_data.is_visited, 0, N * sizeof(int));
}

// Function for initializing edge matrices on device.
void finalize(void)
{

	delete[] d_vertices.row_assignments;
	delete[] d_vertices.col_assignments;
	delete[] d_vertices.row_covers;
	delete[] d_vertices.col_covers;

	delete[] d_edges.costs;
	delete[] d_edges.masks;

	delete[] d_row_data.parents;
	delete[] d_row_data.is_visited;

	delete[] d_col_data.parents;
	delete[] d_col_data.is_visited;
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

// Function for printing the output log.
void printLog(int prno, int repetition, int costrange, int obj_val, int init_assignments, double total_time, int *stepcounts, double *steptimes, const char *logpath)
{
	std::ofstream logfile(logpath, std::ios_base::app);

	logfile << prno << "\t" << N << "\t[0, " << costrange << "]\t" << obj_val << "\t" << init_assignments << "\t"
			<< stepcounts[0] << "\t" << stepcounts[1] << "\t" << stepcounts[2] << "\t" << stepcounts[3] << "\t" << stepcounts[4] << "\t" << stepcounts[5] << "\t" << stepcounts[6]
			<< "\t" << steptimes[0] << "\t" << steptimes[1] << "\t" << steptimes[2] << "\t" << steptimes[3] << "\t" << steptimes[4] << "\t" << steptimes[5] << "\t" << steptimes[6] << "\t" << total_time << std::endl;

	logfile.close();
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

// Function for sequential exclusive scan.
void exclusiveSumScan(int *array, int size)
{

	int sum = 0;
	int val = 0;

	for (int i = 0; i <= size; i++)
	{
		sum += val;
		val = array[i];
		array[i] = sum;
	}
}