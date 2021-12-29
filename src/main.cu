/*
 * Created by Ketan Date
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <omp.h>
#include "../include/structures.h"
#include "../include/variables.h"
#include "../include/helper_utils.h"
#include "../include/culap.h"
#include "../include/sfmt.h"

void modifyCosts(double *, int, int, int);

int main(int argc, char **argv)
{

	int size = atoi(argv[1]);
	int costrange = atoi(argv[2]);
	int problemcount = atoi(argv[3]);
	int repetitions = atoi(argv[4]);

	int numdev = atoi(argv[5]);
	int spcount = atoi(argv[6]);

	const char *filename = argv[7];

	int multiplier = 1;

	int init_assignments = 0;
	int stepcounts[7];
	double steptimes[9];

	std::fill(stepcounts, stepcounts + 7, 0);

	std::stringstream logpath;
	int problemsize = size;

	costrange = problemsize * 10;

	int *row_assignments;
	double *row_duals, *col_duals;
	double *cost_matrix;
	double *obj_val;

	int devid = 0;

	cudaSetDevice(devid);

	double *h_cost = new double[spcount * size * size];
	int *h_ass = new int[spcount * size];
	double *h_row_dual = new double[spcount * size];
	double *h_col_dual = new double[spcount * size];

	cudaMalloc((void **)&row_assignments, spcount * size * sizeof(int));
	cudaMalloc((void **)&row_duals, spcount * size * sizeof(double));
	cudaMalloc((void **)&col_duals, spcount * size * sizeof(double));
	cudaMalloc((void **)&cost_matrix, spcount * size * size * sizeof(double));
	cudaMalloc((void **)&obj_val, spcount * sizeof(double));

	cudaMemset(row_assignments, -1, spcount * size * sizeof(int));
	cudaMemset(row_duals, 0, spcount * size * sizeof(double));
	cudaMemset(col_duals, 0, spcount * size * sizeof(double));
	cudaMemset(obj_val, 0, spcount * sizeof(double));

	//	readFile(h_cost, filename, spcount);
	generateProblem(h_cost, spcount, size, costrange);

	cudaMemcpy(cost_matrix, h_cost, spcount * size * size * sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 0; i < repetitions; i++)
	{

		std::cout << "Size: " << problemsize << "\tCostrange: [0, " << costrange << "]" << std::endl;

		double start = omp_get_wtime();

		size_t total, free1, free2;

		//		cudaMemGetInfo(&free1, &total);

		CuLAP lpx(problemsize, spcount, devid, false, stepcounts);
		lpx.solve(cost_matrix, row_assignments, row_duals, col_duals, obj_val);

		//		cudaMemGetInfo(&free2, &total);

		//		std::cout << "Leakage: " << free1 - free2 << "B" << std::endl;

		double end = omp_get_wtime();

		double total_time = (end - start);

		printDebugArray(obj_val, spcount, "obj_val");

		std::cout << "Itn count: " << stepcounts[3] << "\tOriginal time: " << total_time << " s" << std::endl;

		/*		cudaMemcpy(h_ass, row_assignments, spcount * size * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_row_dual, row_duals, spcount * size * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_col_dual, col_duals, spcount * size * sizeof(double), cudaMemcpyDeviceToHost);

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		modifyCosts(h_cost, spcount, size, size / 10);
		cudaMemcpy(cost_matrix, h_cost, spcount * size * size * sizeof(double), cudaMemcpyHostToDevice);

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		cudaMemcpy(row_assignments, h_ass, spcount * size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(row_duals, h_row_dual, spcount * size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(col_duals, h_col_dual, spcount * size * sizeof(double), cudaMemcpyHostToDevice);

		start = omp_get_wtime();

		CuLAP lpy(problemsize, spcount, devid, false);
		lpy.solve(cost_matrix, row_assignments, row_duals, col_duals, obj_val);

		end = omp_get_wtime();

		printDebugArray(obj_val, spcount, "obj_val");
//		printDebugArray(row_assignments, spcount * size, "assignment");

		total_time = (end - start);

		std::cout << "Re-solve time: " << total_time << " s" << std::endl;

		///////////////////////////////////////////////////////////////////////////////////////////////////////////////

		cudaMemcpy(row_assignments, h_ass, spcount * size * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(row_duals, h_row_dual, spcount * size * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(col_duals, h_col_dual, spcount * size * sizeof(double), cudaMemcpyHostToDevice);

		start = omp_get_wtime();

		CuLAP lpz(problemsize, spcount, devid, true);
		lpz.solve(cost_matrix, row_assignments, row_duals, col_duals, obj_val);

		end = omp_get_wtime();

		total_time = (end - start);

		printDebugArray(obj_val, spcount, "obj_val");
//		printDebugArray(row_assignments, spcount * size, "assignment");

		std::cout << "Dynamic time: " << total_time << " s" << std::endl;
*/
	}

	cudaFree(row_assignments);
	cudaFree(row_duals);
	cudaFree(col_duals);
	cudaFree(cost_matrix);
	cudaFree(obj_val);

	delete[] h_cost;
	delete[] h_ass;
	delete[] h_row_dual;
	delete[] h_col_dual;

	return 0;
}

void modifyCosts(double *cost_matrix, int SP, int N, int mod_count)
{

	CRandomSFMT randomGenerator(SEED);

	for (int i = 0; i < SP; i++)
	{

		for (int j = 0; j < mod_count; j++)
		{

			//double val = randomGenerator.Random();
			double sign = randomGenerator.Random();
			double val = (double)randomGenerator.IRandomX(0, 20);
			double delta = (sign < 0.5) ? -val : val;
			long id = (long)randomGenerator.IRandomX(0, N * N);
			long tid = i * N * N + id;

			cost_matrix[tid] += delta;
		}
	}
}
