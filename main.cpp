/*
 * Created by Ketan Date
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <omp.h>
#include "include/structures.h"
#include "include/variables.h"
#include "include/helper_utils.h"
#include "include/sqlap.h"
#include "include/sfmt.h"

#define SEED 24071987

//CRandomSFMT randomGenerator(SEED);

void modifyCosts(double *, int, int, int);

int main(int argc, char **argv)
{

	/*	int size = atoi(argv[1]);
	int costrange = atoi(argv[2]);
	int problemcount = atoi(argv[3]);
	int repetitions = atoi(argv[4]);

	const char *filename = argv[5];*/

	int size = 4;
	double *h_cost = new double[size * size];

	int multiplier = 2;

	int init_assignments = 0;
	int stepcounts[7];
	double steptimes[7];

	std::stringstream filename;
	filename << "file.txt";

	readFile(h_cost, filename.str().c_str());

	double obj_val = 0;
	SqLAP lpx(size);
	lpx.solve(h_cost, obj_val, stepcounts, steptimes, init_assignments);

	delete[] h_cost;

	/*	for (int kk = 1; kk <= multiplier; kk++) {

		int problemsize = kk * size;
		costrange *= kk;

		int *h_cost = new int[problemsize * problemsize];

		std::stringstream logpath;
		logpath << "sequential_tree_hungarian_log_" << problemsize << "_" << costrange << ".txt";

		std::cout << "Problem size: " << problemsize << "\nCost range: [0, " << costrange << "]" << "\nProblem count: " << problemcount << "\nRepetitions: " << repetitions << std::endl;

		std::ofstream logfile(logpath.str().c_str());
		logfile << "pr.\tsize\tcost range\tobjective\tinitial assignments\t" << "count 0\tcount 1\tcount 2\tcount 3\tcount 4\tcount 5\tcount 6\t" << "time 0\ttime 1\ttime 2\ttime 3\ttime 4\ttime 5\ttime 6\ttotal time" << std::endl;
		logfile.close();

		for (int j = 0; j < problemcount; j++) {

//			readFile(h_cost, filename, spcount);
			generateProblem(h_cost, problemsize, costrange);

			for (int i = 0; i < repetitions; i++) {

				init_assignments = 0;
				std::fill(stepcounts, stepcounts + 7, 0);
				std::fill(steptimes, steptimes + 7, 0);

				double start = omp_get_wtime();

				double obj_val = 0;
				SqLAP lpx(problemsize);
				lpx.solve(h_cost, obj_val, stepcounts, steptimes, init_assignments);

				double end = omp_get_wtime();

				double total_time = (end - start);

				printLog(j + 1, i + 1, problemsize, costrange, obj_val, init_assignments, total_time, stepcounts, steptimes, logpath.str().c_str());

				std::cout << (j + 1) << "\t" << (i + 1) << "\tObj val: " << obj_val << "\tTotal time: " << total_time << " s" << std::endl;

			}
		}

		delete[] h_cost;
	} */

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
