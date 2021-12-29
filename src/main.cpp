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
#include "../include/LinearAssignmentProblem.h"

int main(int argc, char **argv)
{

	int size = atoi(argv[1]);
	int costrange = atoi(argv[2]);
	int problemcount = atoi(argv[3]);
	int repetitions = atoi(argv[4]);

	int numdev = atoi(argv[5]);

	const char *filename = argv[6];

	int multiplier = 1;

	int init_assignments = 0;
	int stepcounts[7];
	double steptimes[9];

	std::stringstream logpath;
	int problemsize = size;

	double *cost_matrix = new double[problemsize * problemsize];

	if (argc <= 6)
	{
		generateProblem(cost_matrix, problemsize, costrange);
	}
	else
	{
		readFile(cost_matrix, filename);
	}

	for (int i = 0; i < repetitions; i++)
	{

		double start = omp_get_wtime();

		double obj_val = 0;
		LinearAssignmentProblem lpx(problemsize, 1);
		lpx.solve(cost_matrix, obj_val);

		double end = omp_get_wtime();

		double total_time = (end - start);

		std::cout << "Obj val: " << obj_val << "\tTotal time: " << total_time << " s" << std::endl;
	}

	delete[] cost_matrix;

	return 0;
}
