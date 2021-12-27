/*
 * Created by Ketan Date
 */

#include <iostream>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <cstring>
#include "include/structures.h"
#include "include/variables.h"
#include "include/helper_utils.h"
#include "include/hungarian_algorithm.h"

int main(int argc, char **argv)
{

	int size = atoi(argv[1]);
	int costrange = atoi(argv[2]);
	int problemcount = atoi(argv[3]);
	int repetitions = atoi(argv[4]);
	int multiplier = atoi(argv[5]);
	const char *filename;

	//	int size = PROBLEMSIZE;
	//	int costrange = COSTRANGE;
	//	int problemcount = PROBLEMCOUNT;
	//	int repetitions = REPETITIONS;
	//	const char *filename = INPUTFILE;
	//	int multiplier = 1;

	int init_assignments = 0;
	int stepcounts[7];
	double steptimes[7];

	for (int i = 1; i <= multiplier; i++)
	{
		int problemsize = size * i;
		costrange = problemsize;

		// Printing the output log
		std::stringstream logpath;

		logpath << "log/log_" << problemsize << "_" << costrange << "_normal.txt";

		std::ofstream logfile(logpath.str().c_str());
		logfile << "pr.\tsize\tcost range\tobjective\tinitial assignments\t"
				<< "count 0\tcount 1\tcount 2\tcount 3\tcount 4\tcount 5\tcount 6\t"
				<< "time 0\ttime 1\ttime 2\ttime 3\ttime 4\ttime 5\ttime 6\ttotal time" << std::endl;
		logfile.close();

		std::cout << "Problem size: " << problemsize << "\nCost range: [0, " << costrange << "]"
				  << "\nProblem count: " << problemcount << "\nRepetitions: " << repetitions;

		std::cout << "\nVersion: normal" << std::endl;

		for (int p = 0; p < problemcount; p++)
		{
			if (argc <= 6)
			{
				generateProblem(problemsize, costrange);
			}
			else
			{
				filename = argv[6];
				readFile(filename);
			}

			for (int q = 0; q < repetitions; q++)
			{
				memset(&stepcounts[0], 0, 7 * sizeof(int));
				memset(&steptimes[0], 0, 7 * sizeof(double));

				double start = omp_get_wtime();

				initialize();
				int obj_val = solve(&stepcounts[0], &steptimes[0], init_assignments);
				finalize();

				double end = omp_get_wtime();

				double total_time = (end - start);

				printLog(p + 1, q + 1, costrange, obj_val, init_assignments, total_time, stepcounts, steptimes, logpath.str().c_str());

				std::cout << (p + 1) << "\t" << (q + 1) << "\tObj val: " << obj_val << "\tTotal time: " << total_time << " s" << std::endl;
			}

			delete[] h_costs.elements;
		}
		std::cout << std::endl;
	}
	return 0;
}
