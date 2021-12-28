/*
 * Created by Ketan Date
 */

#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include "include/structures.h"
#include "include/variables.h"
#include "include/helper_utils.h"
#include "include/hungarian_algorithm.h"

int main(int argc, char **argv)
{
	cudaSetDevice(0);
	/*
	int size = atoi(argv[1]);
	int costrange = atoi(argv[2]);
	int problemcount = atoi(argv[3]);
	int repetitions = atoi(argv[4]);
	const char *filename;
	*/

	int size = PROBLEMSIZE;
	int costrange = COSTRANGE;
	int problemcount = PROBLEMCOUNT;
	int repetitions = REPETITIONS;
	const char *filename = INPUTFILE;

	int init_assignments = 0;
	int stepcounts[7];
	float steptimes[7];

	for (int i = 1; i <= 1; i++)
	{
		int problemsize = size * i;

		// Printing the output log
		std::stringstream logpath;
		logpath << "log_" << problemsize << "_" << costrange << ".txt";
		std::ofstream logfile(logpath.str().c_str());
		logfile << "pr.\tsize\tcost range\tobjective\tinitial assignments\t"
				<< "count 0\tcount 1\tcount 2\tcount 3\tcount 4\tcount 5\tcount 6\t"
				<< "time 0\ttime 1\ttime 2\ttime 3\ttime 4\ttime 5\ttime 6\ttotal time" << std::endl;
		logfile.close();

		std::cout << "Problem size: " << problemsize << "\nCost range: [0, " << costrange << "]"
				  << "\nProblem count: " << problemcount << "\nRepetitions: " << repetitions << std::endl;

		for (int p = 0; p < problemcount; p++)
		{
			if (argc <= 5)
			{
				generateProblem(problemsize, costrange);
			}
			else
			{
				filename = argv[5];
				readFile(filename);
			}

#ifdef LIGHT
			h_red_costs.rowsize = N;
			h_red_costs.colsize = N;
			h_red_costs.elements = new int[N2];
#endif

			for (int q = 0; q < repetitions; q++)
			{
				memset(&stepcounts[0], 0, 7 * sizeof(int));
				memset(&steptimes[0], 0, 7 * sizeof(float));

				clock_t start = clock();

				initialize();
				int obj_val = solve(&stepcounts[0], &steptimes[0], init_assignments);
				finalize();

				clock_t end = clock();

				int total_time = (end - start) / (CLOCKS_PER_SEC / 1000);

				printLog(p + 1, q + 1, costrange, obj_val, init_assignments, total_time, stepcounts, steptimes, logpath.str().c_str());

				std::cout << (p + 1) << "\t" << (q + 1) << "\tObj val: " << obj_val << "\tTotal time: " << total_time << " ms" << std::endl;
			}

			delete[] h_costs.elements;
#ifdef LIGHT
			delete[] h_red_costs.elements;
#endif
		}
		std::cout << std::endl;
	}
	return 0;
}