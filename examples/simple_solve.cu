#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <random>
#include <omp.h>
#include "culap.h"

#define COSTRANGE 1000
#define PROBLEMCOUNT 1
#define REPETITIONS 1
#define DEVICECOUNT 1

int main(int argc, char **argv) {

	int problemsize = PROBLEMSIZE;
	int costrange = COSTRANGE;
	int problemcount = PROBLEMCOUNT;
	int repetitions = REPETITIONS;
	int batchsize = BATCHSIZE;

	int dev_id = 0;

	cudaSetDevice(dev_id);
	cudaDeviceSynchronize();

	float *h_cost = new float[batchsize * problemsize * problemsize];

	generateProblem(h_cost, batchsize, problemsize, costrange);

		float start = omp_get_wtime();

		// Create an instance of CuLAP using problem size, number of subproblems, and device id.
		CuLAP lpx(problemsize, batchsize, dev_id);

		// Solve LAP(s) for given cost matrix
		lpx.solve(h_cost);
		float end = omp_get_wtime();
		float total_time = (end - start);

		// Use getPrimalObjectiveValue and getDualObjectiveValue APIs to get primal and dual objectives. At optimality both values should match.
		for (int k = 0; k < batchsize; k++) {

			printf("%d:%d:%d:%f:%f:%f\n", j, i, k, lpx.getPrimalObjectiveValue(k), lpx.getDualObjectiveValue(k), total_time);

			// Use getAssignmentVector API to get the optimal row assignments for specified problem id.
			// Assignment vector of the k'th problem -
			int *assignment_sp1 = new int[problemsize];
			lpx.getAssignmentVector(assignment_sp1, k);

			// Use getRowDualVector and getColDualVector API to get the optimal row duals and column duals for specified problem id.
			float *row_dual_sp1 = new float[problemsize];
			lpx.getRowDualVector(row_dual_sp1, k);

		}

	delete[] h_cost;
	return 0;
}
