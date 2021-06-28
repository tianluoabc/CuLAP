/*
*      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
*      Authors: Ketan Date and Rakesh Nagi
*
*      Article reference:
*	   Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
*
*/

#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <random>
#include <omp.h>
#include "d_structs.h"
#include "d_vars.h"
#include "f_cutils.h"
#include "culap.h"

int main(int argc, char **argv) {

	int problemsize = PROBLEMSIZE;
	int costrange = COSTRANGE;
	int problemcount = PROBLEMCOUNT;
	int repetitions = REPETITIONS;
	int batchsize = BATCHSIZE;

	int devid = 0;

	cudaSetDevice(devid);
	cudaDeviceSynchronize();

	float *h_cost = new float[batchsize * problemsize * problemsize];

	printf("(%d, %d)\n", problemsize, costrange);

	for (int j = 0; j < problemcount; j++) {

		generateProblem(h_cost, batchsize, problemsize, costrange);

		for (int i = 0; i < repetitions; i++) {

			float start = omp_get_wtime();

			// Create an instance of CuLAP using problem size, number of subproblems, and device id.
			CuLAP lpx(problemsize, batchsize, devid);

			// Solve LAP(s) for given cost matrix
			lpx.solve(h_cost);

			float end = omp_get_wtime();

			float total_time = (end - start);

			// Use getPrimalObjectiveValue and getDualObjectiveValue APIs to get primal and dual objectives. At optimality both values should match.
			for (int k = 0; k < batchsize; k++) {

				printf("%d:%d:%d:%f:%f:%f\n", j, i, k, lpx.getPrimalObjectiveValue(k), lpx.getDualObjectiveValue(k), total_time);

			}

//			Use getAssignmentVector API to get the optimal row assignments for specified problem id.
//			Example is shown below.

//			int *assignment_sp1 = new int[problemsize];
//			lpx.getAssignmentVector(assignment_sp1, 15);
//
//			std::cout << "\nPrinting assignment vector for subproblem 1" << std::endl;
//			for (int z = 0; z < problemsize; z++) {
//				std::cout << z << "\t" << assignment_sp1[z] << std::endl;
//			}
//
//			delete[] assignment_sp1;

//			Use getRowDualVector and getColDualVector API to get the optimal row duals and column duals for specified problem id.
//			Example is shown below.

//			float *row_dual_sp1 = new float[problemsize];
//			lpx.getRowDualVector(row_dual_sp1, 15);
//
//			std::cout << "\nPrinting row dual vector for subproblem 1" << std::endl;
//			for (int z = 0; z < problemsize; z++) {
//				std::cout << z << "\t" << row_dual_sp1[z] << std::endl;
//			}
//
//			delete[] row_dual_sp1;
		}
	}

	delete[] h_cost;

	return 0;
}